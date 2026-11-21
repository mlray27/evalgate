"""Single-agent conversation runtime.

One agent per conversation: full history plus the SOP library in context, no handoffs.
The runtime streams text, executes tool calls through the guarded executor, and records a trace.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import async_sessionmaker

from evalgate.config import Settings
from evalgate.llm import LLM, Completion, TextDelta
from evalgate.sops import AgentVersion
from evalgate.tools import SessionState, ToolContext, ToolExecutor, ToolFaults
from evalgate.trace import Trace

FALLBACK_STUCK = (
    "I'm sorry, I'm having trouble with that. I'll have a team member call you back shortly."
)


def build_system_prompt(version: AgentVersion, settings: Settings) -> str:
    today = settings.today
    procedures = "\n\n".join(
        f"[{s.id}] {s.title}\n{s.text}" for s in sorted(version.sops, key=lambda s: s.priority)
    )
    return (
        f"You are the phone dispatcher for {settings.company_name}, a home heating, cooling, and "
        f"plumbing company. You are talking to a caller in real time. Your goal is to understand "
        f"their problem, keep them safe, and book the right visit when appropriate.\n\n"
        f"Today is {today.strftime('%A %B')} {today.day}, {today.year}.\n\n"
        f"Rules that always apply:\n"
        f"- Use tools for every fact about customers, availability, prices, and appointments. "
        f"Never invent ids, times, or prices.\n"
        f"- Bookings, cancellations, and reschedules need propose_* first, a read-back, the "
        f"caller's agreement in their next message, then confirm_* with the token.\n"
        f"- Procedures are listed by priority; a lower number wins when two conflict.\n\n"
        f"PROCEDURES\n\n{procedures}"
    )


@dataclass
class TurnMetrics:
    turn: int
    ttft_ms: float | None
    total_ms: float
    llm_calls: int
    tool_calls: int
    prompt_tokens: int
    completion_tokens: int
    hit_step_limit: bool


class AgentSession:
    def __init__(
        self,
        version: AgentVersion,
        llm: LLM,
        sm: async_sessionmaker,
        settings: Settings,
        session_id: str | None = None,
        faults: ToolFaults | None = None,
    ) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.version = version
        self.llm = llm
        self.settings = settings
        self.state = SessionState(session_id=self.session_id)
        self.executor = ToolExecutor(ToolContext(self.state, sm, settings), faults)
        self.messages: list[dict] = [
            {"role": "system", "content": build_system_prompt(version, settings)}
        ]
        self.trace = Trace(self.session_id, version.id, llm.model)
        self.metrics: list[TurnMetrics] = []

    @property
    def done(self) -> bool:
        return self.state.ended is not None or self.state.escalated is not None

    async def respond(self, user_text: str) -> AsyncIterator[str]:
        """Handle one caller message; yields the agent's reply as text deltas."""
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        self.state.turn_index += 1
        turn = self.state.turn_index
        self.messages.append({"role": "user", "content": user_text})
        self.trace.add("user", turn=turn, text=user_text)

        ttft: float | None = None
        llm_calls = tool_calls = p_tok = c_tok = 0
        spoken: list[str] = []
        hit_limit = False
        try:
            for _ in range(self.settings.max_tool_steps_per_turn):
                llm_calls += 1
                completion: Completion | None = None
                async for ev in self.llm.stream(self.messages, self.executor.schemas()):
                    if isinstance(ev, TextDelta):
                        if ttft is None:
                            ttft = (loop.time() - t0) * 1000
                        spoken.append(ev.text)
                        yield ev.text
                    else:
                        completion = ev
                assert completion is not None
                p_tok += completion.prompt_tokens
                c_tok += completion.completion_tokens
                self._append_assistant(completion)
                if not completion.tool_calls:
                    break
                for tc in completion.tool_calls:
                    tool_calls += 1
                    out = await self.executor.run(tc.name, tc.arguments)
                    self.trace.add(
                        "tool",
                        turn=turn,
                        name=out.name,
                        kind=out.kind.value,
                        args=out.args,
                        result=out.result,
                        ms=round(out.duration_ms, 1),
                        attempts=out.attempts,
                    )
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(out.result, default=str),
                        }
                    )
            else:
                hit_limit = True
                self.state.escalated = {"reason": "step_limit", "urgency": "normal"}
                self.trace.add("guard", turn=turn, decision="step_limit_escalation")
                self.messages.append({"role": "assistant", "content": FALLBACK_STUCK})
                self.trace.add("assistant", turn=turn, text=FALLBACK_STUCK)
                yield FALLBACK_STUCK
        except asyncio.CancelledError:
            # Barge-in: keep history consistent with what the caller actually heard.
            self._close_dangling_tool_calls()
            partial = "".join(spoken).strip()
            if partial:
                self.messages.append({"role": "assistant", "content": partial + " [interrupted]"})
            self.trace.add("interrupted", turn=turn, heard=partial)
            raise
        finally:
            m = TurnMetrics(
                turn,
                ttft,
                (loop.time() - t0) * 1000,
                llm_calls,
                tool_calls,
                p_tok,
                c_tok,
                hit_limit,
            )
            self.metrics.append(m)
            self.trace.add("turn_metrics", **m.__dict__)

    async def say(self, user_text: str) -> str:
        """Convenience: run a turn and return the full reply."""
        return "".join([d async for d in self.respond(user_text)]).strip()

    def _close_dangling_tool_calls(self) -> None:
        """Every tool call in history needs a result, or the next LLM request is invalid."""
        answered = {m["tool_call_id"] for m in self.messages if m.get("role") == "tool"}
        for m in list(self.messages):
            for tc in m.get("tool_calls") or []:
                if tc["id"] not in answered:
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": json.dumps({"ok": False, "error": "interrupted"}),
                        }
                    )

    def _append_assistant(self, c: Completion) -> None:
        msg: dict = {"role": "assistant", "content": c.text or None}
        if c.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": t.id,
                    "type": "function",
                    "function": {"name": t.name, "arguments": t.arguments},
                }
                for t in c.tool_calls
            ]
        self.messages.append(msg)
        if c.text:
            self.trace.add("assistant", turn=self.state.turn_index, text=c.text.strip())
