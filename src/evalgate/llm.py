"""LLM interface: one streaming call that yields text deltas, then a final completion.

OpenAICompatLLM works with any OpenAI-compatible endpoint (Together, Fireworks, Groq, vLLM).
ScriptedLLM replays canned turns so the whole runtime is testable without network or keys.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, exactly as the model produced it


@dataclass
class TextDelta:
    text: str


@dataclass
class Completion:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


LLMEvent = TextDelta | Completion


class LLM(Protocol):
    model: str

    def stream(self, messages: list[dict], tools: list[dict]) -> AsyncIterator[LLMEvent]: ...


class OpenAICompatLLM:
    """Any OpenAI-compatible endpoint: Together, Groq, Gemini, Fireworks, vLLM, Ollama.

    Rate limits (429) are retried with backoff by the SDK, honoring retry-after headers. If they
    persist (for example a daily quota), openai.RateLimitError propagates to the caller.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        timeout: float = 30,
        max_retries: int = 6,
    ) -> None:
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries
        )
        self.model = model
        self.temperature = temperature
        self._usage_supported = True

    async def _create(self, messages: list[dict], tools: list[dict]):
        from openai import BadRequestError

        kwargs = dict(
            model=self.model,
            messages=messages,
            tools=tools or None,
            temperature=self.temperature,
            stream=True,
        )
        if self._usage_supported:
            try:
                return await self.client.chat.completions.create(
                    **kwargs, stream_options={"include_usage": True}
                )
            except BadRequestError as e:
                if "stream_options" not in str(e):
                    raise
                self._usage_supported = False  # some providers reject it; skip token counts
        return await self.client.chat.completions.create(**kwargs)

    async def stream(self, messages: list[dict], tools: list[dict]) -> AsyncIterator[LLMEvent]:
        resp = await self._create(messages, tools)
        text: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        usage = None
        async for chunk in resp:
            if chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                text.append(delta.content)
                yield TextDelta(delta.content)
            for tc in delta.tool_calls or []:
                # Some providers omit `index` and send each call whole; key those by id.
                idx = tc.index
                if idx is None:
                    idx = next(
                        (i for i, c in calls.items() if tc.id and c["id"] == tc.id), len(calls)
                    )
                slot = calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
        yield Completion(
            text="".join(text),
            tool_calls=[
                ToolCall(c["id"] or f"call_{i}", c["name"], c["arguments"] or "{}")
                for i, c in sorted(calls.items())
            ],
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


@dataclass
class Step:
    """One scripted model response."""

    text: str = ""
    calls: list[tuple[str, dict | str]] = field(default_factory=list)


class ScriptedLLM:
    """Replays a fixed list of Steps, or computes each step from the transcript.

    A step's tool-call arguments may be a callable(messages) -> dict, so a script can reference
    values (tokens, ids) produced by earlier tool results.
    """

    model = "scripted"

    def __init__(self, steps: list[Step | Callable[[list[dict]], Step]]) -> None:
        self.steps = list(steps)
        self.calls_made = 0

    async def stream(self, messages: list[dict], tools: list[dict]) -> AsyncIterator[LLMEvent]:
        if not self.steps:
            raise RuntimeError("ScriptedLLM ran out of steps")
        step = self.steps.pop(0)
        if callable(step):
            step = step(messages)
        self.calls_made += 1
        for word in step.text.split(" ") if step.text else []:
            yield TextDelta(word + " ")
        calls = []
        for i, (name, args) in enumerate(step.calls):
            if callable(args):
                args = args(messages)
            raw = args if isinstance(args, str) else json.dumps(args)
            calls.append(ToolCall(f"call_{self.calls_made}_{i}", name, raw))
        yield Completion(text=step.text, tool_calls=calls)


def last_tool_result(messages: list[dict], tool_name: str | None = None) -> dict:
    """Helper for scripts: the most recent tool result (optionally for a given tool)."""
    names = {}
    for m in messages:
        for tc in m.get("tool_calls") or []:
            names[tc["id"]] = tc["function"]["name"]
    for m in reversed(messages):
        if m.get("role") == "tool" and (
            tool_name is None or names.get(m["tool_call_id"]) == tool_name
        ):
            return json.loads(m["content"])
    raise KeyError(tool_name)
