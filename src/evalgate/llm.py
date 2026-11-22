"""LLM interface: one streaming call that yields text deltas, then a final completion.

OpenAICompatLLM works with any OpenAI-compatible endpoint (Together, Fireworks, Groq, vLLM).
ScriptedLLM replays canned turns so the whole runtime is testable without network or keys.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, exactly as the model produced it
    # Provider-specific data that must be sent back unchanged with this call in later requests.
    # Gemini puts a required "thought signature" here; dropping it makes the next request a 400.
    extra: dict | None = None


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

    Rate limits: free tiers have per-minute limits that the SDK's own short retries can't wait
    out. On a 429 we wait as long as the provider asks (or up to a minute) and keep trying for
    `rate_limit_patience_s`. If it is still limited after that, it is almost certainly a daily
    quota, and openai.RateLimitError propagates (the eval runner then stops cleanly).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        timeout: float = 30,
        max_retries: int = 2,
        reasoning_effort: str | None = None,
        rate_limit_patience_s: float = 300,
        rate_limit_wait_cap_s: float = 60,
    ) -> None:
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries
        )
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort or None
        self.patience = rate_limit_patience_s
        self.wait_cap = rate_limit_wait_cap_s
        self._usage_supported = True
        self.rate_limit_waits = 0

    @staticmethod
    def _suggested_wait(err) -> float | None:
        """Seconds the provider asked us to wait, from headers or a Gemini-style body."""
        resp = getattr(err, "response", None)
        headers = getattr(resp, "headers", None) or {}
        for h in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
            v = headers.get(h)
            if v:
                m = re.match(r"^([\d.]+)(ms|s|m)?", str(v).strip())
                if m:
                    n = float(m.group(1))
                    return n / 1000 if m.group(2) == "ms" else n * 60 if m.group(2) == "m" else n
        m = re.search(r'retryDelay["\']?\s*[:=]\s*["\']?([\d.]+)s', str(getattr(err, "body", "")))
        return float(m.group(1)) if m else None

    async def _create(self, messages: list[dict], tools: list[dict]):
        from openai import RateLimitError

        waited = 0.0
        attempt = 0
        while True:
            try:
                return await self._create_once(messages, tools)
            except RateLimitError as e:
                attempt += 1
                wait = self._suggested_wait(e) or min(10 * 2 ** (attempt - 1), self.wait_cap)
                wait = min(max(wait, 1.0), self.wait_cap)
                if waited + wait > self.patience:
                    raise RateLimitError(
                        f"{self.model} @ {self.base_url} still rate limited after {waited:.0f}s "
                        f"(likely a daily quota): {getattr(e, 'body', None) or e}",
                        response=e.response,
                        body=e.body,
                    ) from e
                self.rate_limit_waits += 1
                await asyncio.sleep(wait)
                waited += wait

    async def _create_once(self, messages: list[dict], tools: list[dict]):
        from openai import BadRequestError

        kwargs = dict(
            model=self.model,
            messages=messages,
            tools=tools or None,
            temperature=self.temperature,
            stream=True,
        )
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self._usage_supported:
            kwargs["stream_options"] = {"include_usage": True}
        try:
            return await self.client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            msg = f"{e} {getattr(e, 'body', '')}"
            if "stream_options" in msg and self._usage_supported:
                self._usage_supported = False  # some providers reject it; skip token counts
            elif "reasoning_effort" in msg and self.reasoning_effort:
                self.reasoning_effort = None  # not supported by this model; run without it
            else:
                raise
            return await self._create_once(messages, tools)

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
                slot = calls.setdefault(idx, {"id": "", "name": "", "arguments": "", "extra": None})
                extra = (tc.model_extra or {}).get("extra_content")
                if extra:
                    slot["extra"] = extra
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
        yield Completion(
            text="".join(text),
            tool_calls=[
                ToolCall(c["id"] or f"call_{i}", c["name"], c["arguments"] or "{}", c["extra"])
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
