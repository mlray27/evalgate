"""OpenAI-compatible client behavior that only shows up against real providers."""

import httpx
import pytest
from openai import BadRequestError, RateLimitError
from openai.types.chat import ChatCompletionChunk

from evalgate.llm import Completion, OpenAICompatLLM


def _chunk(content: str):
    return ChatCompletionChunk.model_validate(
        {
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        }
    )


def _err(cls, status, body=None, headers=None):
    req = httpx.Request("POST", "http://x")
    return cls(
        "err", response=httpx.Response(status, request=req, headers=headers or {}), body=body
    )


def _llm(script, **kw):
    """script: list of exceptions or text replies, consumed one per request."""
    calls = []

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            item = script.pop(0)
            if isinstance(item, Exception):
                raise item

            async def gen():
                yield _chunk(item)

            return gen()

    llm = OpenAICompatLLM("http://x", "k", "test-model", **kw)
    llm.client = type("C", (), {"chat": type("Ch", (), {"completions": Completions()})()})()
    return llm, calls


async def _text(llm):
    return [e for e in [ev async for ev in llm.stream([], [])] if isinstance(e, Completion)][0].text


async def test_waits_out_per_minute_rate_limits():
    llm, calls = _llm(
        [
            _err(RateLimitError, 429, headers={"retry-after": "0.01"}),
            _err(RateLimitError, 429, body={"error": {"details": [{"retryDelay": "0.01s"}]}}),
            "hello",
        ]
    )
    assert await _text(llm) == "hello"
    assert llm.rate_limit_waits == 2 and len(calls) == 3


async def test_gives_up_after_patience_and_names_the_provider():
    llm, _ = _llm(
        [_err(RateLimitError, 429, headers={"retry-after": "1"})] * 10, rate_limit_patience_s=2.5
    )
    with pytest.raises(RateLimitError, match="test-model @ http://x still rate limited"):
        await _text(llm)


async def test_drops_unsupported_options():
    llm, calls = _llm(
        [
            _err(BadRequestError, 400, body="Unrecognized request argument: reasoning_effort"),
            _err(BadRequestError, 400, body="stream_options not supported"),
            "ok",
        ],
        reasoning_effort="low",
    )
    assert await _text(llm) == "ok"
    assert "reasoning_effort" in calls[0] and "reasoning_effort" not in calls[2]
    assert "stream_options" not in calls[2]
