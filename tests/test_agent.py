"""End-to-end runtime tests with a scripted model: full conversations, no network."""

import asyncio

from sqlalchemy import func, select

from evalgate.agent import FALLBACK_STUCK, AgentSession, build_system_prompt
from evalgate.db import Appointment
from evalgate.llm import ScriptedLLM, Step, last_tool_result
from evalgate.tools import ToolFaults


def first_slot(messages):
    return last_tool_result(messages, "check_availability")["slots"][0]["slot_id"]


def token(messages):
    return {"token": last_tool_result(messages, "propose_booking")["token"]}


def booking_script(retry_confirm: bool = False) -> list:
    steps = [
        # turn 1: caller reports no heat
        Step(calls=[("lookup_customer", {"phone": "555-0101"})]),
        Step(
            calls=[("check_availability", {"service": "furnace_repair", "time_of_day": "morning"})]
        ),
        lambda m: Step(
            calls=[
                (
                    "propose_booking",
                    {
                        "customer_id": "C001",
                        "address_id": "A001",
                        "slot_id": first_slot(m),
                        "service": "furnace_repair",
                        "notes": "no heat since last night",
                    },
                )
            ]
        ),
        Step(
            text="I can book a furnace repair at 14 Maple Ave on the first morning. Does that work?"
        ),
        # turn 2: caller agrees
        lambda m: Step(calls=[("confirm_booking", token(m))]),
    ]
    if retry_confirm:
        steps.append(lambda m: Step(calls=[("confirm_booking", token(m))]))
    steps.append(Step(text="You're all set."))
    return steps


async def session_bookings(sm, sid):
    async with sm() as s:
        return await s.scalar(
            select(func.count()).select_from(Appointment).where(Appointment.session_id == sid)
        )


async def test_system_prompt_contains_procedures_by_priority(version, settings):
    p = build_system_prompt(version, settings)
    assert "Monday January 12, 2026" in p
    assert p.index("[safety-gas-smell]") < p.index("[tone]")


async def test_happy_path_booking(sm, settings, version):
    sess = AgentSession(version, ScriptedLLM(booking_script()), sm, settings)
    reply = await sess.say("Hi, my heat's out. My number is 555-0101.")
    assert "Does that work?" in reply
    assert await session_bookings(sm, sess.session_id) == 0  # nothing booked before agreement
    reply = await sess.say("Yes, that works.")
    assert reply == "You're all set."
    assert await session_bookings(sm, sess.session_id) == 1
    names = [e["name"] for e in sess.trace.of_type("tool")]
    assert names == [
        "lookup_customer",
        "check_availability",
        "propose_booking",
        "confirm_booking",
    ]


async def test_model_confirming_before_caller_agrees_is_blocked(sm, settings, version):
    steps = booking_script()
    # The model skips the read-back and confirms in the same turn.
    steps[3] = lambda m: Step(calls=[("confirm_booking", token(m))])
    steps.insert(4, Step(text="Let me confirm that with you first. Does the first morning work?"))
    sess = AgentSession(version, ScriptedLLM(steps), sm, settings)
    await sess.say("My heat's out, 555-0101.")
    denied = sess.trace.of_type("tool")[-1]
    assert denied["name"] == "confirm_booking"
    assert denied["result"]["message"] == "no_caller_turn_since_proposal"
    assert await session_bookings(sm, sess.session_id) == 0


async def test_timeout_then_retry_creates_one_booking(sm, settings, version):
    faults = ToolFaults({"confirm_booking": ["timeout_after"]})
    sess = AgentSession(
        version, ScriptedLLM(booking_script(retry_confirm=True)), sm, settings, faults=faults
    )
    await sess.say("Heat's out, 555-0101.")
    await sess.say("Yes please.")
    confirms = [e for e in sess.trace.of_type("tool") if e["name"] == "confirm_booking"]
    assert confirms[0]["result"]["error"] == "timeout"
    assert confirms[1]["result"]["replayed"] is True
    assert await session_bookings(sm, sess.session_id) == 1


async def test_step_limit_escalates_instead_of_looping(sm, settings, version):
    # The model keeps calling the wrong tool, like the cancellation specialist that looped 25 times.
    looping = [Step(calls=[("list_appointments", {"customer_id": "C999"})])] * 10
    sess = AgentSession(version, ScriptedLLM(looping), sm, settings)
    reply = await sess.say("I need to cancel.")
    assert reply == FALLBACK_STUCK
    assert sess.state.escalated["reason"] == "step_limit"
    assert len(sess.trace.of_type("tool")) == settings.max_tool_steps_per_turn


async def test_barge_in_keeps_history_valid(sm, settings, version):
    class SlowLLM(ScriptedLLM):
        async def stream(self, messages, tools):
            async for ev in super().stream(messages, tools):
                yield ev
                await asyncio.sleep(0.05)

    sess = AgentSession(
        version,
        SlowLLM(
            [
                Step(
                    text="Okay so the first thing I would like to ask is about your furnace model"
                ),
                Step(text="Got it, the water heater."),
            ]
        ),
        sm,
        settings,
    )

    async def consume():
        async for _ in sess.respond("My furnace is broken"):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.12)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert sess.messages[-1]["content"].endswith("[interrupted]")
    assert sess.trace.of_type("interrupted")
    assert await sess.say("Wait, I meant the water heater") == "Got it, the water heater."


async def test_provider_extra_fields_on_tool_calls_are_echoed_back(sm, settings, version):
    """Gemini requires each tool call's thought signature to be sent back on the next request."""
    from openai.types.chat import ChatCompletionChunk

    from evalgate.llm import OpenAICompatLLM

    def chunk(delta):
        return ChatCompletionChunk.model_validate(
            {
                "id": "x",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "m",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
        )

    sig = {"google": {"thought_signature": "SIG123"}}
    responses = [
        [
            chunk(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "c1",
                            "type": "function",
                            "extra_content": sig,
                            "function": {
                                "name": "lookup_customer",
                                "arguments": '{"phone": "555-0101"}',
                            },
                        }
                    ],
                }
            )
        ],
        [chunk({"content": "Hi Dana!"})],
    ]
    sent = []

    class FakeCompletions:
        async def create(self, **kw):
            sent.append(kw["messages"])

            async def gen():
                for c in responses.pop(0):
                    yield c

            return gen()

    llm = OpenAICompatLLM("http://x", "k", "gemini-test")
    llm.client = type("C", (), {"chat": type("Ch", (), {"completions": FakeCompletions()})()})()
    sess = AgentSession(version, llm, sm, settings)
    assert await sess.say("Hi, 555-0101") == "Hi Dana!"
    replayed = [m for m in sent[1] if m.get("tool_calls")][0]["tool_calls"][0]
    assert replayed["extra_content"] == sig
