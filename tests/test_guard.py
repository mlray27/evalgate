import asyncio

from sqlalchemy import func, select

from evalgate.db import Appointment
from evalgate.domain import Service
from evalgate.tools import (
    CheckAvailabilityArgs,
    ProposeBookingArgs,
    SessionState,
    TokenArgs,
    ToolContext,
    ToolExecutor,
    ToolFaults,
    check_availability,
    confirm_booking,
    decline_proposal,
    propose_booking,
)


async def count_session_bookings(sm, sid: str) -> int:
    async with sm() as s:
        return await s.scalar(
            select(func.count()).select_from(Appointment).where(Appointment.session_id == sid)
        )


async def stage(sm, settings, sid="s1", service=Service.FURNACE_REPAIR, slot_index=0):
    c = ToolContext(SessionState(sid), sm, settings)
    c.state.turn_index = 1
    slots = (await check_availability(c, CheckAvailabilityArgs(service=service)))["slots"]
    r = await propose_booking(
        c,
        ProposeBookingArgs(
            customer_id="C001",
            address_id="A001",
            slot_id=slots[slot_index]["slot_id"],
            service=service,
        ),
    )
    assert r["ok"], r
    return c, r["token"], slots[slot_index]["slot_id"]


async def test_confirm_in_same_turn_is_denied(sm, settings):
    c, token, _ = await stage(sm, settings)
    r = await confirm_booking(c, TokenArgs(token=token))
    assert r == {"ok": False, "error": "guard_denied", "message": "no_caller_turn_since_proposal"}
    assert await count_session_bookings(sm, "s1") == 0


async def test_confirm_after_caller_turn_books(sm, settings):
    c, token, _ = await stage(sm, settings)
    c.state.turn_index += 1
    r = await confirm_booking(c, TokenArgs(token=token))
    assert r["ok"] and r["appointment_id"].startswith("AP-")
    assert await count_session_bookings(sm, "s1") == 1


async def test_declined_token_cannot_be_confirmed(sm, settings):
    c, token, _ = await stage(sm, settings)
    await decline_proposal(c, TokenArgs(token=token))
    c.state.turn_index += 1
    r = await confirm_booking(c, TokenArgs(token=token))
    assert r["message"] == "caller_declined"


async def test_new_proposal_supersedes_old(sm, settings):
    c, old, _ = await stage(sm, settings)
    slots = (await check_availability(c, CheckAvailabilityArgs(service=Service.FURNACE_REPAIR)))[
        "slots"
    ]
    new = (
        await propose_booking(
            c,
            ProposeBookingArgs(
                customer_id="C001",
                address_id="A001",
                slot_id=slots[1]["slot_id"],
                service=Service.FURNACE_REPAIR,
            ),
        )
    )["token"]
    c.state.turn_index += 1
    assert (await confirm_booking(c, TokenArgs(token=old)))[
        "message"
    ] == "superseded_by_newer_proposal"
    assert (await confirm_booking(c, TokenArgs(token=new)))["ok"]


async def test_stale_proposal_expires(sm, settings):
    c, token, _ = await stage(sm, settings)
    c.state.turn_index += settings.proposal_ttl_turns + 1
    assert (await confirm_booking(c, TokenArgs(token=token)))["message"] == "proposal_expired"


async def test_unknown_token(sm, settings):
    c, _, _ = await stage(sm, settings)
    c.state.turn_index += 1
    assert (await confirm_booking(c, TokenArgs(token="book-99")))["message"] == "unknown_token"


async def test_retry_after_lost_response_books_exactly_once(sm, settings):
    """The confirm succeeds but the response is lost; the model retries. One booking, not two."""
    c, token, _ = await stage(sm, settings)
    c.state.turn_index += 1
    ex = ToolExecutor(c, ToolFaults({"confirm_booking": ["timeout_after", None]}))
    first = await ex.run("confirm_booking", {"token": token})
    assert first.result["error"] == "timeout"
    second = await ex.run("confirm_booking", {"token": token})
    assert second.result["ok"] and second.result["replayed"] is True
    assert await count_session_bookings(sm, "s1") == 1


async def test_concurrent_sessions_cannot_double_book_a_slot(sm, settings):
    a, ta, slot = await stage(sm, settings, sid="sa")
    b, tb, slot_b = await stage(sm, settings, sid="sb")
    assert slot == slot_b
    a.state.turn_index += 1
    b.state.turn_index += 1
    ra, rb = await asyncio.gather(
        confirm_booking(a, TokenArgs(token=ta)), confirm_booking(b, TokenArgs(token=tb))
    )
    assert sorted([ra["ok"], rb["ok"]]) == [False, True]
    loser = ra if not ra["ok"] else rb
    assert loser["error"] == "slot_unavailable"
    async with sm() as s:
        n = await s.scalar(
            select(func.count())
            .select_from(Appointment)
            .where(Appointment.slot_id == slot, Appointment.status == "booked")
        )
    assert n == 1


async def test_invalid_arguments_return_structured_error(sm, settings):
    c, _, _ = await stage(sm, settings)
    ex = ToolExecutor(c)
    out = await ex.run("check_availability", '{"service": "roof_repair"}')
    assert out.result["error"] == "invalid_arguments"
    out = await ex.run("check_availability", "{not json")
    assert out.result["error"] == "invalid_arguments"
    out = await ex.run("launch_missiles", "{}")
    assert out.result["error"] == "unknown_tool"


async def test_read_tool_retries_after_timeout(sm, settings):
    c, _, _ = await stage(sm, settings)
    ex = ToolExecutor(c, ToolFaults({"lookup_customer": ["timeout_before", None]}))
    out = await ex.run("lookup_customer", {"phone": "555-0101"})
    assert out.result["found"] and out.attempts == 2
