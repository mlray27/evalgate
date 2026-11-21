from sqlalchemy import func, select

from evalgate.db import Appointment, Customer, Slot, make_engine, make_sessionmaker
from evalgate.domain import Service
from evalgate.seed import seed
from evalgate.sops import load_library
from evalgate.tools import (
    CheckAvailabilityArgs,
    GetPricingArgs,
    LookupCustomerArgs,
    ProposeBookingArgs,
    SessionState,
    ToolContext,
    check_availability,
    get_pricing,
    lookup_customer,
    propose_booking,
)


def ctx(sm, settings, sid="s1"):
    return ToolContext(SessionState(sid), sm, settings)


async def booked_slots(sm) -> list[str]:
    async with sm() as s:
        return sorted(
            (
                await s.scalars(select(Appointment.slot_id).where(Appointment.status == "booked"))
            ).all()
        )


async def test_seed_is_deterministic(sm, settings, tmp_path):
    first = await booked_slots(sm)
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/other.db")
    other = make_sessionmaker(engine)
    await seed(engine, other, settings.today, settings.seed)
    assert await booked_slots(other) == first
    await engine.dispose()
    async with sm() as s:
        assert await s.scalar(select(func.count()).select_from(Customer)) == 20
        assert await s.get(Appointment, "P9001") is not None


def test_sop_library_loads():
    sops = load_library()
    assert len(sops) >= 20
    assert {"booking-confirmation", "corrections-override", "safety-gas-smell"} <= {
        s.id for s in sops
    }


async def test_lookup_customer_any_phone_format(sm, settings):
    r = await lookup_customer(ctx(sm, settings), LookupCustomerArgs(phone="(555) 0103"))
    assert r["found"] and r["customer_id"] == "C003" and len(r["addresses"]) == 2
    r = await lookup_customer(ctx(sm, settings), LookupCustomerArgs(phone="555-9999"))
    assert r == {"ok": True, "found": False}


async def test_availability_excludes_booked_and_matches_trade(sm, settings):
    r = await check_availability(
        ctx(sm, settings), CheckAvailabilityArgs(service=Service.WATER_HEATER_REPAIR)
    )
    booked = set(await booked_slots(sm))
    async with sm() as s:
        for slot in r["slots"]:
            assert slot["slot_id"] not in booked
            assert (await s.get(Slot, slot["slot_id"])).trade == "plumbing"
            assert not slot["emergency"]


async def test_emergency_maintenance_refused(sm, settings):
    r = await check_availability(
        ctx(sm, settings), CheckAvailabilityArgs(service=Service.HVAC_MAINTENANCE, emergency=True)
    )
    assert r["error"] == "not_allowed"


async def test_pricing_comes_from_policy(sm, settings):
    r = await get_pricing(
        ctx(sm, settings), GetPricingArgs(service=Service.FURNACE_REPAIR, emergency=True)
    )
    assert r["diagnostic_fee"] == 89 and r["emergency_fee"] == 150


async def test_propose_rejects_trade_mismatch_and_foreign_address(sm, settings):
    c = ctx(sm, settings)
    plumbing = (await check_availability(c, CheckAvailabilityArgs(service=Service.DRAIN_CLOG)))[
        "slots"
    ][0]
    r = await propose_booking(
        c,
        ProposeBookingArgs(
            customer_id="C001",
            address_id="A001",
            slot_id=plumbing["slot_id"],
            service=Service.FURNACE_REPAIR,
        ),
    )
    assert r["error"] == "trade_mismatch"
    r = await propose_booking(
        c,
        ProposeBookingArgs(
            customer_id="C001",
            address_id="A002",
            slot_id=plumbing["slot_id"],
            service=Service.DRAIN_CLOG,
        ),
    )
    assert r["error"] == "invalid_address"
    assert c.state.proposals == {}
