"""Deterministic seed data. Every eval run starts from exactly this state."""

import asyncio
import random
from datetime import date, datetime, time, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from evalgate.config import get_settings
from evalgate.db import (
    Address,
    Appointment,
    Customer,
    Slot,
    Technician,
    create_all,
    make_engine,
    make_sessionmaker,
)
from evalgate.domain import SERVICE_TRADE, Service, Trade

CUSTOMERS = [
    # id, name, phone, member, [(line1, city, zip), ...]
    ("C001", "Dana Whitfield", "555-0101", True, [("14 Maple Ave", "Montclair", "07042")]),
    ("C002", "Luis Ortega", "555-0102", False, [("220 Grove St", "Jersey City", "07302")]),
    (
        "C003",
        "Priya Raman",
        "555-0103",
        False,
        [("8 Birch Ln", "Edison", "08820"), ("41 Harbor Rd", "Hoboken", "07030")],
    ),
    ("C004", "Tom Becker", "555-0104", True, [("903 Oak Ridge Dr", "Morristown", "07960")]),
    ("C005", "Grace Kim", "555-0105", False, [("17 Linden Pl", "Fort Lee", "07024")]),
    ("C006", "Marcus Hill", "555-0106", False, [("55 Summit Ave", "Summit", "07901")]),
    (
        "C007",
        "Elena Petrova",
        "555-0107",
        True,
        [("3 Cedar Ct", "Princeton", "08540"), ("120 Main St Apt 4B", "Princeton", "08542")],
    ),
    ("C008", "Sam Okafor", "555-0108", False, [("76 Lake Dr", "Wayne", "07470")]),
    ("C009", "Rachel Stein", "555-0109", False, [("12 Willow Way", "Westfield", "07090")]),
    ("C010", "Jamal Carter", "555-0110", True, [("400 River Rd", "Edgewater", "07020")]),
    ("C011", "Mei Chen", "555-0111", False, [("29 Spruce St", "Metuchen", "08840")]),
    ("C012", "Owen Gallagher", "555-0112", False, [("88 Hillside Ave", "Nutley", "07110")]),
    ("C013", "Fatima Noor", "555-0113", False, [("5 Orchard Rd", "Livingston", "07039")]),
    ("C014", "Brian Walsh", "555-0114", True, [("610 Park Ave", "Hoboken", "07030")]),
    ("C015", "Sofia Rossi", "555-0115", False, [("33 Elm St", "Red Bank", "07701")]),
    ("C016", "Kevin Nguyen", "555-0116", False, [("71 Pine Ter", "Clifton", "07011")]),
    ("C017", "Hannah Brooks", "555-0117", False, [("19 Chestnut St", "Ridgewood", "07450")]),
    ("C018", "Diego Alvarez", "555-0118", True, [("250 Bergen Blvd", "Fairview", "07022")]),
    ("C019", "Aisha Bello", "555-0119", False, [("6 Meadow Ln", "Somerset", "08873")]),
    ("C020", "Peter Lindqvist", "555-0120", False, [("44 Shore Rd", "Asbury Park", "07712")]),
]

TECHNICIANS = [
    ("T01", "Ray", Trade.HVAC),
    ("T02", "Monica", Trade.HVAC),
    ("T03", "Andre", Trade.HVAC),
    ("T04", "Lena", Trade.PLUMBING),
    ("T05", "Victor", Trade.PLUMBING),
]

WINDOWS = [(time(8), time(12)), (time(13), time(17))]
EMERGENCY_WINDOW = (time(18), time(21))
HORIZON_DAYS = 14
PREBOOK_RATE = 0.35


def _slots(today: date) -> list[Slot]:
    slots: list[Slot] = []
    n = 0
    for d in range(HORIZON_DAYS):
        day = today + timedelta(days=d)
        if day.weekday() == 6:  # closed Sundays
            continue
        for tech_id, _, trade in TECHNICIANS:
            for start, end in WINDOWS:
                n += 1
                slots.append(
                    Slot(
                        id=f"S{n:03d}",
                        technician_id=tech_id,
                        trade=trade.value,
                        start=datetime.combine(day, start),
                        end=datetime.combine(day, end),
                        is_emergency=False,
                    )
                )
        # One evening emergency slot per trade per day, held for same-day emergencies.
        for trade, tech_id in ((Trade.HVAC, "T03"), (Trade.PLUMBING, "T05")):
            n += 1
            slots.append(
                Slot(
                    id=f"S{n:03d}",
                    technician_id=tech_id,
                    trade=trade.value,
                    start=datetime.combine(day, EMERGENCY_WINDOW[0]),
                    end=datetime.combine(day, EMERGENCY_WINDOW[1]),
                    is_emergency=True,
                )
            )
    return slots


async def seed(engine: AsyncEngine, sm: async_sessionmaker, today: date, rng_seed: int) -> None:
    await create_all(engine, drop=True)
    rng = random.Random(rng_seed)
    async with sm() as s, s.begin():
        addr_n = 0
        for cid, name, phone, member, addrs in CUSTOMERS:
            s.add(Customer(id=cid, name=name, phone=phone, is_member=member))
            for line1, city, zipc in addrs:
                addr_n += 1
                s.add(
                    Address(id=f"A{addr_n:03d}", customer_id=cid, line1=line1, city=city, zip=zipc)
                )
        for tid, name, trade in TECHNICIANS:
            s.add(Technician(id=tid, name=name, trade=trade.value))
        await s.flush()

        slots = _slots(today)
        s.add_all(slots)
        await s.flush()

        # Pre-book a deterministic fraction of regular slots so availability is realistic.
        services_by_trade = {
            t: [svc for svc, tr in SERVICE_TRADE.items() if tr == t] for t in Trade
        }
        appt_n = 0
        for slot in slots:
            if slot.is_emergency or rng.random() > PREBOOK_RATE:
                continue
            cid = rng.choice(CUSTOMERS[5:])[0]  # keep C001-C005 free of history
            cust = await s.get(Customer, cid)
            appt_n += 1
            s.add(
                Appointment(
                    id=f"P{appt_n:04d}",
                    customer_id=cid,
                    address_id=cust.addresses[0].id,
                    slot_id=slot.id,
                    service=rng.choice(services_by_trade[Trade(slot.trade)]).value,
                    status="booked",
                )
            )

        # Known history for scenarios: C003 has an upcoming water heater visit to reschedule.
        free = (
            await s.scalars(
                select(Slot).where(Slot.trade == Trade.PLUMBING.value, Slot.is_emergency.is_(False))
            )
        ).all()
        booked = {a.slot_id for a in s.new if isinstance(a, Appointment)}
        target = next(
            sl for sl in free if sl.id not in booked and sl.start.date() > today + timedelta(days=2)
        )
        s.add(
            Appointment(
                id="P9001",
                customer_id="C003",
                address_id="A003",
                slot_id=target.id,
                service=Service.WATER_HEATER_REPAIR.value,
                status="booked",
            )
        )


async def seed_from_settings() -> None:
    st = get_settings()
    engine = make_engine(st.database_url)
    await seed(engine, make_sessionmaker(engine), st.today, st.seed)
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(seed_from_settings())
    print("seeded")
