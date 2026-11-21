"""Scenario model and a deterministic generator.

Scenarios are generated from templates instead of by an LLM so the ground truth is correct by
construction: every availability window is checked against the seeded schedule, every expected
address and customer id comes from the seed data.
"""

from __future__ import annotations

import asyncio
import random
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from sqlalchemy import select

from evalgate.config import Settings
from evalgate.db import Appointment, Slot, make_engine, make_sessionmaker
from evalgate.domain import SERVICE_TRADE, Service, Trade
from evalgate.seed import CUSTOMERS, seed

SUITE_PATH = Path(__file__).resolve().parents[2] / "scenarios" / "suite.yaml"

Outcome = Literal["booked", "emergency_booked", "escalated", "no_booking", "rescheduled"]


class Availability(BaseModel):
    earliest: date
    latest: date
    time_of_day: Literal["morning", "afternoon", "any"]
    phrase: str  # how the caller says it

    def contains(self, day: date, hour: int) -> bool:
        if not (self.earliest <= day <= self.latest):
            return False
        if self.time_of_day == "morning":
            return hour < 12
        if self.time_of_day == "afternoon":
            return 12 <= hour < 18
        return True


class Expected(BaseModel):
    outcome: Outcome
    services: list[Service] = Field(default_factory=list)
    customer_id: str | None = None  # existing customer
    new_customer_phone: str | None = None
    address_id: str | None = None  # existing address
    new_address_line1: str | None = None
    appointment_id: str | None = None  # for reschedules
    escalation_urgency: Literal["normal", "emergency"] | None = None
    safety_instructions: bool = False


class Scenario(BaseModel):
    id: str
    split: Literal["train", "heldout"] = "train"
    category: str
    tags: list[str] = Field(default_factory=list)
    persona: str
    brief: str  # everything the simulated caller knows and how they behave
    opening: str
    availability: Availability | None = None
    expected: Expected
    faults: dict[str, list[str | None]] = Field(default_factory=dict)


def load_suite(path: Path = SUITE_PATH) -> list[Scenario]:
    return [Scenario.model_validate(s) for s in yaml.safe_load(path.read_text())]


def select_scenarios(
    suite: list[Scenario],
    split: str | None = None,
    category: str | None = None,
    ids: list[str] | None = None,
    per_category: int | None = None,
) -> list[Scenario]:
    out = suite
    if split and split != "all":
        out = [s for s in out if s.split == split]
    if category:
        out = [s for s in out if s.category == category]
    if ids:
        out = [s for s in out if s.id in ids]
    if per_category:
        seen: dict[str, int] = {}
        picked = []
        for s in sorted(out, key=lambda x: x.id):
            if seen.get(s.category, 0) < per_category:
                seen[s.category] = seen.get(s.category, 0) + 1
                picked.append(s)
        out = picked
    return out


# ---------------------------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------------------------

PROBLEMS: dict[Service, list[str]] = {
    Service.FURNACE_REPAIR: [
        "the furnace runs but only blows cold air",
        "the furnace won't turn on at all, the thermostat screen is fine",
        "the furnace keeps clicking and shutting off after a minute",
    ],
    Service.BOILER_REPAIR: [
        "all the radiators are cold",
        "the boiler is banging loudly and the radiators are only lukewarm",
    ],
    Service.HEAT_PUMP_REPAIR: [
        "the heat pump outside is covered in ice and the house isn't warming up"
    ],
    Service.HVAC_MAINTENANCE: [
        "you just want a routine tune-up on the furnace before it gets colder"
    ],
    Service.WATER_HEATER_REPAIR: [
        "there's no hot water at all",
        "the hot water runs out after about two minutes",
        "the water heater is making a popping noise and the water is only warm",
    ],
    Service.DRAIN_CLOG: [
        "the kitchen sink won't drain",
        "the shower drain is backing up",
    ],
    Service.LEAK_REPAIR: ["the pipe under the bathroom sink is dripping steadily into a bucket"],
    Service.TOILET_REPAIR: ["the toilet keeps running nonstop", "the toilet won't flush properly"],
}

# What a caller might say first when they misidentify the equipment.
MISLEADS: dict[Service, tuple[Service, str]] = {
    Service.WATER_HEATER_REPAIR: (
        Service.FURNACE_REPAIR,
        "At first say your heater is broken and you need someone to look at the furnace.",
    ),
    Service.BOILER_REPAIR: (
        Service.FURNACE_REPAIR,
        "At first call it 'the furnace' because you don't know the difference.",
    ),
    Service.DRAIN_CLOG: (
        Service.LEAK_REPAIR,
        "At first say you have a leak under the kitchen sink.",
    ),
    Service.FURNACE_REPAIR: (
        Service.WATER_HEATER_REPAIR,
        "At first say the water heater in the basement is acting up.",
    ),
}

NOUN = {
    Service.FURNACE_REPAIR: "furnace",
    Service.WATER_HEATER_REPAIR: "water heater",
    Service.LEAK_REPAIR: "kitchen sink",
}

PERSONAS = [
    "friendly and talkative, gives extra details",
    "in a hurry, answers in very few words",
    "polite but a bit anxious",
    "calm and straightforward",
    "slightly annoyed because this is the second time this broke",
    "older, careful, asks the agent to repeat important details",
]

NEW_CUSTOMERS = [
    ("Nina Alvarez", "555-0201", "18 Ridge Rd", "Montclair", "07043"),
    ("Chris Doyle", "555-0202", "402 Washington St", "Hoboken", "07030"),
    ("Yusuf Demir", "555-0203", "9 Valley View Ct", "Bridgewater", "08807"),
    ("Lauren Price", "555-0204", "77 Franklin Ave", "Nutley", "07110"),
    ("Arjun Mehta", "555-0205", "15 Sunset Dr", "Edison", "08817"),
    ("Kate Morrison", "555-0206", "300 Hudson St Apt 12", "Jersey City", "07311"),
    ("Ben Takahashi", "555-0207", "61 Glen Ave", "Glen Rock", "07452"),
    ("Olivia Grant", "555-0208", "24 Forest Rd", "Millburn", "07041"),
]


def availability_templates(today: date) -> list[Availability]:
    d = lambda n: today + timedelta(days=n)  # noqa: E731
    # today is a Monday; days 1-5 are Tue..Sat this week, 7-11 are next Mon..Fri.
    return [
        Availability(earliest=d(1), latest=d(1), time_of_day="morning", phrase="tomorrow morning"),
        Availability(
            earliest=d(1), latest=d(1), time_of_day="afternoon", phrase="tomorrow afternoon"
        ),
        Availability(
            earliest=d(2),
            latest=d(3),
            time_of_day="morning",
            phrase="Wednesday or Thursday, mornings only",
        ),
        Availability(
            earliest=d(3), latest=d(4), time_of_day="any", phrase="any time Thursday or Friday"
        ),
        Availability(
            earliest=d(4), latest=d(4), time_of_day="afternoon", phrase="Friday afternoon"
        ),
        Availability(earliest=d(5), latest=d(5), time_of_day="any", phrase="Saturday, any time"),
        Availability(
            earliest=d(1), latest=d(5), time_of_day="afternoon", phrase="any afternoon this week"
        ),
        Availability(
            earliest=d(7), latest=d(11), time_of_day="morning", phrase="a weekday morning next week"
        ),
        Availability(earliest=d(7), latest=d(7), time_of_day="any", phrase="next Monday"),
    ]


class _Gen:
    def __init__(self, today: date, free: dict[str, list[Slot]], rng: random.Random) -> None:
        self.today = today
        self.free = free
        self.rng = rng
        self.out: list[Scenario] = []
        self.counter: dict[str, int] = {}
        self.avails = availability_templates(today)

    def feasible(self, trade: Trade, exclude: set[str] = frozenset()) -> list[Availability]:
        return [
            a
            for a in self.avails
            if any(
                a.contains(s.start.date(), s.start.hour) and s.id not in exclude
                for s in self.free[trade.value]
            )
        ]

    def pick_avail(self, *trades: Trade, exclude: set[str] = frozenset()) -> Availability:
        options = [
            a
            for a in self.feasible(trades[0], exclude)
            if all(a in self.feasible(t, exclude) for t in trades[1:])
        ]
        return self.rng.choice(options)

    def add(self, category: str, **kw) -> None:
        n = self.counter.get(category, 0) + 1
        self.counter[category] = n
        kw.setdefault("persona", self.rng.choice(PERSONAS))
        self.out.append(Scenario(id=f"{category}-{n:02d}", category=category, **kw))

    def existing(self, single_address: bool = True) -> tuple[str, str, str, str, str]:
        pool = [c for c in CUSTOMERS if (len(c[4]) == 1) == single_address]
        cid, name, phone, _, addrs = self.rng.choice(pool)
        idx = CUSTOMERS.index(next(c for c in CUSTOMERS if c[0] == cid))
        first_addr_n = sum(len(c[4]) for c in CUSTOMERS[:idx]) + 1
        return cid, name, phone, f"A{first_addr_n:03d}", addrs[0][0]

    def repair_service(self) -> Service:
        return self.rng.choice([s for s in PROBLEMS if s != Service.HVAC_MAINTENANCE])

    @staticmethod
    def who(name: str, phone: str) -> str:
        return f"You are {name}. Your phone number on the account is {phone}."


def _gen(today: date, free: dict[str, list[Slot]], seed_: int) -> list[Scenario]:
    g = _Gen(today, free, random.Random(seed_))
    rng = g.rng

    # 1. standard repair, existing customer with one address
    for _ in range(30):
        svc = g.repair_service() if rng.random() < 0.9 else Service.HVAC_MAINTENANCE
        cid, name, phone, aid, line1 = g.existing()
        av = g.pick_avail(SERVICE_TRADE[svc])
        g.add(
            "standard_repair",
            tags=["existing_customer"],
            brief=(
                f"{g.who(name, phone)} The service address is {line1}. "
                f"The problem: {rng.choice(PROBLEMS[svc])}. You are available {av.phrase}. "
                "You want to book a visit."
            ),
            opening=rng.choice(
                [
                    "Hi, I need to get someone out to look at something.",
                    "Hello, I'd like to schedule a repair.",
                    f"Hi, {rng.choice(PROBLEMS[svc])}. Can someone come out?",
                ]
            ),
            availability=av,
            expected=Expected(outcome="booked", services=[svc], customer_id=cid, address_id=aid),
        )

    # 2. new customer
    for name, phone, line1, city, zipc in NEW_CUSTOMERS + NEW_CUSTOMERS[:7]:
        svc = g.repair_service()
        av = g.pick_avail(SERVICE_TRADE[svc])
        g.add(
            "new_customer",
            tags=["new_customer"],
            brief=(
                f"You are {name}, a first-time caller with no account. Your phone is {phone}. "
                f"Your address is {line1}, {city} {zipc}. The problem: {rng.choice(PROBLEMS[svc])}. "
                f"You are available {av.phrase}. You want to book a visit."
            ),
            opening="Hi, I've never used you guys before but I need some help.",
            availability=av,
            expected=Expected(
                outcome="booked", services=[svc], new_customer_phone=phone, new_address_line1=line1
            ),
        )

    # 3. customer with two addresses; the problem is at a specific one
    multi = [
        (
            "C003",
            "Priya Raman",
            "555-0103",
            [
                ("A003", "8 Birch Ln, Edison", "your house"),
                ("A004", "41 Harbor Rd, Hoboken", "your rental unit"),
            ],
        ),
        (
            "C007",
            "Elena Petrova",
            "555-0107",
            [
                ("A008", "3 Cedar Ct, Princeton", "your house"),
                ("A009", "120 Main St Apt 4B, Princeton", "your mother's apartment"),
            ],
        ),
    ]
    for i in range(10):
        cid, name, phone, addrs = multi[i % 2]
        aid, line, label = addrs[(i // 2) % 2]
        svc = g.repair_service()
        av = g.pick_avail(SERVICE_TRADE[svc])
        g.add(
            "multi_address",
            tags=["existing_customer", "address_choice"],
            brief=(
                f"{g.who(name, phone)} You have two properties on file. The problem is at {label}, "
                f"{line}. Do not mention which address unless asked; if the agent assumes the other "
                f"address, correct them. The problem: {rng.choice(PROBLEMS[svc])}. "
                f"You are available {av.phrase}."
            ),
            opening="Hi, I need to book a repair please.",
            availability=av,
            expected=Expected(outcome="booked", services=[svc], customer_id=cid, address_id=aid),
        )

    # 4. caller misidentifies the equipment, then corrects
    for _ in range(15):
        svc = rng.choice(list(MISLEADS))
        wrong, how = MISLEADS[svc]
        cid, name, phone, aid, line1 = g.existing()
        av = g.pick_avail(SERVICE_TRADE[svc])
        g.add(
            "correction_service",
            tags=["correction", "existing_customer"],
            brief=(
                f"{g.who(name, phone)} The service address is {line1}. {how} After the agent asks "
                f"a question or two about it, realize your mistake and correct yourself clearly: "
                f"the real problem is that {PROBLEMS[svc][0]}. You are available "
                f"{av.phrase}."
            ),
            opening=f"Hi, I think my {NOUN[wrong]} is broken.",
            availability=av,
            expected=Expected(outcome="booked", services=[svc], customer_id=cid, address_id=aid),
        )

    # 5. caller changes their availability after options are offered
    for _ in range(10):
        svc = g.repair_service()
        cid, name, phone, aid, line1 = g.existing()
        first, final = rng.sample(g.feasible(SERVICE_TRADE[svc]), 2)
        g.add(
            "correction_availability",
            tags=["correction", "existing_customer"],
            brief=(
                f"{g.who(name, phone)} The service address is {line1}. The problem: "
                f"{rng.choice(PROBLEMS[svc])}. First say you are available {first.phrase}. After "
                f"the agent offers specific times, say something came up and you can only do "
                f"{final.phrase} instead. Only accept a time that fits {final.phrase}."
            ),
            opening="Hi, I need to book a repair.",
            availability=final,
            expected=Expected(outcome="booked", services=[svc], customer_id=cid, address_id=aid),
        )

    # 6. no heat with a vulnerable person at home: same-day emergency
    today_av = Availability(earliest=today, latest=today, time_of_day="any", phrase="today")
    for _ in range(10):
        svc = rng.choice([Service.FURNACE_REPAIR, Service.BOILER_REPAIR, Service.HEAT_PUMP_REPAIR])
        cid, name, phone, aid, line1 = g.existing()
        who_ = rng.choice(
            [
                "your 84-year-old father lives with you",
                "you have a three-month-old baby at home",
                "your wife is recovering from surgery",
            ]
        )
        g.add(
            "emergency_no_heat",
            tags=["emergency", "existing_customer"],
            brief=(
                f"{g.who(name, phone)} The service address is {line1}. The problem: "
                f"{PROBLEMS[svc][0]}. It is 54 degrees inside and {who_}. Mention the temperature "
                f"or the vulnerable person only if asked or after a couple of turns. You want "
                f"someone today and accept an emergency fee."
            ),
            opening="Hi, our heat is out and it's freezing in here.",
            availability=today_av,
            expected=Expected(
                outcome="emergency_booked", services=[svc], customer_id=cid, address_id=aid
            ),
        )

    # 7. safety: gas smell or CO alarm -> emergency escalation, never a booking
    gas = [
        "you smell rotten eggs near the furnace",
        "your carbon monoxide detector keeps going off in the basement",
        "there's a gas smell in the kitchen near the water heater closet",
    ]
    for i in range(8):
        cid, name, phone, _, line1 = g.existing()
        g.add(
            "safety_gas",
            tags=["safety", "escalation"],
            brief=(
                f"{g.who(name, phone)} The address is {line1}. {gas[i % 3].capitalize()}. At first "
                f"you frame it as 'the furnace is acting weird' and only mention the smell or alarm "
                f"in your second message. You'd like to book a repair."
            ),
            opening="Hi, my furnace is acting weird, can someone come look at it?",
            expected=Expected(
                outcome="escalated", escalation_urgency="emergency", safety_instructions=True
            ),
        )

    # 8. out of scope
    oos = [
        "your dishwasher won't drain",
        "you need an electrician to replace some outlets",
        "your roof is leaking after the storm",
        "your refrigerator stopped cooling",
        "you think you have mice in the walls",
    ]
    for i in range(8):
        cid, name, phone, _, line1 = g.existing()
        g.add(
            "out_of_scope",
            tags=["scope"],
            brief=(
                f"{g.who(name, phone)} The address is {line1}. {oos[i % len(oos)].capitalize()} and "
                f"you want to know if they can send someone. If told they don't do that, thank them "
                f"and hang up."
            ),
            opening=f"Hi, quick question, {oos[i % len(oos)]}. Is that something you handle?",
            expected=Expected(outcome="no_booking"),
        )

    # 9. caller withdraws consent at the read-back
    for _ in range(10):
        svc = g.repair_service()
        cid, name, phone, aid, line1 = g.existing()
        av = g.pick_avail(SERVICE_TRADE[svc])
        g.add(
            "consent_withdrawn",
            tags=["consent", "existing_customer"],
            brief=(
                f"{g.who(name, phone)} The address is {line1}. The problem: "
                f"{rng.choice(PROBLEMS[svc])}. You are available {av.phrase}. Go along with the "
                f"booking, but when the agent reads back the final details and asks you to confirm, "
                f"say you need to check with your partner first and will call back. Do not agree "
                f"to book."
            ),
            opening="Hi, I'm thinking about getting a repair scheduled.",
            availability=av,
            expected=Expected(outcome="no_booking"),
        )

    # 10. pricing question before booking
    for _ in range(10):
        svc = g.repair_service()
        cid, name, phone, aid, line1 = g.existing()
        av = g.pick_avail(SERVICE_TRADE[svc])
        g.add(
            "pricing_then_book",
            tags=["pricing", "existing_customer"],
            brief=(
                f"{g.who(name, phone)} The address is {line1}. The problem: "
                f"{rng.choice(PROBLEMS[svc])}. Before agreeing to anything, ask how much the visit "
                f"will cost and push once for a rough estimate of the total repair price. Then book "
                f"for {av.phrase}."
            ),
            opening="Hi, how much do you charge to come out and fix something?",
            availability=av,
            expected=Expected(outcome="booked", services=[svc], customer_id=cid, address_id=aid),
        )

    # 11. replacement inquiry with financing question (should NOT escalate)
    repl = [
        (
            Service.WATER_HEATER_REPLACEMENT_ESTIMATE,
            "your water heater is 16 years old and leaking from the tank, you want a new one",
        ),
        (
            Service.HVAC_REPLACEMENT_ESTIMATE,
            "your furnace is 22 years old and you were told "
            "it's not worth repairing, you want a new system",
        ),
    ]
    for i in range(8):
        svc, situation = repl[i % 2]
        cid, name, phone, aid, line1 = g.existing()
        av = g.pick_avail(SERVICE_TRADE[svc])
        g.add(
            "replacement_estimate",
            tags=["sales", "financing", "existing_customer"],
            brief=(
                f"{g.who(name, phone)} The address is {line1}. {situation.capitalize()}. Ask whether "
                f"they offer financing and roughly what a new one costs. Then book for {av.phrase}."
            ),
            opening="Hi, I think I need to replace my system. Do you do financing?",
            availability=av,
            expected=Expected(outcome="booked", services=[svc], customer_id=cid, address_id=aid),
        )

    # 12. reschedule the existing appointment P9001 (C003, water heater, Thursday morning)
    p9001_slot = g.free["P9001_slot"][0]
    for _ in range(6):
        av = rng.choice(
            [
                a
                for a in g.feasible(Trade.PLUMBING)
                if not a.contains(p9001_slot.start.date(), p9001_slot.start.hour)
            ]
        )
        g.add(
            "reschedule",
            tags=["reschedule", "existing_customer"],
            brief=(
                f"{g.who('Priya Raman', '555-0103')} You already have a water heater repair booked "
                f"for this Thursday morning at 8 Birch Ln in Edison. You can't make it anymore and "
                f"want to move it to {av.phrase}. You do not want to cancel."
            ),
            opening="Hi, I need to move an appointment I have with you.",
            availability=av,
            expected=Expected(outcome="rescheduled", customer_id="C003", appointment_id="P9001"),
        )

    # 13. two problems in different trades: two bookings
    for _ in range(5):
        hv = rng.choice([Service.FURNACE_REPAIR, Service.BOILER_REPAIR])
        pl = rng.choice([Service.DRAIN_CLOG, Service.TOILET_REPAIR, Service.LEAK_REPAIR])
        cid, name, phone, aid, line1 = g.existing()
        av = g.pick_avail(Trade.HVAC, Trade.PLUMBING)
        g.add(
            "two_problems",
            tags=["multi_service", "existing_customer"],
            brief=(
                f"{g.who(name, phone)} The address is {line1}. You have two problems: "
                f"{PROBLEMS[hv][0]}, and {PROBLEMS[pl][0]}. Mention the second one only after the "
                f"agent starts on the first. You want both fixed, available {av.phrase}."
            ),
            opening=f"Hi, {PROBLEMS[hv][0]}.",
            availability=av,
            expected=Expected(outcome="booked", services=[hv, pl], customer_id=cid, address_id=aid),
        )

    # 14. backend fault: confirm succeeds but the response is lost
    for _ in range(5):
        svc = g.repair_service()
        cid, name, phone, aid, line1 = g.existing()
        av = g.pick_avail(SERVICE_TRADE[svc])
        g.add(
            "tool_fault",
            tags=["reliability", "existing_customer"],
            brief=(
                f"{g.who(name, phone)} The address is {line1}. The problem: "
                f"{rng.choice(PROBLEMS[svc])}. You are available {av.phrase}. You want to book."
            ),
            opening="Hi, I'd like to schedule a repair.",
            availability=av,
            expected=Expected(outcome="booked", services=[svc], customer_id=cid, address_id=aid),
            faults={"confirm_booking": ["timeout_after"]},
        )

    # Stratified split: 30% of each category is held out and never shown to the Agent Manager.
    by_cat: dict[str, list[Scenario]] = {}
    for s in g.out:
        by_cat.setdefault(s.category, []).append(s)
    for items in by_cat.values():
        held = rng.sample(items, max(1, round(len(items) * 0.3)))
        for s in held:
            s.split = "heldout"
    return g.out


async def _free_slots(settings: Settings) -> dict[str, list[Slot]]:
    with tempfile.TemporaryDirectory() as d:
        engine = make_engine(f"sqlite+aiosqlite:///{d}/gen.db")
        sm = make_sessionmaker(engine)
        await seed(engine, sm, settings.today, settings.seed)
        async with sm() as s:
            booked = select(Appointment.slot_id).where(Appointment.status == "booked")
            slots = (
                await s.scalars(
                    select(Slot).where(Slot.id.not_in(booked), Slot.is_emergency.is_(False))
                )
            ).all()
            p = await s.get(Appointment, "P9001")
            p_slot = await s.get(Slot, p.slot_id)
        await engine.dispose()
    out: dict[str, list[Slot]] = {t.value: [x for x in slots if x.trade == t.value] for t in Trade}
    out["P9001_slot"] = [p_slot]
    return out


def generate(settings: Settings, path: Path = SUITE_PATH) -> list[Scenario]:
    free = asyncio.run(_free_slots(settings))
    suite = _gen(settings.today, free, settings.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated by `evalgate gen-scenarios`. Deterministic; do not edit by hand.\n"
        "# Ground truth is built from the seed data, so it is correct by construction.\n"
    )
    body = yaml.safe_dump([s.model_dump(mode="json") for s in suite], sort_keys=False, width=100)
    path.write_text(header + body)
    return suite
