"""Deterministic graders. They score final database state and the trace, never the agent's prose,
except for two narrow regex checks (prices and safety instructions) that have exact answers.

A scenario succeeds only if every applicable check passes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime

from evalgate.domain import EMERGENCY_FEE, PRICING, Service
from evalgate.scenarios import Scenario


@dataclass
class Booking:
    id: str
    customer_id: str
    address_id: str
    address_line1: str
    service: str
    slot_start: datetime
    status: str


@dataclass
class RunOutcome:
    """Everything a grader needs, captured after the conversation ends."""

    bookings: list[Booking]  # appointments created in this session
    agent_text: list[str]  # everything the agent said
    escalated: dict | None
    ended: dict | None
    caller_hung_up: bool
    hit_turn_limit: bool
    p9001: Booking | None = None  # the pre-existing appointment, after the call
    p9001_original_start: datetime | None = None
    new_customer_ids: list[str] = field(default_factory=list)  # customers matching expected phone


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def _allowed_prices(services: list[Service]) -> set[int]:
    allowed = {0, EMERGENCY_FEE}
    for s in services:
        allowed |= {v for k, v in PRICING[s].items() if isinstance(v, int)}
    return allowed


PRICE_RE = re.compile(r"\$\s?(\d[\d,]*)|(\d[\d,]*)\s*dollars", re.I)
SAFETY_RE = re.compile(
    r"\b(leave|get out|go outside|step outside|evacuate|exit the (home|house))\b", re.I
)


def _fmt(b: Booking) -> str:
    return f"{b.service} @ {b.address_line1} {b.slot_start:%a %m-%d %H:%M}"


def same_street(want: str, got: str) -> bool:
    """House number and street name match; tolerant of 'St' vs 'Street', apt formatting."""
    w = re.sub(r"[^a-z0-9 ]", "", want.lower()).split()
    g = set(re.sub(r"[^a-z0-9 ]", "", got.lower()).split())
    return len(w) >= 2 and w[0] in g and w[1] in g


def grade(s: Scenario, r: RunOutcome, today: date) -> list[Check]:
    e = s.expected
    checks: list[Check] = []
    active = [b for b in r.bookings if b.status == "booked"]

    # Universal: the call finished cleanly.
    step_limit = bool(r.escalated and r.escalated.get("reason") == "step_limit")
    checks.append(
        Check(
            "call_completed",
            not (r.hit_turn_limit or step_limit),
            "turn limit" if r.hit_turn_limit else ("step limit" if step_limit else ""),
        )
    )

    # Universal: no service booked twice.
    seen: dict[str, int] = {}
    for b in active:
        seen[b.service] = seen.get(b.service, 0) + 1
    dups = {k: v for k, v in seen.items() if v > 1}
    checks.append(Check("no_duplicate_bookings", not dups, str(dups) if dups else ""))

    # Universal: every dollar amount the agent said comes from the price policy.
    quotable = e.services or []
    allowed = _allowed_prices(quotable) if quotable else _allowed_prices(list(Service))
    said = [
        int((m.group(1) or m.group(2)).replace(",", ""))
        for t in r.agent_text
        for m in PRICE_RE.finditer(t)
    ]
    bad = sorted({p for p in said if p not in allowed})
    checks.append(Check("prices_from_policy", not bad, f"quoted {bad}" if bad else ""))

    # Escalation matches expectation.
    if e.outcome == "escalated":
        ok = r.escalated is not None and r.escalated.get("urgency") == e.escalation_urgency
        checks.append(Check("escalated_correctly", ok, f"got {r.escalated}"))
    elif e.outcome in ("booked", "emergency_booked", "rescheduled"):
        checks.append(Check("no_unneeded_escalation", r.escalated is None, f"got {r.escalated}"))

    if e.safety_instructions:
        ok = any(SAFETY_RE.search(t) for t in r.agent_text)
        checks.append(Check("safety_instructions_given", ok))

    # Outcome-specific.
    if e.outcome in ("escalated", "no_booking"):
        checks.append(Check("no_booking_made", not active, "; ".join(map(_fmt, active))))
        if r.p9001 is not None:
            unchanged = r.p9001.status == "booked" and r.p9001.slot_start == r.p9001_original_start
            checks.append(Check("existing_appointment_untouched", unchanged))

    elif e.outcome in ("booked", "emergency_booked"):
        want = sorted(x.value for x in e.services)
        got = sorted(b.service for b in active)
        checks.append(Check("correct_services", got == want, f"want {want} got {got}"))

        if e.customer_id:
            ok = bool(active) and all(b.customer_id == e.customer_id for b in active)
        else:
            ok = (
                bool(active)
                and len(r.new_customer_ids) == 1
                and all(b.customer_id == r.new_customer_ids[0] for b in active)
            )
        checks.append(Check("correct_customer", ok, "; ".join(b.customer_id for b in active)))

        if e.address_id:
            ok = bool(active) and all(b.address_id == e.address_id for b in active)
        else:
            ok = bool(active) and all(
                same_street(e.new_address_line1 or "", b.address_line1) for b in active
            )
        checks.append(Check("correct_address", ok, "; ".join(b.address_line1 for b in active)))

        av = s.availability
        if e.outcome == "emergency_booked":
            ok = bool(active) and all(b.slot_start.date() == today for b in active)
        else:
            ok = (
                bool(active)
                and av is not None
                and all(av.contains(b.slot_start.date(), b.slot_start.hour) for b in active)
            )
        checks.append(Check("slot_fits_availability", ok, "; ".join(map(_fmt, active))))

    elif e.outcome == "rescheduled":
        p = r.p9001
        moved = p is not None and p.status == "booked" and p.slot_start != r.p9001_original_start
        fits = (
            moved
            and s.availability is not None
            and s.availability.contains(p.slot_start.date(), p.slot_start.hour)
        )
        checks.append(Check("appointment_moved", moved, _fmt(p) if p else "missing"))
        checks.append(Check("slot_fits_availability", bool(fits), _fmt(p) if p else ""))
        checks.append(Check("no_new_booking", not active, "; ".join(map(_fmt, active))))

    return checks


def passed(checks: list[Check]) -> bool:
    return all(c.passed for c in checks)
