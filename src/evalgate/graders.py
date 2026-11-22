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
    address_zip: str = ""
    customer_name: str = ""


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
    events: list[dict] = field(default_factory=list)  # the session trace, for process checks


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

    checks.extend(process_checks(r.events))

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

        if e.new_customer_phone:
            # The account must hold what the caller said, not details the model filled in.
            last = (e.new_customer_name or "").split()[-1].lower()
            ok = bool(active) and all(
                b.address_zip.strip() == (e.new_address_zip or "")
                and last in b.customer_name.lower()
                for b in active
            )
            detail = "; ".join(f"{b.customer_name} / zip {b.address_zip}" for b in active)
            checks.append(Check("customer_details_as_stated", ok, detail))

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


# ---------------------------------------------------------------------------------------------
# Process checks: how the agent got there, not just where it ended up. These catch failures the
# database happened to absorb (e.g. a lookup on a number nobody said that just returned "not
# found"). They are computed from the trace alone, so old runs can be re-scored offline.
# ---------------------------------------------------------------------------------------------

PROCESS_CHECKS = (
    "no_unspoken_phone_lookup",
    "no_invented_customer_details",
    "no_premature_confirmation",
)

# "Your appointment is confirmed", "I have a repair scheduled for you", "you're all set", ...
CLAIM_RE = re.compile(
    r"\b(?:"
    r"(?:is|are|has been|have been|now)\s+(?:all\s+)?(?:confirmed|scheduled|booked)"
    r"|(?:i've|i have|we've|we have)\s+(?:\w+\s+){0,8}?(?:scheduled|booked|confirmed)"
    r"|you(?:'re| are)\s+(?:all set|booked|confirmed|scheduled)"
    r"|(?:appointment|booking|visit)\s+(?:is\s+)?(?:confirmed|booked)"
    r")\b",
    re.I,
)
COMMIT_TOOLS = ("confirm_booking", "confirm_reschedule", "confirm_cancellation")


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _words(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()) + " "


def process_checks(events: list[dict]) -> list[Check]:
    said_by_turn: dict[int, str] = {}
    for e in events:
        if e["type"] == "user":
            said_by_turn[e["turn"]] = e["text"]

    def said_up_to(turn: int) -> str:
        return " ".join(t for k, t in sorted(said_by_turn.items()) if k <= turn)

    unspoken, invented, premature = [], [], []
    committed_turns = {
        e["turn"]
        for e in events
        if e["type"] == "tool" and e["name"] in COMMIT_TOOLS and e["result"].get("ok")
    }
    for e in events:
        if e["type"] == "tool":
            heard = said_up_to(e["turn"])
            a = e.get("args") or {}
            phone = _digits(str(a.get("phone", "")))[-7:]
            if phone and phone not in _digits(heard):
                unspoken.append(f"{e['name']}({a.get('phone')})")
            if e["name"] in ("create_customer", "add_address"):
                for key in ("city", "zip"):
                    val = str(a.get(key, "")).strip()
                    norm = _digits(val) if key == "zip" else _words(val).strip()
                    hay = _digits(heard) if key == "zip" else _words(heard)
                    if val and (not norm or norm not in hay):
                        invented.append(f"{key}={val!r}")
        elif e["type"] == "assistant":
            m = CLAIM_RE.search(e["text"])
            if m and not any(t <= e["turn"] for t in committed_turns):
                premature.append(f"turn {e['turn']}: {m.group(0)!r}")
    return [
        Check("no_unspoken_phone_lookup", not unspoken, "; ".join(unspoken)),
        Check("no_invented_customer_details", not invented, "; ".join(invented)),
        Check("no_premature_confirmation", not premature, "; ".join(premature)),
    ]


# ---------------------------------------------------------------------------------------------
# Simulator sanity. A failed run is only evidence against the agent if the caller played its
# part. These flags mark runs where the simulated caller visibly went off-script; reports show
# success with and without them so simulator noise can't masquerade as agent failure.
# ---------------------------------------------------------------------------------------------

SIM_JUNK_RE = re.compile(
    r"no reply needed|supposed to respond|thank you for choosing|i've scheduled a technician"
    r"|as an ai|\[(?:agent|system|caller)\]",
    re.I,
)
SLOT_RE = re.compile(r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday) (\w+) (\d+)")


def simulator_flags(s: Scenario, checks: list[Check], events: list[dict]) -> list[str]:
    flags = []
    caller = [e["text"] for e in events if e["type"] == "user"]
    if any(SIM_JUNK_RE.search(t) for t in caller):
        flags.append("caller_off_script")
    failed = {c.name for c in checks if not c.passed}
    if "slot_fits_availability" in failed:
        # Did the caller itself name the out-of-window date it ended up with?
        for e in events:
            if e["type"] == "tool" and e["name"] in COMMIT_TOOLS and e["result"].get("ok"):
                m = SLOT_RE.search(e["result"].get("summary", ""))
                if not m:
                    continue
                _, month, day = m.groups()
                pat = re.compile(rf"\b({month}\s+{day}|{day}(st|nd|rd|th))\b", re.I)
                if any(pat.search(t) for t in caller):
                    flags.append("caller_agreed_outside_its_availability")
                    break
    return flags


def passed(checks: list[Check]) -> bool:
    return all(c.passed for c in checks)
