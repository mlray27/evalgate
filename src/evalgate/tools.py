"""Typed tools, the action guard, and the executor that ties them together.

Design: the LLM only *proposes* actions. Irreversible actions are a two-step protocol:

    propose_*(details)  -> token + human-readable summary   (validated, nothing written)
    <agent reads summary back; caller replies>
    confirm_*(token)    -> executes exactly the proposed details, once

Because confirm_* takes only a token, the details that execute are exactly the details that were
read back. The guard decides whether a confirm may run; the database enforces one booking per slot;
and an idempotency record written in the same transaction makes retries safe.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import and_, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from evalgate.config import Settings
from evalgate.db import Address, Appointment, Customer, IdempotencyRecord, Slot
from evalgate.domain import EMERGENCY_FEE, PRICING, SERVICE_TRADE, Service

# ---------------------------------------------------------------------------------------------
# Session state shared between runtime, guard and tools
# ---------------------------------------------------------------------------------------------


class ToolKind(StrEnum):
    READ = "read"  # no side effects; may be retried, may run speculatively
    WRITE = "write"  # reversible, low-risk writes (create customer, add address)
    PROPOSE = "propose"  # validates and stages an irreversible action; writes nothing
    COMMIT = "commit"  # irreversible; guarded + idempotent
    CONTROL = "control"  # escalate, end call, decline proposal


@dataclass
class Proposal:
    token: str
    kind: Literal["booking", "cancellation", "reschedule"]
    details: dict[str, Any]
    summary: str
    created_turn: int
    status: Literal["open", "declined", "superseded", "committed", "expired"] = "open"


@dataclass
class SessionState:
    session_id: str
    turn_index: int = 0  # incremented on every caller message
    proposals: dict[str, Proposal] = field(default_factory=dict)
    escalated: dict | None = None
    ended: dict | None = None
    _token_n: int = 0

    def new_token(self, kind: str) -> str:
        self._token_n += 1
        return f"{kind[:4]}-{self._token_n}"


@dataclass
class ToolContext:
    state: SessionState
    sm: async_sessionmaker
    settings: Settings


# ---------------------------------------------------------------------------------------------
# Argument schemas (also rendered to JSON Schema for the LLM)
# ---------------------------------------------------------------------------------------------


class LookupCustomerArgs(BaseModel):
    phone: str = Field(description="Phone number the caller gives, any format")


class CreateCustomerArgs(BaseModel):
    name: str
    phone: str
    line1: str = Field(description="Street address including apartment")
    city: str
    zip: str


class AddAddressArgs(BaseModel):
    customer_id: str
    line1: str
    city: str
    zip: str


class CheckAvailabilityArgs(BaseModel):
    service: Service
    earliest_date: date | None = Field(None, description="YYYY-MM-DD; defaults to today")
    latest_date: date | None = Field(None, description="YYYY-MM-DD; defaults to earliest + 6 days")
    time_of_day: Literal["morning", "afternoon", "any"] = "any"
    emergency: bool = Field(False, description="Only today's emergency evening slot")


class ListAppointmentsArgs(BaseModel):
    customer_id: str


class GetPricingArgs(BaseModel):
    service: Service
    emergency: bool = False


class ProposeBookingArgs(BaseModel):
    customer_id: str
    address_id: str
    slot_id: str
    service: Service
    notes: str = Field("", max_length=400, description="Short description of the problem")


class ProposeCancellationArgs(BaseModel):
    appointment_id: str


class ProposeRescheduleArgs(BaseModel):
    appointment_id: str
    new_slot_id: str


class TokenArgs(BaseModel):
    token: str


class EscalateArgs(BaseModel):
    reason: str
    urgency: Literal["normal", "emergency"] = "normal"


class EndCallArgs(BaseModel):
    summary: str


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


def _err(code: str, message: str, **extra: Any) -> dict:
    return {"ok": False, "error": code, "message": message, **extra}


def _digits(phone: str) -> str:
    d = re.sub(r"\D", "", phone)
    return d[-7:] if len(d) >= 7 else d


def _fmt_hour(h: int) -> str:
    if h == 12:
        return "noon"
    return f"{h % 12 or 12}{'am' if h < 12 else 'pm'}"


def describe_slot(slot: Slot) -> str:
    day = slot.start.strftime("%A %B ") + str(slot.start.day)
    return f"{day}, {_fmt_hour(slot.start.hour)} to {_fmt_hour(slot.end.hour)}"


def _slot_dict(slot: Slot) -> dict:
    return {
        "slot_id": slot.id,
        "when": describe_slot(slot),
        "date": slot.start.date().isoformat(),
        "technician": slot.technician.name,
        "emergency": slot.is_emergency,
    }


async def _slot_is_free(s: AsyncSession, slot_id: str) -> bool:
    n = await s.scalar(
        select(func.count())
        .select_from(Appointment)
        .where(Appointment.slot_id == slot_id, Appointment.status == "booked")
    )
    return n == 0


def _addr_str(a: Address) -> str:
    return f"{a.line1}, {a.city} {a.zip}"


# ---------------------------------------------------------------------------------------------
# READ tools
# ---------------------------------------------------------------------------------------------


async def lookup_customer(ctx: ToolContext, a: LookupCustomerArgs) -> dict:
    async with ctx.sm() as s:
        customers = (await s.scalars(select(Customer))).all()
        match = next((c for c in customers if _digits(c.phone) == _digits(a.phone)), None)
        if not match:
            return {"ok": True, "found": False}
        return {
            "ok": True,
            "found": True,
            "customer_id": match.id,
            "name": match.name,
            "is_member": match.is_member,
            "addresses": [{"address_id": x.id, "address": _addr_str(x)} for x in match.addresses],
        }


async def check_availability(ctx: ToolContext, a: CheckAvailabilityArgs) -> dict:
    today = ctx.settings.today
    trade = SERVICE_TRADE[a.service].value
    if a.emergency:
        if a.service in (Service.HVAC_MAINTENANCE,):
            return _err("not_allowed", "Maintenance visits cannot use emergency slots.")
        start, end = today, today
    else:
        start = a.earliest_date or today
        end = a.latest_date or (start + timedelta(days=6))
        if start < today:
            return _err("invalid_date", f"earliest_date is in the past; today is {today}.")
        if end < start:
            return _err("invalid_date", "latest_date is before earliest_date.")
    async with ctx.sm() as s:
        booked = select(Appointment.slot_id).where(Appointment.status == "booked")
        q = (
            select(Slot)
            .where(
                and_(
                    Slot.trade == trade,
                    Slot.is_emergency.is_(a.emergency),
                    Slot.start >= datetime.combine(start, datetime.min.time()),
                    Slot.start < datetime.combine(end + timedelta(days=1), datetime.min.time()),
                    Slot.id.not_in(booked),
                )
            )
            .order_by(Slot.start, Slot.technician_id)
        )
        slots = (await s.scalars(q)).all()
    if a.time_of_day == "morning":
        slots = [x for x in slots if x.start.hour < 12]
    elif a.time_of_day == "afternoon":
        slots = [x for x in slots if x.start.hour >= 12]
    return {"ok": True, "slots": [_slot_dict(x) for x in slots[:6]], "total_matching": len(slots)}


async def list_appointments(ctx: ToolContext, a: ListAppointmentsArgs) -> dict:
    async with ctx.sm() as s:
        rows = (
            await s.execute(
                select(Appointment, Slot)
                .join(Slot, Slot.id == Appointment.slot_id)
                .where(Appointment.customer_id == a.customer_id, Appointment.status == "booked")
                .order_by(Slot.start)
            )
        ).all()
        out = []
        for appt, slot in rows:
            addr = await s.get(Address, appt.address_id)
            out.append(
                {
                    "appointment_id": appt.id,
                    "service": appt.service,
                    "when": describe_slot(slot),
                    "address": _addr_str(addr),
                }
            )
    return {"ok": True, "appointments": out}


async def get_pricing(ctx: ToolContext, a: GetPricingArgs) -> dict:
    p = dict(PRICING[a.service])
    if a.emergency:
        p["emergency_fee"] = EMERGENCY_FEE
    return {"ok": True, "service": a.service.value, **p}


# ---------------------------------------------------------------------------------------------
# WRITE tools (reversible)
# ---------------------------------------------------------------------------------------------


async def create_customer(ctx: ToolContext, a: CreateCustomerArgs) -> dict:
    async with ctx.sm() as s, s.begin():
        for c in (await s.scalars(select(Customer))).all():
            if _digits(c.phone) == _digits(a.phone):
                return _err("exists", "A customer with this phone exists.", customer_id=c.id)
        n = await s.scalar(select(func.count()).select_from(Customer))
        n_addr = await s.scalar(select(func.count()).select_from(Address))
        cid, aid = f"C{n + 1:03d}", f"A{n_addr + 1:03d}"
        s.add(Customer(id=cid, name=a.name, phone=a.phone))
        s.add(Address(id=aid, customer_id=cid, line1=a.line1, city=a.city, zip=a.zip))
    return {"ok": True, "customer_id": cid, "address_id": aid}


async def add_address(ctx: ToolContext, a: AddAddressArgs) -> dict:
    async with ctx.sm() as s, s.begin():
        if not await s.get(Customer, a.customer_id):
            return _err("not_found", "Unknown customer_id.")
        n_addr = await s.scalar(select(func.count()).select_from(Address))
        aid = f"A{n_addr + 1:03d}"
        s.add(Address(id=aid, customer_id=a.customer_id, line1=a.line1, city=a.city, zip=a.zip))
    return {"ok": True, "address_id": aid}


# ---------------------------------------------------------------------------------------------
# PROPOSE tools: validate and stage, write nothing
# ---------------------------------------------------------------------------------------------


def _supersede(state: SessionState, kind: str, details: dict) -> None:
    """A new proposal replaces open ones for the same thing, not every open proposal.

    Same booking = same service or same slot (the caller corrected something). A second booking
    for a different service, like a furnace and a clogged drain in one call, stays open.
    Same cancellation/reschedule = same appointment.
    """
    for p in state.proposals.values():
        if p.kind != kind or p.status != "open":
            continue
        d = p.details
        if kind == "booking":
            same = d["service"] == details["service"] or d["slot_id"] == details["slot_id"]
        else:
            same = d["appointment_id"] == details["appointment_id"]
        if same:
            p.status = "superseded"


async def _validate_booking(
    s: AsyncSession, customer_id: str, address_id: str, slot_id: str, service: Service, today: date
) -> dict | tuple[Customer, Address, Slot]:
    cust = await s.get(Customer, customer_id)
    if not cust:
        return _err("not_found", "Unknown customer_id.")
    addr = await s.get(Address, address_id)
    if not addr or addr.customer_id != customer_id:
        return _err("invalid_address", "address_id does not belong to this customer.")
    slot = await s.get(Slot, slot_id)
    if not slot:
        return _err("not_found", "Unknown slot_id. Use check_availability.")
    if slot.trade != SERVICE_TRADE[service].value:
        return _err(
            "trade_mismatch",
            f"Slot is for {slot.trade} but {service.value} needs {SERVICE_TRADE[service].value}.",
        )
    if slot.start.date() < today:
        return _err("invalid_slot", "Slot is in the past.")
    if slot.is_emergency and service == Service.HVAC_MAINTENANCE:
        return _err("not_allowed", "Maintenance cannot use emergency slots.")
    if not await _slot_is_free(s, slot_id):
        return _err("slot_unavailable", "That slot was just taken. Check availability again.")
    return cust, addr, slot


async def propose_booking(ctx: ToolContext, a: ProposeBookingArgs) -> dict:
    async with ctx.sm() as s:
        v = await _validate_booking(
            s, a.customer_id, a.address_id, a.slot_id, a.service, ctx.settings.today
        )
        if isinstance(v, dict):
            return v
        cust, addr, slot = v
        summary = (
            f"{a.service.value.replace('_', ' ')} for {cust.name} at {_addr_str(addr)}, "
            f"{describe_slot(slot)}"
        )
    _supersede(ctx.state, "booking", a.model_dump(mode="json"))
    token = ctx.state.new_token("booking")
    ctx.state.proposals[token] = Proposal(
        token=token,
        kind="booking",
        details=a.model_dump(mode="json"),
        summary=summary,
        created_turn=ctx.state.turn_index,
    )
    return {
        "ok": True,
        "token": token,
        "summary": summary,
        "next": "Read the summary back and wait for the caller to agree before confirm_booking.",
    }


async def propose_cancellation(ctx: ToolContext, a: ProposeCancellationArgs) -> dict:
    async with ctx.sm() as s:
        appt = await s.get(Appointment, a.appointment_id)
        if not appt or appt.status != "booked":
            return _err("not_found", "No active appointment with that id. Use list_appointments.")
        slot = await s.get(Slot, appt.slot_id)
        summary = f"cancel the {appt.service.replace('_', ' ')} visit on {describe_slot(slot)}"
    _supersede(ctx.state, "cancellation", a.model_dump())
    token = ctx.state.new_token("cancellation")
    ctx.state.proposals[token] = Proposal(
        token, "cancellation", a.model_dump(), summary, ctx.state.turn_index
    )
    return {"ok": True, "token": token, "summary": summary}


async def propose_reschedule(ctx: ToolContext, a: ProposeRescheduleArgs) -> dict:
    async with ctx.sm() as s:
        appt = await s.get(Appointment, a.appointment_id)
        if not appt or appt.status != "booked":
            return _err("not_found", "No active appointment with that id. Use list_appointments.")
        v = await _validate_booking(
            s,
            appt.customer_id,
            appt.address_id,
            a.new_slot_id,
            Service(appt.service),
            ctx.settings.today,
        )
        if isinstance(v, dict):
            return v
        old = await s.get(Slot, appt.slot_id)
        summary = (
            f"move the {appt.service.replace('_', ' ')} visit from {describe_slot(old)} "
            f"to {describe_slot(v[2])}"
        )
    _supersede(ctx.state, "reschedule", a.model_dump())
    token = ctx.state.new_token("reschedule")
    ctx.state.proposals[token] = Proposal(
        token, "reschedule", a.model_dump(), summary, ctx.state.turn_index
    )
    return {"ok": True, "token": token, "summary": summary}


# ---------------------------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------------------------


@dataclass
class GuardDecision:
    allow: bool
    reason: str
    proposal: Proposal | None = None


def guard_commit(ctx: ToolContext, token: str, expected_kind: str) -> GuardDecision:
    """Deterministic checks before any irreversible action. DB-state checks run in-transaction."""
    st = ctx.state
    p = st.proposals.get(token)
    if p is None:
        return GuardDecision(False, "unknown_token")
    if p.kind != expected_kind:
        return GuardDecision(False, f"token_is_for_{p.kind}")
    if p.status == "declined":
        return GuardDecision(False, "caller_declined")
    if p.status == "superseded":
        return GuardDecision(False, "superseded_by_newer_proposal")
    if p.status == "expired" or st.turn_index - p.created_turn > ctx.settings.proposal_ttl_turns:
        p.status = "expired"
        return GuardDecision(False, "proposal_expired")
    if st.turn_index <= p.created_turn:
        # The caller has not spoken since the proposal was made, so they cannot have agreed to it.
        return GuardDecision(False, "no_caller_turn_since_proposal")
    if p.status != "open":
        return GuardDecision(False, f"proposal_{p.status}")
    return GuardDecision(True, "ok", p)


# ---------------------------------------------------------------------------------------------
# COMMIT tools: guarded, transactional, idempotent
# ---------------------------------------------------------------------------------------------


def idempotency_key(state: SessionState, token: str) -> str:
    return f"{state.session_id}:{token}"


async def _commit(
    ctx: ToolContext,
    token: str,
    kind: str,
    tool: str,
    apply: Callable[[AsyncSession, Proposal], Awaitable[dict]],
) -> dict:
    key = idempotency_key(ctx.state, token)
    async with ctx.sm() as s:
        prior = await s.get(IdempotencyRecord, key)
        if prior is not None:
            # A retry after a lost response: return the original result, do nothing.
            return {**prior.result, "replayed": True}

    decision = guard_commit(ctx, token, kind)
    if not decision.allow:
        return _err("guard_denied", decision.reason)
    p = decision.proposal
    assert p is not None
    try:
        async with ctx.sm() as s, s.begin():
            result = await apply(s, p)
            if not result.get("ok"):
                raise _Abort(result)
            # Same transaction as the side effect: both commit or neither does.
            s.add(IdempotencyRecord(key=key, tool=tool, result=result))
    except _Abort as e:
        return e.result
    except IntegrityError:
        return _err("slot_unavailable", "That slot was just taken. Check availability again.")
    p.status = "committed"
    return result


class _Abort(Exception):
    def __init__(self, result: dict) -> None:
        self.result = result


async def confirm_booking(ctx: ToolContext, a: TokenArgs) -> dict:
    async def apply(s: AsyncSession, p: Proposal) -> dict:
        d = p.details
        v = await _validate_booking(
            s,
            d["customer_id"],
            d["address_id"],
            d["slot_id"],
            Service(d["service"]),
            ctx.settings.today,
        )
        if isinstance(v, dict):
            return v
        appt_id = "AP-" + uuid.uuid4().hex[:8]
        s.add(
            Appointment(
                id=appt_id,
                customer_id=d["customer_id"],
                address_id=d["address_id"],
                slot_id=d["slot_id"],
                service=d["service"],
                notes=d.get("notes", ""),
                session_id=ctx.state.session_id,
            )
        )
        await s.flush()  # surface unique-index violations inside the transaction
        return {"ok": True, "appointment_id": appt_id, "summary": p.summary}

    return await _commit(ctx, a.token, "booking", "confirm_booking", apply)


async def confirm_cancellation(ctx: ToolContext, a: TokenArgs) -> dict:
    async def apply(s: AsyncSession, p: Proposal) -> dict:
        appt = await s.get(Appointment, p.details["appointment_id"])
        if not appt or appt.status != "booked":
            return _err("not_found", "Appointment is no longer active.")
        appt.status = "cancelled"
        return {"ok": True, "appointment_id": appt.id, "summary": p.summary}

    return await _commit(ctx, a.token, "cancellation", "confirm_cancellation", apply)


async def confirm_reschedule(ctx: ToolContext, a: TokenArgs) -> dict:
    async def apply(s: AsyncSession, p: Proposal) -> dict:
        appt = await s.get(Appointment, p.details["appointment_id"])
        if not appt or appt.status != "booked":
            return _err("not_found", "Appointment is no longer active.")
        new_slot = p.details["new_slot_id"]
        if not await _slot_is_free(s, new_slot):
            return _err("slot_unavailable", "That slot was just taken. Check availability again.")
        appt.slot_id = new_slot
        await s.flush()
        return {"ok": True, "appointment_id": appt.id, "summary": p.summary}

    return await _commit(ctx, a.token, "reschedule", "confirm_reschedule", apply)


# ---------------------------------------------------------------------------------------------
# CONTROL tools
# ---------------------------------------------------------------------------------------------


async def decline_proposal(ctx: ToolContext, a: TokenArgs) -> dict:
    p = ctx.state.proposals.get(a.token)
    if not p:
        return _err("unknown_token", "No such proposal.")
    if p.status == "committed":
        return _err("already_committed", "Already confirmed; propose a cancellation instead.")
    p.status = "declined"
    return {"ok": True}


async def escalate_to_human(ctx: ToolContext, a: EscalateArgs) -> dict:
    ctx.state.escalated = a.model_dump()
    return {"ok": True, "message": "A team member will call the customer back."}


async def end_call(ctx: ToolContext, a: EndCallArgs) -> dict:
    ctx.state.ended = a.model_dump()
    return {"ok": True}


# ---------------------------------------------------------------------------------------------
# Registry + executor
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    name: str
    kind: ToolKind
    args: type[BaseModel]
    handler: Callable[[ToolContext, Any], Awaitable[dict]]
    description: str

    def openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": _compact(self.args.model_json_schema()),
            },
        }


def _compact(schema: dict) -> dict:
    """Drop titles, defaults and formats, inline refs: fewer tokens, wider provider support."""
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(dict(defs[node["$ref"].split("/")[-1]]))
            if "anyOf" in node and any(x.get("type") == "null" for x in node["anyOf"]):
                rest = [x for x in node["anyOf"] if x.get("type") != "null"]
                if len(rest) == 1:
                    merged = {**{k: v for k, v in node.items() if k != "anyOf"}, **rest[0]}
                    return walk(merged)
            # Some providers (Gemini) accept only a subset of JSON Schema; keep to the basics.
            return {k: walk(v) for k, v in node.items() if k not in ("title", "default", "format")}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(schema)


TOOLS: dict[str, ToolSpec] = {
    t.name: t
    for t in [
        ToolSpec(
            "lookup_customer",
            ToolKind.READ,
            LookupCustomerArgs,
            lookup_customer,
            "Find a customer account by phone number.",
        ),
        ToolSpec(
            "check_availability",
            ToolKind.READ,
            CheckAvailabilityArgs,
            check_availability,
            "List open appointment slots for a service.",
        ),
        ToolSpec(
            "list_appointments",
            ToolKind.READ,
            ListAppointmentsArgs,
            list_appointments,
            "List a customer's upcoming appointments.",
        ),
        ToolSpec(
            "get_pricing",
            ToolKind.READ,
            GetPricingArgs,
            get_pricing,
            "Get the company's price policy for a service.",
        ),
        ToolSpec(
            "create_customer",
            ToolKind.WRITE,
            CreateCustomerArgs,
            create_customer,
            "Create a new customer account with a service address.",
        ),
        ToolSpec(
            "add_address",
            ToolKind.WRITE,
            AddAddressArgs,
            add_address,
            "Add a service address to an existing customer.",
        ),
        ToolSpec(
            "propose_booking",
            ToolKind.PROPOSE,
            ProposeBookingArgs,
            propose_booking,
            "Stage a booking. Returns a token and a summary to read back to the caller.",
        ),
        ToolSpec(
            "propose_cancellation",
            ToolKind.PROPOSE,
            ProposeCancellationArgs,
            propose_cancellation,
            "Stage a cancellation. Returns a token and summary.",
        ),
        ToolSpec(
            "propose_reschedule",
            ToolKind.PROPOSE,
            ProposeRescheduleArgs,
            propose_reschedule,
            "Stage a reschedule. Returns a token and summary.",
        ),
        ToolSpec(
            "confirm_booking",
            ToolKind.COMMIT,
            TokenArgs,
            confirm_booking,
            "Execute a proposed booking after the caller agreed. Safe to retry.",
        ),
        ToolSpec(
            "confirm_cancellation",
            ToolKind.COMMIT,
            TokenArgs,
            confirm_cancellation,
            "Execute a proposed cancellation after the caller agreed. Safe to retry.",
        ),
        ToolSpec(
            "confirm_reschedule",
            ToolKind.COMMIT,
            TokenArgs,
            confirm_reschedule,
            "Execute a proposed reschedule after the caller agreed. Safe to retry.",
        ),
        ToolSpec(
            "decline_proposal",
            ToolKind.CONTROL,
            TokenArgs,
            decline_proposal,
            "Mark a proposal as declined because the caller did not agree.",
        ),
        ToolSpec(
            "escalate_to_human",
            ToolKind.CONTROL,
            EscalateArgs,
            escalate_to_human,
            "Hand the call to a human for a callback.",
        ),
        ToolSpec(
            "end_call",
            ToolKind.CONTROL,
            EndCallArgs,
            end_call,
            "End the call with a one-line summary.",
        ),
    ]
}


@dataclass
class ToolFaults:
    """Fault injection for tests and eval scenarios.

    plan maps tool name -> list of faults consumed one per call:
      "timeout_before": raise before executing (nothing happens)
      "timeout_after":  execute, then lose the response (the dangerous case)
      None:             behave normally
    """

    plan: dict[str, list[str | None]] = field(default_factory=dict)

    def next(self, tool: str) -> str | None:
        q = self.plan.get(tool)
        return q.pop(0) if q else None


@dataclass
class ToolOutcome:
    name: str
    args: dict
    result: dict
    kind: ToolKind
    duration_ms: float
    attempts: int = 1


class ToolExecutor:
    def __init__(self, ctx: ToolContext, faults: ToolFaults | None = None) -> None:
        self.ctx = ctx
        self.faults = faults or ToolFaults()

    def schemas(self) -> list[dict]:
        return [t.openai_schema() for t in TOOLS.values()]

    async def run(self, name: str, raw_args: str | dict) -> ToolOutcome:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        spec = TOOLS.get(name)
        if spec is None:
            return ToolOutcome(
                name, {}, _err("unknown_tool", f"No tool named {name}."), ToolKind.CONTROL, 0.0
            )
        try:
            args_dict = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            args = spec.args.model_validate(args_dict or {})
        except (json.JSONDecodeError, ValidationError) as e:
            msg = e.errors(include_url=False) if isinstance(e, ValidationError) else str(e)
            return ToolOutcome(
                name, {"raw": raw_args}, _err("invalid_arguments", str(msg)), spec.kind, 0.0
            )

        attempts = 0
        retries = self.ctx.settings.read_tool_retries if spec.kind == ToolKind.READ else 0
        while True:
            attempts += 1
            try:
                result = await self._call_once(spec, args)
                break
            except TimeoutError:
                if attempts <= retries:
                    continue
                result = _err("timeout", f"{name} timed out.")
                break
        dur = (loop.time() - t0) * 1000
        return ToolOutcome(name, args.model_dump(mode="json"), result, spec.kind, dur, attempts)

    async def _call_once(self, spec: ToolSpec, args: BaseModel) -> dict:
        fault = self.faults.next(spec.name)
        if fault == "timeout_before":
            raise TimeoutError
        coro = spec.handler(self.ctx, args)
        if spec.kind == ToolKind.READ:
            result = await asyncio.wait_for(coro, self.ctx.settings.read_tool_timeout_s)
        else:
            result = await coro
        if fault == "timeout_after":
            raise TimeoutError  # side effect happened; caller never sees the result
        return result
