"""Oracle agent and caller for validating the eval suite itself.

The oracle agent cheats: it reads the scenario's ground truth and makes exactly the right tool
calls. Run against every scenario it should score 100%. If it doesn't, a scenario is infeasible or
a grader is wrong, and the suite needs fixing before any real agent is measured with it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator

from evalgate.llm import Completion, LLMEvent, Step, TextDelta, ToolCall, last_tool_result
from evalgate.scenarios import Scenario
from evalgate.simulator import HANGUP

DONE_PHRASE = "You're all set"
ASK_DETAILS = "Can I get your full name, phone number, and address?"


class _PlanLLM:
    """Runs a generator of Steps; the generator reads self.msgs to see tool results."""

    model = "oracle"

    def __init__(self) -> None:
        self.msgs: list[dict] = []
        self.n = 0
        self._gen: Iterator[Step] = self.plan()

    def plan(self) -> Iterator[Step]:
        raise NotImplementedError

    async def stream(self, messages: list[dict], tools: list[dict]) -> AsyncIterator[LLMEvent]:
        self.msgs = messages
        step = next(self._gen, Step(text="Okay."))
        self.n += 1
        if step.text:
            yield TextDelta(step.text)
        calls = [
            ToolCall(f"o{self.n}_{i}", name, json.dumps(args))
            for i, (name, args) in enumerate(step.calls)
        ]
        yield Completion(step.text, calls)

    def result(self, tool: str) -> dict:
        return last_tool_result(self.msgs, tool)


class OracleAgent(_PlanLLM):
    def __init__(self, scenario: Scenario) -> None:
        self.s = scenario
        super().__init__()

    def _confirm(self, tool: str, token: str) -> Iterator[Step]:
        yield Step(calls=[(tool, {"token": token})])
        if self.result(tool).get("error") == "timeout":
            yield Step(calls=[(tool, {"token": token})])  # safe retry, idempotent

    def plan(self) -> Iterator[Step]:
        e, av = self.s.expected, self.s.availability

        if e.outcome == "escalated":
            yield Step(
                calls=[
                    ("escalate_to_human", {"reason": "possible gas or CO", "urgency": "emergency"})
                ]
            )
            yield Step(
                text="Please leave the house right now and call the gas company from "
                "outside. Someone from our team will call you back."
            )
            return

        if e.outcome == "no_booking":
            yield Step(calls=[("end_call", {"summary": "no booking"})])
            yield Step(text="No problem, call us back anytime.")
            return

        if e.outcome == "rescheduled":
            yield Step(calls=[("list_appointments", {"customer_id": "C003"})])
            yield Step(
                calls=[
                    (
                        "check_availability",
                        {
                            "service": "water_heater_repair",
                            "earliest_date": str(av.earliest),
                            "latest_date": str(av.latest),
                            "time_of_day": av.time_of_day,
                        },
                    )
                ]
            )
            slot = self.result("check_availability")["slots"][0]["slot_id"]
            yield Step(
                calls=[
                    (
                        "propose_reschedule",
                        {"appointment_id": e.appointment_id, "new_slot_id": slot},
                    )
                ]
            )
            yield Step(text=f"I can {self.result('propose_reschedule')['summary']}. Okay?")
            token = self.result("propose_reschedule")["token"]
            yield from self._confirm("confirm_reschedule", token)
            yield Step(calls=[("end_call", {"summary": "rescheduled"})])
            yield Step(text=f"{DONE_PHRASE}.")
            return

        # booked / emergency_booked
        if e.customer_id:
            cid, aid = e.customer_id, e.address_id
        else:
            yield Step(text=ASK_DETAILS)
            yield Step(
                calls=[
                    (
                        "create_customer",
                        {
                            "name": e.new_customer_name,
                            "phone": e.new_customer_phone,
                            "line1": e.new_address_line1,
                            "city": e.new_address_city,
                            "zip": e.new_address_zip,
                        },
                    )
                ]
            )
            r = self.result("create_customer")
            cid, aid = r["customer_id"], r["address_id"]
        emergency = e.outcome == "emergency_booked"
        for svc in e.services:
            args = {"service": svc.value, "emergency": emergency}
            if not emergency:
                args |= {
                    "earliest_date": str(av.earliest),
                    "latest_date": str(av.latest),
                    "time_of_day": av.time_of_day,
                }
            yield Step(calls=[("check_availability", args)])
            slot = self.result("check_availability")["slots"][0]["slot_id"]
            yield Step(
                calls=[
                    (
                        "propose_booking",
                        {
                            "customer_id": cid,
                            "address_id": aid,
                            "slot_id": slot,
                            "service": svc.value,
                        },
                    )
                ]
            )
            summary = self.result("propose_booking")["summary"]
            yield Step(text=f"I can book {summary}. Does that work?")
            yield from self._confirm("confirm_booking", self.result("propose_booking")["token"])
        yield Step(calls=[("end_call", {"summary": "booked"})])
        yield Step(text=f"{DONE_PHRASE}.")


class OracleCaller:
    """Agrees to every read-back, gives its details when asked, hangs up when the agent wraps up."""

    model = "oracle-caller"

    def __init__(self, scenario: Scenario | None = None) -> None:
        self.s = scenario

    async def stream(self, messages: list[dict], tools: list[dict]) -> AsyncIterator[LLMEvent]:
        agent = messages[-1]["content"]
        e = self.s.expected if self.s else None
        if DONE_PHRASE in agent:
            text = f"Great, thanks. Bye. {HANGUP}"
        elif ASK_DETAILS in agent and e and e.new_customer_phone:
            text = (
                f"I'm {e.new_customer_name}, {e.new_customer_phone}, {e.new_address_line1}, "
                f"{e.new_address_city} {e.new_address_zip}."
            )
        else:
            text = "Yes, that works."
        yield TextDelta(text)
        yield Completion(text)


class CarelessAgent(_PlanLLM):
    """Always books the first open slot for furnace repair at the first address. Never escalates.

    Used to check that the graders actually catch wrong behavior.
    """

    def __init__(self, scenario: Scenario) -> None:
        self.s = scenario
        super().__init__()

    def plan(self) -> Iterator[Step]:
        yield Step(calls=[("check_availability", {"service": "furnace_repair"})])
        slot = self.result("check_availability")["slots"][0]["slot_id"]
        yield Step(
            calls=[
                (
                    "propose_booking",
                    {
                        "customer_id": "C001",
                        "address_id": "A001",
                        "slot_id": slot,
                        "service": "furnace_repair",
                    },
                )
            ]
        )
        yield Step(text="Booking a furnace repair, okay? It's usually around $300 total.")
        yield Step(calls=[("confirm_booking", {"token": self.result("propose_booking")["token"]})])
        yield Step(calls=[("end_call", {"summary": "booked"})])
        yield Step(text=f"{DONE_PHRASE}.")
