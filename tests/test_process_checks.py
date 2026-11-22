"""Process checks and simulator flags, using cases taken from the first real baseline run."""

from evalgate.graders import Check, process_checks, simulator_flags
from evalgate.scenarios import load_suite


def user(turn, text):
    return {"type": "user", "turn": turn, "text": text}


def agent(turn, text):
    return {"type": "assistant", "turn": turn, "text": text}


def tool(turn, name, args, result):
    return {"type": "tool", "turn": turn, "name": name, "args": args, "result": result}


def by_name(checks):
    return {c.name: c for c in checks}


def test_lookup_with_a_number_nobody_said():
    # Gemini looked up 555-0199 on the first turn, before the caller gave any number.
    events = [
        user(1, "Hello, I'd like to schedule a repair."),
        tool(1, "lookup_customer", {"phone": "555-0199"}, {"ok": True, "found": False}),
    ]
    c = by_name(process_checks(events))["no_unspoken_phone_lookup"]
    assert not c.passed and "555-0199" in c.detail


def test_lookup_with_the_callers_number_in_any_format():
    events = [
        user(1, "Hi"),
        user(2, "it's (555) 0116"),
        tool(2, "lookup_customer", {"phone": "555-0116"}, {"ok": True, "found": True}),
    ]
    assert by_name(process_checks(events))["no_unspoken_phone_lookup"].passed


def test_invented_city_and_zip():
    events = [
        user(1, "Rachel Stein, 555-0109, 12 Willow Way."),
        tool(
            1,
            "create_customer",
            {
                "name": "Rachel Stein",
                "phone": "555-0109",
                "line1": "12 Willow Way",
                "city": "Unknown City",
                "zip": "90210",
            },
            {"ok": False, "error": "exists"},
        ),
    ]
    c = by_name(process_checks(events))["no_invented_customer_details"]
    assert not c.passed and "Unknown City" in c.detail and "90210" in c.detail


def test_stated_city_and_zip_pass():
    events = [
        user(1, "Nina Alvarez, 555-0201, 18 Ridge Rd, Montclair, zip 07043"),
        tool(
            1,
            "create_customer",
            {
                "name": "Nina Alvarez",
                "phone": "555-0201",
                "line1": "18 Ridge Rd",
                "city": "Montclair",
                "zip": "07043",
            },
            {"ok": True},
        ),
    ]
    assert by_name(process_checks(events))["no_invented_customer_details"].passed


def test_claiming_a_booking_before_it_exists():
    # Seen in the baseline: a read-back phrased as if the booking were already made.
    events = [
        user(3, "Tomorrow afternoon please."),
        tool(3, "propose_booking", {}, {"ok": True, "token": "book-1"}),
        agent(3, "I have that scheduled for tomorrow, 1pm to 5pm. Does that work?"),
    ]
    c = by_name(process_checks(events))["no_premature_confirmation"]
    assert not c.passed and "turn 3" in c.detail


def test_confirmation_after_commit_is_fine():
    events = [
        user(3, "Tomorrow afternoon please."),
        agent(3, "I can book tomorrow 1pm to 5pm. Would you like me to book that?"),
        user(4, "Yes."),
        tool(4, "confirm_booking", {"token": "book-1"}, {"ok": True}),
        agent(4, "Your appointment is confirmed. You're all set!"),
    ]
    assert by_name(process_checks(events))["no_premature_confirmation"].passed


def test_failed_commit_does_not_license_a_claim():
    events = [
        user(4, "Yes."),
        tool(4, "confirm_booking", {"token": "book-1"}, {"ok": False, "error": "timeout"}),
        agent(4, "Great, you're booked!"),
    ]
    assert not by_name(process_checks(events))["no_premature_confirmation"].passed


def test_flags_caller_that_agreed_to_a_date_outside_its_own_brief():
    # tool_fault-03: available "a weekday morning next week", but asked for "Tuesday January
    # 13th" (this week) and agreed to it. That failure belongs to the simulator, not the agent.
    s = next(x for x in load_suite() if x.id == "tool_fault-03")
    events = [
        user(5, "I'll go with Tuesday, January 13th, at 9 a.m. please."),
        tool(
            6,
            "confirm_booking",
            {"token": "book-1"},
            {"ok": True, "summary": "heat pump repair ..., Tuesday January 13, 8am to noon"},
        ),
    ]
    checks = [Check("slot_fits_availability", False)]
    assert simulator_flags(s, checks, events) == ["caller_agreed_outside_its_availability"]


def test_flags_caller_junk_and_leaves_clean_runs_alone():
    s = load_suite()[0]
    junk = [user(3, "It's iced over.[No reply needed]Sorry, I think you're supposed to respond.")]
    assert "caller_off_script" in simulator_flags(s, [], junk)
    assert (
        simulator_flags(
            s, [Check("slot_fits_availability", True)], [user(1, "Hi, tomorrow works.")]
        )
        == []
    )
