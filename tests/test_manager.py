"""Agent Manager and trust gate, end to end with scripted models (no network)."""

import json

import pytest

from evalgate.config import Settings
from evalgate.evals import RunResult, run_suite
from evalgate.gate import SplitStats, compare_runs, decide, sign_test
from evalgate.llm import Completion, ScriptedLLM, Step, TextDelta
from evalgate.manager import cluster_failures, parse_proposal, propose
from evalgate.oracle import OracleAgent, OracleCaller
from evalgate.scenarios import load_suite, select_scenarios
from evalgate.sops import AgentVersion, load_library

FIX = "Never tell the caller a visit is booked or scheduled until confirm_booking succeeds."


class HabitAgent(OracleAgent):
    """Perfect, except it says 'I have that scheduled' at read-back, unless its procedures
    tell it not to. Its behavior depends on the SOP text, like a real model's."""

    async def stream(self, messages, tools):
        fixed = FIX in messages[0]["content"]
        async for ev in super().stream(messages, tools):
            if not fixed and isinstance(ev, TextDelta):
                ev = TextDelta(ev.text.replace("I can book", "I have that scheduled for you:"))
            if not fixed and isinstance(ev, Completion):
                ev = Completion(
                    ev.text.replace("I can book", "I have that scheduled for you:"), ev.tool_calls
                )
            yield ev


def manager_reply(text: str) -> ScriptedLLM:
    return ScriptedLLM([Step(text=text)])


def good_fix(version: AgentVersion) -> str:
    old = next(s.text for s in version.sops if s.id == "booking-confirmation")
    return (
        "```json\n"
        + json.dumps(
            {
                "action": "edit",
                "sop_id": "booking-confirmation",
                "new_text": f"{old} {FIX}",
                "rationale": "The agent claimed bookings during the read-back.",
            }
        )
        + "\n```"
    )


@pytest.fixture
def scenarios():
    return select_scenarios(load_suite(), "all", per_category=1)


async def run(version, scenarios, out, st):
    await run_suite(scenarios, version, HabitAgent, OracleCaller, st, concurrency=16, out_dir=out)


async def test_full_loop_fixes_the_habit_and_gets_promoted(scenarios, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    st = Settings(runs_dir=str(tmp_path))
    base = AgentVersion(sops=load_library(), model="habit")
    await run(base, scenarios, tmp_path / "base", st)

    top = cluster_failures(tmp_path / "base")[0]
    assert top.check == "no_premature_confirmation"
    assert all("-" in ex.scenario_id for ex in top.examples)

    p, cand, _ = await propose(tmp_path / "base", base, manager_reply(good_fix(base)))
    assert not p.errors and not p.flags and cand.parent_id == base.id
    assert (tmp_path / "versions" / f"{cand.id}.json").exists()
    assert AgentVersion.load(tmp_path / "versions" / f"{cand.id}.json").id == cand.id

    await run(cand, scenarios, tmp_path / "cand", st)
    cmp, decision, report = compare_runs(
        tmp_path / "base", tmp_path / "cand", tmp_path / "proposals"
    )
    assert cmp["heldout"].delta_pp > 30 and not cmp["heldout"].broken
    assert decision["decision"] == "promote", decision
    assert "Decision: **promote**" in report.read_text()


async def test_manager_only_sees_training_failures(scenarios, tmp_path):
    st = Settings(runs_dir=str(tmp_path))
    base = AgentVersion(sops=load_library(), model="habit")
    await run(base, scenarios, tmp_path / "base", st)
    held = {s.id for s in scenarios if s.split == "heldout"}
    seen = {ex.scenario_id for c in cluster_failures(tmp_path / "base") for ex in c.examples}
    assert seen and not seen & held


async def test_proposal_that_memorizes_a_customer_is_flagged(scenarios, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    st = Settings(runs_dir=str(tmp_path))
    base = AgentVersion(sops=load_library(), model="habit")
    await run(base, scenarios, tmp_path / "base", st)
    name = next(iter(cluster_failures(tmp_path / "base")[0].examples)).specifics
    who = sorted(x for x in name if not x[0].isdigit())[0]
    reply = json.dumps(
        {
            "action": "add",
            "sop_id": "special-case",
            "title": "Special case",
            "new_text": f"When {who} calls, do not say scheduled early. {FIX}",
        }
    )
    p, cand, _ = await propose(tmp_path / "base", base, manager_reply(reply))
    assert cand is not None and any("specifics" in f for f in p.flags)

    await run(cand, scenarios, tmp_path / "cand", st)
    _, decision, _ = compare_runs(tmp_path / "base", tmp_path / "cand", tmp_path / "proposals")
    assert decision["decision"] == "needs_human"
    assert any("specifics" in r for r in decision["reasons"])


def test_parse_proposal_handles_bad_replies():
    assert parse_proposal("sorry, I can't").errors
    assert parse_proposal("{not json}").errors
    p = parse_proposal('Here you go: {"action": "EDIT", "sop_id": "tone", "new_text": "x"}')
    assert p.action == "edit" and p.sop_id == "tone" and not p.errors


async def test_invalid_proposals_never_become_versions(scenarios, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    st = Settings(runs_dir=str(tmp_path))
    base = AgentVersion(sops=load_library(), model="habit")
    await run(base, scenarios, tmp_path / "base", st)
    bad = json.dumps({"action": "edit", "sop_id": "does-not-exist", "new_text": "Be nice."})
    p, cand, _ = await propose(tmp_path / "base", base, manager_reply(bad))
    assert cand is None and p.errors
    assert not (tmp_path / "versions").exists()


def st_(n, base, cand, fixed=0, broken=0):
    return SplitStats(
        n=n, base_rate=base, cand_rate=cand, fixed=["f"] * fixed, broken=["b"] * broken
    )


def cmp_(train, held, cats=()):
    return {
        "train": train,
        "heldout": held,
        "broken_categories": list(cats),
        "generalization_gap_pp": round(train.delta_pp - held.delta_pp, 1),
    }


@pytest.mark.parametrize(
    ("cmp", "flags", "want"),
    [
        (cmp_(st_(30, 40, 60, 6), st_(14, 40, 60, 3)), [], "promote"),
        (cmp_(st_(30, 40, 60, 6), st_(14, 50, 43, 0, 1)), [], "reject"),  # held-out down
        (cmp_(st_(30, 40, 60, 6, 1), st_(14, 40, 60, 3), ["safety_gas"]), [], "reject"),
        (cmp_(st_(30, 40, 80, 12), st_(14, 40, 45, 1)), [], "needs_human"),  # big gap
        (cmp_(st_(30, 40, 60, 6), st_(6, 40, 60, 1)), [], "needs_human"),  # thin evidence
        (cmp_(st_(30, 40, 60, 6), st_(14, 40, 60, 3)), ["memorized"], "needs_human"),
    ],
)
def test_gate_decisions(cmp, flags, want):
    assert decide(cmp, flags)["decision"] == want


def test_sign_test():
    assert sign_test(0, 0) is None
    assert sign_test(6, 0) == 0.0312
    assert sign_test(3, 3) == 1.0


def test_version_file_must_match_its_contents(tmp_path):
    v = AgentVersion(sops=load_library(), model="m")
    path = v.save(tmp_path)
    d = json.loads(path.read_text())
    d["sops"][0]["text"] += " tampered"
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="hash"):
        AgentVersion.load(path)


def test_runresult_roundtrip_has_split():
    assert "split" in RunResult.__dataclass_fields__


async def test_compare_skips_scenarios_that_changed_between_runs(scenarios, tmp_path):
    from evalgate.gate import compare

    st = Settings(runs_dir=str(tmp_path))
    base = AgentVersion(sops=load_library(), model="habit")
    await run(base, scenarios, tmp_path / "a", st)
    await run(base, scenarios, tmp_path / "b", st)
    trace = next((tmp_path / "b" / "traces").glob("*.json"))
    t = json.loads(trace.read_text())
    t["scenario"]["brief"] += " (edited)"
    trace.write_text(json.dumps(t))
    cmp = compare(tmp_path / "a", tmp_path / "b")
    assert cmp["skipped_changed_scenarios"] == 1
    assert cmp["paired_conversations"] == len(scenarios) - 1


def test_net_harm_is_rejected():
    # The real first round: held-out flat, train lost 6 and gained 3.
    cmp = cmp_(st_(25, 40, 28, 3, 6), st_(28, 25, 25, 2, 2))
    d = decide(cmp)
    assert d["decision"] == "reject"
    assert any("net harm" in r for r in d["reasons"])


class RecordingLLM:
    model = "recording"

    def __init__(self, reply):
        self.reply, self.seen = reply, []

    async def stream(self, messages, tools):
        self.seen.append(messages)
        yield Completion(self.reply)


async def test_next_proposal_learns_from_a_rejected_one(scenarios, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    st = Settings(runs_dir=str(tmp_path))
    base = AgentVersion(sops=load_library(), model="habit")
    await run(base, scenarios, tmp_path / "base", st)

    # Round 1: a useless edit that changes nothing the agent does.
    useless = json.dumps(
        {"action": "edit", "sop_id": "tone", "new_text": "Be calm and kind.", "rationale": "tone"}
    )
    _, cand, _ = await propose(tmp_path / "base", base, manager_reply(useless))
    await run(cand, scenarios, tmp_path / "cand1", st)
    _, decision, _ = compare_runs(tmp_path / "base", tmp_path / "cand1", tmp_path / "proposals")
    assert decision["decision"] != "promote"
    saved = json.loads((tmp_path / "proposals" / f"{cand.id}.json").read_text())
    assert saved["gate_decision"] == decision["decision"] and saved["candidate_run"]

    # Round 2: the Manager is shown what was tried and why it didn't ship.
    rec = RecordingLLM(good_fix(base))
    p2, cand2, _ = await propose(tmp_path / "base", base, rec)
    prompt = rec.seen[0][1]["content"]
    assert "EARLIER ATTEMPTS THAT DID NOT WORK" in prompt and "Be calm and kind." in prompt
    assert cand2 is not None and cand2.id != cand.id


def test_outcome_regression_is_rejected_even_on_already_failing_calls():
    # Round 1 again: the candidate hung up on calls that already failed a process check, so they
    # weren't "newly broken", but the right-customer/address checks went from 0 to 8 failures.
    cmp = cmp_(st_(25, 40, 40, 3, 3), st_(28, 25, 25, 2, 2))
    cmp["check_failures"] = {
        "correct_address": {"base": 0, "candidate": 8},
        "no_unspoken_phone_lookup": {"base": 16, "candidate": 6},
    }
    d = decide(cmp)
    assert d["decision"] == "reject"
    assert any("correct_address failed in 0 -> 8" in r for r in d["reasons"])
