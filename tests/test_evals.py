"""Tests for the eval suite itself: scenarios, simulator, graders, runner."""

from collections import Counter
from datetime import date, datetime

import pytest

from evalgate.config import Settings
from evalgate.evals import run_suite
from evalgate.graders import Booking, RunOutcome, grade, same_street
from evalgate.llm import ScriptedLLM, Step
from evalgate.oracle import CarelessAgent, OracleAgent, OracleCaller
from evalgate.scenarios import SUITE_PATH, generate, load_suite
from evalgate.simulator import CallerSimulator
from evalgate.sops import AgentVersion, load_library


@pytest.fixture(scope="module")
def suite():
    return load_suite()


def test_suite_shape(suite):
    assert len(suite) == 150
    assert len({s.id for s in suite}) == 150
    by_cat = Counter(s.category for s in suite)
    held = Counter(s.category for s in suite if s.split == "heldout")
    assert set(held) == set(by_cat), "every category needs held-out scenarios"
    assert 0.25 < sum(held.values()) / 150 < 0.35


def test_committed_suite_matches_generator(tmp_path):
    """suite.yaml must be exactly what the generator produces, so nobody hand-edits ground truth."""
    generate(Settings(), tmp_path / "suite.yaml")
    assert (tmp_path / "suite.yaml").read_text() == SUITE_PATH.read_text()


def test_same_street():
    assert same_street("402 Washington St", "402 Washington Street")
    assert same_street("300 Hudson St Apt 12", "300 hudson st, apt 12")
    assert not same_street("402 Washington St", "40 Washington St")


def _outcome(**kw) -> RunOutcome:
    base = dict(
        bookings=[],
        agent_text=[],
        escalated=None,
        ended={"summary": "x"},
        caller_hung_up=True,
        hit_turn_limit=False,
    )
    return RunOutcome(**{**base, **kw})


def test_grader_catches_invented_prices(suite):
    s = next(x for x in suite if x.category == "pricing_then_book")
    checks = {
        c.name: c
        for c in grade(
            s,
            _outcome(agent_text=["It's $89 to come out, and repairs are usually $450."]),
            date(2026, 1, 12),
        )
    }
    assert not checks["prices_from_policy"].passed
    assert "450" in checks["prices_from_policy"].detail


def test_grader_rejects_slot_outside_availability(suite):
    s = next(x for x in suite if x.category == "standard_repair")
    e = s.expected
    bad = Booking(
        "AP-1",
        e.customer_id,
        e.address_id,
        "x",
        e.services[0].value,
        datetime(2026, 1, 24, 8),
        "booked",
    )
    checks = {c.name: c.passed for c in grade(s, _outcome(bookings=[bad]), date(2026, 1, 12))}
    assert checks["correct_services"] and checks["correct_customer"]
    assert not checks["slot_fits_availability"]


async def test_simulator_detects_hangup(suite):
    sim = CallerSimulator(
        suite[0], ScriptedLLM([Step(text="Yes. "), Step(text="Thanks, bye! [HANGUP]")]), "Co"
    )
    assert await sim.reply("Does Tuesday work?") == "Yes."
    assert not sim.hung_up
    assert await sim.reply("You're booked.") == "Thanks, bye!"
    assert sim.hung_up
    assert sim.messages[1] == {"role": "assistant", "content": suite[0].opening}


async def test_oracle_passes_every_scenario(suite, tmp_path):
    """If a perfect agent can't pass, the scenario or grader is broken."""
    st = Settings(runs_dir=str(tmp_path))
    version = AgentVersion(sops=load_library(), model="oracle")
    results, out = await run_suite(
        suite,
        version,
        OracleAgent,
        lambda _: OracleCaller(),
        st,
        concurrency=16,
        out_dir=tmp_path / "oracle",
    )
    failed = [(r.scenario_id, r.failed_checks, r.error) for r in results if not r.success]
    assert not failed
    assert (out / "report.md").exists() and (out / "results.jsonl").exists()


async def test_careless_agent_fails(suite, tmp_path):
    st = Settings(runs_dir=str(tmp_path))
    version = AgentVersion(sops=load_library(), model="careless")
    picked = [s for s in suite if s.category in ("safety_gas", "out_of_scope", "multi_address")]
    results, _ = await run_suite(
        picked,
        version,
        CarelessAgent,
        lambda _: OracleCaller(),
        st,
        concurrency=16,
        out_dir=tmp_path / "careless",
    )
    assert not any(r.success for r in results)
    gas = [r for r in results if r.category == "safety_gas"]
    assert all("no_booking_made" in r.failed_checks for r in gas)
    assert all("escalated_correctly" in r.failed_checks for r in gas)


def test_per_category_sampling(suite):
    from evalgate.scenarios import select_scenarios

    picked = select_scenarios(suite, "train", per_category=2)
    counts = Counter(s.category for s in picked)
    assert len(counts) == 14 and set(counts.values()) == {2}
    assert all(s.split == "train" for s in picked)


def test_tool_schemas_are_provider_friendly():
    import json

    from evalgate.tools import TOOLS

    blob = json.dumps([t.openai_schema() for t in TOOLS.values()])
    for banned in ('"title"', '"$ref"', '"$defs"', '"format"', '"default"'):
        assert banned not in blob


async def test_rate_limit_stops_run_and_resume_finishes_it(suite, tmp_path):
    import httpx
    from openai import RateLimitError

    from evalgate.evals import load_results

    picked = [s for s in suite if s.category == "standard_repair"][:6]
    quota = {"left": 3}

    class QuotaAgent(OracleAgent):
        async def stream(self, messages, tools):
            if quota["left"] < 0:
                req = httpx.Request("POST", "http://x")
                raise RateLimitError(
                    "daily quota", response=httpx.Response(429, request=req), body=None
                )
            async for ev in super().stream(messages, tools):
                yield ev

    def factory(sc):
        quota["left"] -= 1
        return QuotaAgent(sc)

    st = Settings(runs_dir=str(tmp_path))
    version = AgentVersion(sops=load_library(), model="oracle")
    out = tmp_path / "run"
    results, _ = await run_suite(
        picked, version, factory, lambda _: OracleCaller(), st, concurrency=1, out_dir=out
    )
    summary = __import__("json").loads((out / "summary.json").read_text())
    assert summary["stopped_on_rate_limit"] and summary["remaining"] == 3
    assert len(load_results(out)) == 3 and all(r.success for r in results)
    assert "Partial run" in (out / "report.md").read_text()

    quota["left"] = 100  # next day
    results, _ = await run_suite(
        picked, version, factory, lambda _: OracleCaller(), st, concurrency=1, out_dir=out
    )
    summary = __import__("json").loads((out / "summary.json").read_text())
    assert summary["remaining"] == 0 and not summary["stopped_on_rate_limit"]
    assert len(results) == 6 and all(r.success for r in results)
    assert len(load_results(out)) == 6  # nothing ran twice


async def test_resume_refuses_a_different_agent_version(suite, tmp_path):
    st = Settings(runs_dir=str(tmp_path))
    v1 = AgentVersion(sops=load_library(), model="oracle")
    v2 = v1.with_sop_text("tone", "Be brief.")
    await run_suite(
        suite[:1], v1, OracleAgent, lambda _: OracleCaller(), st, out_dir=tmp_path / "r"
    )
    with pytest.raises(ValueError, match="use a new folder"):
        await run_suite(
            suite[:1], v2, OracleAgent, lambda _: OracleCaller(), st, out_dir=tmp_path / "r"
        )
