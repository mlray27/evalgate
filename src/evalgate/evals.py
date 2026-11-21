"""Eval runner: simulated conversations against an agent version, graded and summarized.

Each conversation gets its own copy of the seeded database, so runs are independent and can go
in parallel. Results are written as JSONL (one line per conversation) plus a markdown report.
"""

from __future__ import annotations

import asyncio
import copy
import json
import shutil
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

from evalgate.agent import AgentSession
from evalgate.config import Settings
from evalgate.db import Address, Appointment, Customer, Slot, make_engine, make_sessionmaker
from evalgate.graders import Booking, RunOutcome, grade, passed
from evalgate.llm import LLM
from evalgate.scenarios import Scenario
from evalgate.seed import seed
from evalgate.simulator import CallerSimulator
from evalgate.sops import AgentVersion
from evalgate.tools import ToolFaults, _digits

LLMFactory = Callable[[Scenario], LLM]
RATE_LIMITED = "rate_limited"


@dataclass
class RunResult:
    scenario_id: str
    category: str
    split: str
    run: int
    version_id: str
    success: bool
    checks: list[dict]
    caller_turns: int = 0
    tool_calls: int = 0
    guard_denials: int = 0
    ttft_ms: list[float] = field(default_factory=list)
    agent_prompt_tokens: int = 0
    agent_completion_tokens: int = 0
    sim_prompt_tokens: int = 0
    sim_completion_tokens: int = 0
    duration_s: float = 0.0
    error: str | None = None
    trace_file: str = ""

    @property
    def failed_checks(self) -> list[str]:
        return [c["name"] for c in self.checks if not c["passed"]]


async def _booking(s, appt: Appointment) -> Booking:
    slot = await s.get(Slot, appt.slot_id)
    addr = await s.get(Address, appt.address_id)
    return Booking(
        appt.id,
        appt.customer_id,
        appt.address_id,
        addr.line1,
        appt.service,
        slot.start,
        appt.status,
    )


async def _snapshot(sm, scenario: Scenario, session: AgentSession, p9001_start) -> dict:
    async with sm() as s:
        appts = (
            await s.scalars(select(Appointment).where(Appointment.session_id == session.session_id))
        ).all()
        bookings = [await _booking(s, a) for a in appts]
        p = await s.get(Appointment, "P9001")
        p9001 = await _booking(s, p) if p else None
        new_ids = []
        if scenario.expected.new_customer_phone:
            want = _digits(scenario.expected.new_customer_phone)
            new_ids = [
                c.id for c in (await s.scalars(select(Customer))).all() if _digits(c.phone) == want
            ]
    return {
        "bookings": bookings,
        "p9001": p9001,
        "p9001_original_start": p9001_start,
        "new_customer_ids": new_ids,
    }


async def run_conversation(
    scenario: Scenario,
    version: AgentVersion,
    agent_llm: LLM,
    sim_llm: LLM,
    settings: Settings,
    template_db: Path,
    workdir: Path,
    run_idx: int = 0,
) -> RunResult:
    sid = f"{scenario.id}-r{run_idx}"
    db_path = workdir / f"{sid}.db"
    shutil.copy(template_db, db_path)
    engine = make_engine(f"sqlite+aiosqlite:///{db_path}")
    sm = make_sessionmaker(engine)
    t0 = time.monotonic()
    session = AgentSession(
        version,
        agent_llm,
        sm,
        settings,
        session_id=sid,
        faults=ToolFaults(copy.deepcopy(scenario.faults)),
    )
    sim = CallerSimulator(scenario, sim_llm, settings.company_name)
    transcript: list[dict] = []
    agent_text: list[str] = []
    hit_turn_limit = False
    error = None
    async with sm() as s:
        p = await s.get(Appointment, "P9001")
        p9001_start = (await s.get(Slot, p.slot_id)).start

    async def converse() -> None:
        nonlocal hit_turn_limit
        caller = sim.opening
        for _ in range(settings.max_caller_turns):
            transcript.append({"role": "caller", "text": caller})
            reply = await session.say(caller)
            agent_text.append(reply)
            transcript.append({"role": "agent", "text": reply})
            if session.done:
                return
            caller = await sim.reply(reply)
            if sim.hung_up:
                transcript.append({"role": "caller", "text": caller, "hangup": True})
                return
        hit_turn_limit = True

    try:
        await asyncio.wait_for(converse(), settings.conversation_timeout_s)
    except TimeoutError:
        hit_turn_limit = True
        error = "conversation_timeout"
    except Exception as e:  # a crashed run is a failed run, never a silent skip
        error = f"{type(e).__name__}: {getattr(e, 'body', None) or e}"
        if type(e).__name__ == "RateLimitError":
            error = f"{RATE_LIMITED}: {e}"

    snap = await _snapshot(sm, scenario, session, p9001_start)
    outcome = RunOutcome(
        bookings=snap["bookings"],
        agent_text=agent_text,
        escalated=session.state.escalated,
        ended=session.state.ended,
        caller_hung_up=sim.hung_up,
        hit_turn_limit=hit_turn_limit,
        p9001=snap["p9001"],
        p9001_original_start=snap["p9001_original_start"],
        new_customer_ids=snap["new_customer_ids"],
    )
    checks = grade(scenario, outcome, settings.today)
    await engine.dispose()
    db_path.unlink(missing_ok=True)

    tools = session.trace.of_type("tool")
    result = RunResult(
        scenario_id=scenario.id,
        category=scenario.category,
        split=scenario.split,
        run=run_idx,
        version_id=version.id,
        success=passed(checks) and error is None,
        checks=[asdict(c) for c in checks],
        caller_turns=sum(1 for t in transcript if t["role"] == "caller"),
        tool_calls=len(tools),
        guard_denials=sum(1 for t in tools if t["result"].get("error") == "guard_denied"),
        ttft_ms=[m.ttft_ms for m in session.metrics if m.ttft_ms is not None],
        agent_prompt_tokens=sum(m.prompt_tokens for m in session.metrics),
        agent_completion_tokens=sum(m.completion_tokens for m in session.metrics),
        sim_prompt_tokens=sim.prompt_tokens,
        sim_completion_tokens=sim.completion_tokens,
        duration_s=round(time.monotonic() - t0, 2),
        error=error,
    )
    trace_file = workdir / "traces" / f"{sid}.json"
    trace_file.parent.mkdir(exist_ok=True)
    trace_file.write_text(
        json.dumps(
            {
                "scenario": scenario.model_dump(mode="json"),
                "result": asdict(result),
                "transcript": transcript,
                "trace": session.trace.to_dict(),
            },
            indent=1,
            default=str,
        )
    )
    result.trace_file = str(trace_file)
    return result


def load_results(out: Path) -> list[RunResult]:
    path = out / "results.jsonl"
    if not path.exists():
        return []
    return [RunResult(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


async def run_suite(
    scenarios: list[Scenario],
    version: AgentVersion,
    agent_llm: LLMFactory,
    sim_llm: LLMFactory,
    settings: Settings,
    runs: int = 1,
    concurrency: int | None = None,
    out_dir: Path | None = None,
    progress: Callable[[RunResult], None] | None = None,
) -> tuple[list[RunResult], Path]:
    """Run scenarios x runs. If out_dir already has results, finished conversations are skipped,
    so a run cut short by a daily quota can be resumed later into the same report.

    A conversation that hits a provider rate limit is not recorded (it would score the provider,
    not the agent) and stops the remaining work.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out_dir or Path(settings.runs_dir) / f"{stamp}-{version.id}"
    out.mkdir(parents=True, exist_ok=True)

    version_file = out / "version.json"
    if version_file.exists():
        prev = json.loads(version_file.read_text())["id"]
        if prev != version.id:
            raise ValueError(f"{out} holds results for {prev}, not {version.id}; use a new folder")
    version_file.write_text(
        json.dumps(
            {
                "id": version.id,
                "parent_id": version.parent_id,
                "model": version.model,
                "note": version.note,
                "sops": [s.__dict__ for s in version.sops],
            },
            indent=1,
        )
    )

    done = {(r.scenario_id, r.run) for r in load_results(out)}
    todo = [(sc, k) for sc in scenarios for k in range(runs) if (sc.id, k) not in done]

    template = out / "template.db"
    engine = make_engine(f"sqlite+aiosqlite:///{template}")
    await seed(engine, make_sessionmaker(engine), settings.today, settings.seed)
    await engine.dispose()

    sem = asyncio.Semaphore(concurrency or settings.eval_concurrency)
    stop = asyncio.Event()
    new: list[RunResult] = []
    results_file = (out / "results.jsonl").open("a")

    async def one(sc: Scenario, k: int) -> None:
        async with sem:
            if stop.is_set():
                return
            r = await run_conversation(
                sc, version, agent_llm(sc), sim_llm(sc), settings, template, out, k
            )
        if r.error and r.error.startswith(RATE_LIMITED):
            stop.set()
            Path(r.trace_file).unlink(missing_ok=True)  # noqa: ASYNC240
            return
        new.append(r)
        results_file.write(json.dumps(asdict(r)) + "\n")
        results_file.flush()
        if progress:
            progress(r)

    try:
        await asyncio.gather(*(one(sc, k) for sc, k in todo))
    finally:
        results_file.close()
        template.unlink(missing_ok=True)

    wanted = {(sc.id, k) for sc in scenarios for k in range(runs)}
    results = sorted(
        (r for r in load_results(out) if (r.scenario_id, r.run) in wanted),
        key=lambda r: (r.scenario_id, r.run),
    )
    summary = summarize(results)
    summary["remaining"] = len(wanted) - len(results)
    summary["stopped_on_rate_limit"] = stop.is_set()
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "report.md").write_text(render_report(summary, version))
    return results, out


# ---------------------------------------------------------------------------------------------
# Summary + report
# ---------------------------------------------------------------------------------------------


def _pct(xs: list[bool]) -> float:
    return round(100 * sum(xs) / len(xs), 1) if xs else 0.0


def _p(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1)


def summarize(results: list[RunResult]) -> dict:
    by_split: dict[str, list[bool]] = defaultdict(list)
    by_cat: dict[str, list[bool]] = defaultdict(list)
    by_scenario: dict[str, list[bool]] = defaultdict(list)
    check_fail: dict[str, int] = defaultdict(int)
    check_total: dict[str, int] = defaultdict(int)
    for r in results:
        by_split[r.split].append(r.success)
        by_cat[r.category].append(r.success)
        by_scenario[r.scenario_id].append(r.success)
        for c in r.checks:
            check_total[c["name"]] += 1
            check_fail[c["name"]] += not c["passed"]
    ttft = [t for r in results for t in r.ttft_ms]
    runs = max((r.run for r in results), default=0) + 1
    return {
        "conversations": len(results),
        "scenarios": len(by_scenario),
        "runs_per_scenario": runs,
        "success_rate": _pct([r.success for r in results]),
        "pass_all_k": _pct([all(v) for v in by_scenario.values()]),
        "by_split": {k: _pct(v) for k, v in sorted(by_split.items())},
        "by_category": {k: {"success": _pct(v), "n": len(v)} for k, v in sorted(by_cat.items())},
        "check_failure_rate": {
            k: round(100 * check_fail[k] / check_total[k], 1) for k in sorted(check_total)
        },
        "guard_denials": sum(r.guard_denials for r in results),
        "errors": sum(r.error is not None for r in results),
        "ttft_ms_p50": _p(ttft, 0.5),
        "ttft_ms_p95": _p(ttft, 0.95),
        "mean_caller_turns": round(statistics.mean(r.caller_turns for r in results), 1)
        if results
        else 0,
        "tokens": {
            "agent_prompt": sum(r.agent_prompt_tokens for r in results),
            "agent_completion": sum(r.agent_completion_tokens for r in results),
            "sim_prompt": sum(r.sim_prompt_tokens for r in results),
            "sim_completion": sum(r.sim_completion_tokens for r in results),
        },
        "failures": [
            {
                "scenario": r.scenario_id,
                "run": r.run,
                "failed": [
                    f"{c['name']}: {c['detail']}".rstrip(": ") for c in r.checks if not c["passed"]
                ],
                "error": r.error,
            }
            for r in results
            if not r.success
        ],
    }


def render_report(s: dict, version: AgentVersion) -> str:
    lines = [
        f"# Eval report: {version.id}",
        "",
        f"Model `{version.model}`. {s['scenarios']} scenarios x {s['runs_per_scenario']} run(s) = "
        f"{s['conversations']} conversations.",
        "",
        *(
            [
                f"> **Partial run:** {s['remaining']} conversations still to go "
                f"(stopped on a provider rate limit; rerun with --resume)."
            ]
            if s.get("remaining")
            else []
        ),
        "",
        f"- **Success rate:** {s['success_rate']}%",
        f"- **pass^k (all runs of a scenario pass):** {s['pass_all_k']}%",
        "- **By split:** " + ", ".join(f"{k} {v}%" for k, v in s["by_split"].items()),
        f"- **TTFT:** p50 {s['ttft_ms_p50']} ms, p95 {s['ttft_ms_p95']} ms",
        f"- **Mean caller turns:** {s['mean_caller_turns']}",
        f"- **Guard denials:** {s['guard_denials']}, **errors:** {s['errors']}",
        "",
        "## By category",
        "",
        "| Category | Success | n |",
        "|---|---|---|",
        *[f"| {k} | {v['success']}% | {v['n']} |" for k, v in s["by_category"].items()],
        "",
        "## Check failure rates",
        "",
        "| Check | Fail % |",
        "|---|---|",
        *[f"| {k} | {v}% |" for k, v in s["check_failure_rate"].items()],
        "",
        "## Failures",
        "",
    ]
    for f in s["failures"]:
        why = "; ".join(f["failed"]) or ""
        if f["error"]:
            why = f"{why}; error {f['error']}".strip("; ")
        lines.append(f"- `{f['scenario']}` run {f['run']}: {why}")
    if not s["failures"]:
        lines.append("None.")
    return "\n".join(lines) + "\n"
