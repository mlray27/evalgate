"""Agent Manager: reads failures from an eval run and proposes one procedure change.

What the Manager can see is deliberately limited to what a production system would have:
transcripts, tool calls, and which automated check failed. It never sees scenario briefs (the
ground truth), held-out results, or the graders' code, and it can only output procedure text.
It cannot change scenarios, graders or the trust gate, and it cannot promote its own changes:
every proposal becomes a new immutable agent version that has to pass `evalgate compare`.
"""

from __future__ import annotations

import difflib
import json
import re
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path

from evalgate.evals import RunResult, load_results
from evalgate.graders import PROCESS_CHECKS
from evalgate.llm import LLM, Completion, TextDelta
from evalgate.sops import SOP, AgentVersion

# Plain-English meaning of each check, so the Manager understands a failure without seeing code.
CHECK_DESCRIPTIONS = {
    "call_completed": "The call ended cleanly (no turn limit, no agent stuck in a tool loop).",
    "no_duplicate_bookings": "The same service was not booked twice.",
    "prices_from_policy": "Every dollar amount the agent said came from get_pricing.",
    "escalated_correctly": "The agent escalated to a human with the right urgency when required.",
    "no_unneeded_escalation": "The agent did not escalate a call it should have handled itself.",
    "safety_instructions_given": "On a gas/CO call the agent told the caller to leave the home.",
    "no_booking_made": "No appointment was booked on a call that should not produce one.",
    "existing_appointment_untouched": "An existing appointment was not changed by mistake.",
    "correct_services": "The booked service type matched the caller's actual problem.",
    "correct_customer": "The booking was made for the right customer account.",
    "correct_address": "The booking was made for the right service address.",
    "customer_details_as_stated": "A new account's name and zip match what the caller said.",
    "slot_fits_availability": "The booked time is one the caller said they can make.",
    "appointment_moved": "The caller's appointment was actually rescheduled.",
    "no_new_booking": "Rescheduling did not create a second appointment.",
    "no_unspoken_phone_lookup": "Every phone number used in a tool call was said by the caller.",
    "no_invented_customer_details": "Every city/zip saved for a customer was said by the caller.",
    "no_premature_confirmation": (
        "The agent never told the caller a visit was booked/scheduled/confirmed before a "
        "confirm_* tool call had succeeded."
    ),
}

MAX_TEXT = 1500
EVALUATOR_WORDS = re.compile(
    r"\b(test|tests|grader|graders|scenario|scenarios|eval|evals|benchmark)\b", re.I
)


@dataclass
class FailureExample:
    scenario_id: str
    run: int
    detail: str
    transcript: list[dict]
    tool_calls: list[str]
    specifics: set[str] = field(default_factory=set)  # names, phones, addresses seen in the call


@dataclass
class Cluster:
    check: str
    examples: list[FailureExample]

    @property
    def count(self) -> int:
        return len(self.examples)


def _specifics(events: list[dict]) -> set[str]:
    """Customer-identifying strings from a call, used to spot proposals that memorize examples."""
    out: set[str] = set()
    for e in events:
        if e["type"] != "tool":
            continue
        blobs = [e.get("args") or {}, e.get("result") or {}]
        for b in blobs:
            for k in ("name", "phone", "line1"):
                v = b.get(k)
                if isinstance(v, str) and len(v) >= 5:
                    out.add(v.lower())
            m = re.search(r" for (.+?) at (.+?),", str(b.get("summary", "")))
            if m:  # propose/confirm summaries: "<service> for <name> at <street>, <city> ..."
                out.update(x.lower() for x in m.groups() if len(x) >= 5)
            for a in b.get("addresses", []) if isinstance(b.get("addresses"), list) else []:
                line = str(a.get("address", "")).split(",")[0].lower()
                if len(line) >= 5:
                    out.add(line)
    return out


def _compact_tool(e: dict) -> str:
    args = json.dumps(e.get("args"), default=str)[:160]
    res = e.get("result") or {}
    brief = {k: res[k] for k in ("ok", "error", "message", "found", "token", "summary") if k in res}
    return f"turn {e['turn']}: {e['name']}({args}) -> {json.dumps(brief, default=str)[:200]}"


def cluster_failures(run_dir: Path, split: str = "train") -> list[Cluster]:
    """Group failed checks across a run. Only the given split; never held-out by default.

    Runs where the simulated caller went off-script are skipped: they are evidence about the
    simulator, not the agent.
    """
    traces = {p.stem: json.loads(p.read_text()) for p in (run_dir / "traces").glob("*.json")}
    by_check: dict[str, list[FailureExample]] = {}
    for r in load_results(run_dir):
        if r.split != split or r.success or r.sim_flags or r.error:
            continue
        t = traces.get(f"{r.scenario_id}-r{r.run}")
        if t is None:
            continue
        events = t["trace"]["events"]
        transcript = [x for x in t["transcript"] if x["role"] in ("caller", "agent")]
        tools = [_compact_tool(e) for e in events if e["type"] == "tool"]
        for c in r.checks:
            if c["passed"]:
                continue
            by_check.setdefault(c["name"], []).append(
                FailureExample(
                    r.scenario_id, r.run, c["detail"], transcript, tools, _specifics(events)
                )
            )
    clusters = [Cluster(k, v) for k, v in by_check.items()]
    # Most frequent first; process checks break ties since they point at agent habits.
    clusters.sort(key=lambda c: (-c.count, c.check not in PROCESS_CHECKS, c.check))
    return clusters


def past_attempts(proposals_dir: Path, base_version: str) -> list[Proposal]:
    """Earlier proposals against the same base version that the gate did not promote."""
    out = []
    for p in sorted(proposals_dir.glob("*.json")) if proposals_dir.exists() else []:
        try:
            prop = Proposal(**json.loads(p.read_text()))
        except TypeError:
            continue
        if prop.base_version == base_version and prop.gate_decision in ("reject", "needs_human"):
            out.append(prop)
    return out


def _broken_example(prop: Proposal) -> str:
    """Opening of one conversation the candidate broke, so the Manager sees what went wrong."""
    if not prop.candidate_run or not prop.broken:
        return ""
    path = Path(prop.candidate_run) / "traces" / f"{prop.broken[0]}.json"
    if not path.exists():
        return ""
    t = json.loads(path.read_text())
    lines = []
    for e in t["trace"]["events"][:10]:
        if e["type"] == "user":
            lines.append(f"  caller: {e['text']}")
        elif e["type"] == "assistant":
            lines.append(f"  agent: {e['text']}")
        elif e["type"] == "tool":
            lines.append(f"  tool: {e['name']}({json.dumps(e.get('args'), default=str)[:120]})")
    return "\n".join(lines)


def manager_messages(
    version: AgentVersion,
    cluster: Cluster,
    max_examples: int = 4,
    history: list[Proposal] | None = None,
) -> list[dict]:
    procedures = "\n\n".join(
        f"[{s.id}] (priority {s.priority}, {s.category}) {s.title}\n{s.text}"
        for s in sorted(version.sops, key=lambda s: s.priority)
    )
    examples = []
    for i, ex in enumerate(cluster.examples[:max_examples], 1):
        lines = "\n".join(f"  {x['role']}: {x['text']}" for x in ex.transcript)
        tools = "\n".join(f"  {t}" for t in ex.tool_calls)
        examples.append(
            f"Example {i} (problem: {ex.detail or 'see transcript'})\n"
            f"Conversation:\n{lines}\nTool calls:\n{tools}"
        )
    system = (
        "You maintain the written operating procedures for an AI phone dispatcher at a home "
        "heating, cooling and plumbing company. The dispatcher follows these procedures "
        "literally. You will see real conversations where the dispatcher failed one automated "
        "quality check. Propose exactly ONE change that fixes the underlying behavior for all "
        "future callers: either edit one existing procedure or add one new procedure.\n\n"
        "Rules:\n"
        "- Fix the general behavior. Do not mention any specific customer, name, phone number, "
        "address, or date from the examples.\n"
        "- Keep the procedure short, concrete and phrased as instructions to the dispatcher.\n"
        "- Don't break what already works: keep the rest of an edited procedure's meaning.\n"
        "- Do not refer to tests, checks, graders or evaluation.\n\n"
        "Reply with ONLY a JSON object, no other text:\n"
        '{"action": "edit" or "add", "sop_id": "<existing id to edit, or a new kebab-case id>", '
        '"title": "<title, for add>", "category": "<category, for add>", '
        '"priority": <1-6, for add>, "new_text": "<the full new procedure text>", '
        '"rationale": "<one sentence>"}'
    )
    past = ""
    for i, h in enumerate(history or [], 1):
        worse = (
            ", ".join(f"{k} ({v[0]} -> {v[1]} failing calls)" for k, v in h.regressions.items())
            or "none recorded"
        )
        example = _broken_example(h)
        past += (
            f"\nPrevious attempt {i} ({h.action} {h.sop_id}), NOT shipped: {h.gate_decision}.\n"
            f"Its text: {h.new_text}\n"
            f"Why it failed: {'; '.join(h.gate_reasons)}\n"
            f"Checks that got worse: {worse}\n"
            + (f"A conversation it broke:\n{example}\n" if example else "")
        )
    if past:
        past = (
            "\nEARLIER ATTEMPTS THAT DID NOT WORK. Do not repeat them; work out why they "
            "failed and try a different approach.\n" + past
        )
    user = (
        f"Failed check: {cluster.check}\n"
        f"What it means: {CHECK_DESCRIPTIONS.get(cluster.check, cluster.check)}\n"
        f"It failed in {cluster.count} conversation(s). Examples:\n\n"
        + "\n\n".join(examples)
        + f"\n{past}"
        + f"\n\nCURRENT PROCEDURES\n\n{procedures}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@dataclass
class Proposal:
    action: str
    sop_id: str
    new_text: str
    rationale: str = ""
    title: str = ""
    category: str = "general"
    priority: int = 3
    check: str = ""
    base_version: str = ""
    candidate_version: str = ""
    example_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    # Filled in by `evalgate compare`, so later proposals can learn from this one.
    gate_decision: str = ""
    gate_reasons: list[str] = field(default_factory=list)
    candidate_run: str = ""
    regressions: dict = field(default_factory=dict)  # check -> [base failures, candidate failures]
    broken: list[str] = field(default_factory=list)


def parse_proposal(text: str) -> Proposal:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return Proposal("invalid", "", "", errors=["no JSON object in the Manager's reply"])
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return Proposal("invalid", "", "", errors=[f"bad JSON: {e}"])
    try:
        prio = int(d.get("priority") or 3)
    except (TypeError, ValueError):
        prio = 3
    return Proposal(
        action=str(d.get("action", "")).lower(),
        sop_id=str(d.get("sop_id", "")).strip(),
        new_text=str(d.get("new_text", "")).strip(),
        rationale=str(d.get("rationale", "")).strip(),
        title=str(d.get("title", "")).strip(),
        category=str(d.get("category", "general")).strip() or "general",
        priority=max(1, min(6, prio)),
    )


def validate(p: Proposal, version: AgentVersion, cluster: Cluster) -> Proposal:
    """Hard errors block the proposal. Flags don't, but they stop the gate from auto-promoting."""
    ids = {s.id for s in version.sops}
    if p.action not in ("edit", "add"):
        p.errors.append(f"action must be edit or add, got {p.action!r}")
    if p.action == "edit" and p.sop_id not in ids:
        p.errors.append(f"no procedure with id {p.sop_id!r}")
    if p.action == "add" and (p.sop_id in ids or not re.fullmatch(r"[a-z0-9-]{3,60}", p.sop_id)):
        p.errors.append(f"new procedure id {p.sop_id!r} is taken or not kebab-case")
    if p.action == "add" and not p.title:
        p.errors.append("a new procedure needs a title")
    if not p.new_text or len(p.new_text) > MAX_TEXT:
        p.errors.append(f"procedure text must be 1-{MAX_TEXT} characters")
    if p.action == "edit" and p.sop_id in ids:
        old = next(s.text for s in version.sops if s.id == p.sop_id)
        if old.strip() == p.new_text.strip():
            p.errors.append("the edit doesn't change anything")

    low = p.new_text.lower()
    seen = set().union(*(ex.specifics for ex in cluster.examples)) if cluster.examples else set()
    leaked = sorted(x for x in seen if x in low)
    if leaked:
        p.flags.append(f"mentions specifics from the failing calls: {', '.join(leaked[:5])}")
    if EVALUATOR_WORDS.search(p.new_text):
        p.flags.append("talks about tests or evaluation (possible gaming of the evaluator)")
    p.check = cluster.check
    p.base_version = version.id
    p.example_ids = [f"{ex.scenario_id}-r{ex.run}" for ex in cluster.examples]
    return p


def apply(version: AgentVersion, p: Proposal) -> AgentVersion:
    note = f"manager: fix {p.check}. {p.rationale}"[:300]
    if p.action == "edit":
        return version.with_sop_text(p.sop_id, p.new_text, note=note)
    sop = SOP(id=p.sop_id, title=p.title, category=p.category, priority=p.priority, text=p.new_text)
    return version.with_new_sop(sop, note=note)


def diff(version: AgentVersion, p: Proposal) -> str:
    old = next((s.text for s in version.sops if s.id == p.sop_id), "")
    wrap = lambda t: textwrap.wrap(t, 90) or [""]  # noqa: E731
    return "\n".join(
        difflib.unified_diff(
            wrap(old),
            wrap(p.new_text),
            f"{p.sop_id} (current)",
            f"{p.sop_id} (proposed)",
            lineterm="",
        )
    )


async def propose(
    run_dir: Path,
    version: AgentVersion,
    llm: LLM,
    check: str | None = None,
    versions_dir: Path = Path("versions"),
    proposals_dir: Path = Path("proposals"),
) -> tuple[Proposal, AgentVersion | None, Cluster]:
    clusters = cluster_failures(run_dir)
    if check:
        clusters = [c for c in clusters if c.check == check]
    if not clusters:
        raise ValueError("no failures to work on in the training split of this run")
    cluster = clusters[0]

    history = past_attempts(proposals_dir, version.id)
    text: list[str] = []
    async for ev in llm.stream(manager_messages(version, cluster, history=history), []):
        if isinstance(ev, TextDelta):
            text.append(ev.text)
        elif isinstance(ev, Completion) and not text:
            text.append(ev.text)
    p = validate(parse_proposal("".join(text)), version, cluster)

    candidate = None
    if not p.errors:
        candidate = apply(version, p)
        p.candidate_version = candidate.id
        candidate.save(versions_dir)
    save_proposal(p, proposals_dir, f"rejected-{version.id}-{cluster.check}")
    return p, candidate, cluster


def save_proposal(p: Proposal, proposals_dir: Path, fallback_name: str) -> Path:
    proposals_dir.mkdir(parents=True, exist_ok=True)
    path = proposals_dir / f"{p.candidate_version or fallback_name}.json"
    path.write_text(json.dumps(asdict(p), indent=1))
    return path


def load_proposal(proposals_dir: Path, candidate_id: str) -> Proposal | None:
    path = proposals_dir / f"{candidate_id}.json"
    return Proposal(**json.loads(path.read_text())) if path.exists() else None


__all__ = ["Cluster", "Proposal", "RunResult", "cluster_failures", "propose", "validate"]
