"""Trust gate: decides whether a candidate agent version may replace its parent.

The gate compares two eval runs on the same scenarios and looks at the held-out split above all,
because the Agent Manager only ever saw training failures. A fix that helps on train but not on
held-out callers learned the examples, not the behavior.

Decisions:
  promote      clear held-out gain, almost nothing broken, no warning flags
  needs_human  plausibly good, but the evidence is thin or something looks off
  reject       held-out got worse, or a safety-critical scenario broke
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from evalgate.evals import RunResult, load_results

SAFETY_CATEGORIES = {"safety_gas", "emergency_no_heat"}


@dataclass
class Thresholds:
    min_heldout_gain_pp: float = 5.0
    max_broken: int = 1
    max_generalization_gap_pp: float = 15.0
    min_heldout_n: int = 10
    # Outcome checks (right customer, address, slot...) may not get worse by more than this many
    # conversations, even when those conversations were already failing for other reasons.
    max_outcome_regression: int = 1


@dataclass
class SplitStats:
    n: int = 0
    base_rate: float = 0.0
    cand_rate: float = 0.0
    fixed: list[str] = field(default_factory=list)
    broken: list[str] = field(default_factory=list)

    @property
    def delta_pp(self) -> float:
        return round(self.cand_rate - self.base_rate, 1)

    @property
    def sign_test_p(self) -> float | None:
        return sign_test(len(self.fixed), len(self.broken))


def sign_test(wins: int, losses: int) -> float | None:
    """Two-sided exact sign test on paired outcomes that changed. None if nothing changed."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return round(min(1.0, 2 * p), 4)


def _key(r: RunResult) -> tuple[str, int]:
    return (r.scenario_id, r.run)


def _usable(r: RunResult) -> bool:
    return not r.sim_flags and r.error is None


def _scenario_defs(run_dir: Path) -> dict[tuple[str, int], str]:
    import json

    out = {}
    for p in (run_dir / "traces").glob("*.json"):
        t = json.loads(p.read_text())
        r = t["result"]
        out[(r["scenario_id"], r["run"])] = json.dumps(t["scenario"], sort_keys=True)
    return out


def compare(base_dir: Path, cand_dir: Path) -> dict:
    """Paired comparison on scenarios that both runs completed cleanly.

    Pairs whose scenario definition differs between the runs (the suite was regenerated in
    between) are skipped: they aren't the same test anymore.
    """
    base = {_key(r): r for r in load_results(base_dir) if _usable(r)}
    cand = {_key(r): r for r in load_results(cand_dir) if _usable(r)}
    bdefs, cdefs = _scenario_defs(base_dir), _scenario_defs(cand_dir)
    changed = {
        k for k in base.keys() & cand.keys() if k in bdefs and k in cdefs and bdefs[k] != cdefs[k]
    }
    common = sorted((base.keys() & cand.keys()) - changed)
    splits: dict[str, SplitStats] = {}
    check_delta: dict[str, list[int]] = {}
    broken_categories: set[str] = set()
    for k in common:
        b, c = base[k], cand[k]
        st = splits.setdefault(b.split, SplitStats())
        st.n += 1
        st.base_rate += b.success
        st.cand_rate += c.success
        label = f"{k[0]}-r{k[1]}"
        if c.success and not b.success:
            st.fixed.append(label)
        elif b.success and not c.success:
            st.broken.append(label)
            broken_categories.add(b.category)
        for name in {x["name"] for x in b.checks} | {x["name"] for x in c.checks}:
            bf = any(x["name"] == name and not x["passed"] for x in b.checks)
            cf = any(x["name"] == name and not x["passed"] for x in c.checks)
            check_delta.setdefault(name, [0, 0])
            check_delta[name][0] += bf
            check_delta[name][1] += cf
    for st in splits.values():
        st.base_rate = round(100 * st.base_rate / st.n, 1) if st.n else 0.0
        st.cand_rate = round(100 * st.cand_rate / st.n, 1) if st.n else 0.0
    train, held = splits.get("train", SplitStats()), splits.get("heldout", SplitStats())
    return {
        "paired_conversations": len(common),
        "skipped_changed_scenarios": len(changed),
        "train": train,
        "heldout": held,
        "generalization_gap_pp": round(train.delta_pp - held.delta_pp, 1),
        "broken_categories": sorted(broken_categories),
        "check_failures": {
            k: {"base": v[0], "candidate": v[1]}
            for k, v in sorted(check_delta.items())
            if v[0] or v[1]
        },
    }


def decide(cmp: dict, flags: list[str] | None = None, t: Thresholds | None = None) -> dict:
    t = t or Thresholds()
    flags = flags or []
    train, held = cmp["train"], cmp["heldout"]
    broken = len(train.broken) + len(held.broken)
    reasons: list[str] = []

    if held.n == 0:
        return {
            "decision": "needs_human",
            "reasons": ["no held-out results: run the candidate with --split all or heldout"],
        }
    if held.delta_pp < 0:
        reasons.append(f"held-out success went down ({held.delta_pp:+} pp)")
    fixed = len(train.fixed) + len(held.fixed)
    broken_total = len(train.broken) + len(held.broken)
    if broken_total > fixed:
        reasons.append(
            f"net harm: broke {broken_total} conversations that used to pass, fixed {fixed}"
        )
    from evalgate.graders import PROCESS_CHECKS

    for name, v in cmp.get("check_failures", {}).items():
        if name not in PROCESS_CHECKS and v["candidate"] - v["base"] > t.max_outcome_regression:
            reasons.append(
                f"outcome regression: {name} failed in {v['base']} -> "
                f"{v['candidate']} conversations"
            )
    safety = set(cmp["broken_categories"]) & SAFETY_CATEGORIES
    if safety:
        reasons.append(f"broke a safety-critical scenario ({', '.join(sorted(safety))})")
    if reasons:
        return {"decision": "reject", "reasons": reasons}

    if held.delta_pp < t.min_heldout_gain_pp:
        reasons.append(f"held-out gain {held.delta_pp:+} pp is below {t.min_heldout_gain_pp} pp")
    if broken > t.max_broken:
        reasons.append(f"{broken} previously passing conversations now fail (max {t.max_broken})")
    if cmp["generalization_gap_pp"] > t.max_generalization_gap_pp:
        reasons.append(
            f"train improved {cmp['generalization_gap_pp']} pp more than held-out: "
            f"looks like it learned the examples"
        )
    if held.n < t.min_heldout_n:
        reasons.append(f"only {held.n} held-out conversations (want {t.min_heldout_n}+)")
    reasons.extend(f"proposal flag: {f}" for f in flags)
    if reasons:
        return {"decision": "needs_human", "reasons": reasons}
    return {
        "decision": "promote",
        "reasons": [
            f"held-out {held.delta_pp:+} pp, {broken} broken, gap {cmp['generalization_gap_pp']} pp"
        ],
    }


def render(cmp: dict, decision: dict, base_id: str, cand_id: str, proposal=None) -> str:
    def row(name: str, st: SplitStats) -> str:
        p = st.sign_test_p
        return (
            f"| {name} | {st.n} | {st.base_rate}% | {st.cand_rate}% | {st.delta_pp:+} | "
            f"{len(st.fixed)} | {len(st.broken)} | {p if p is not None else '-'} |"
        )

    lines = [f"# Candidate {cand_id} vs {base_id}", ""]
    if proposal:
        lines += [
            f"**Proposal:** {proposal.action} `{proposal.sop_id}` to fix "
            f"`{proposal.check}`. {proposal.rationale}",
            "",
            "```",
            proposal.new_text,
            "```",
            "",
        ]
    lines += [
        f"## Decision: **{decision['decision']}**",
        "",
        *[f"- {r}" for r in decision["reasons"]],
        "",
        "| Split | n | Base | Candidate | Delta (pp) | Fixed | Broken | Sign test p |",
        "|---|---|---|---|---|---|---|---|",
        row("train", cmp["train"]),
        row("held-out", cmp["heldout"]),
        "",
        f"Generalization gap (train delta minus held-out delta): {cmp['generalization_gap_pp']} pp",
        f"Paired conversations: {cmp.get('paired_conversations', '?')}; skipped because the "
        f"scenario changed between runs: {cmp.get('skipped_changed_scenarios', 0)}",
        "",
        "Small samples are noisy: a sign test p above 0.05 means the change could be chance.",
        "",
        "## Check failures (conversations failing each check)",
        "",
        "| Check | Base | Candidate |",
        "|---|---|---|",
        *[f"| {k} | {v['base']} | {v['candidate']} |" for k, v in cmp["check_failures"].items()],
        "",
    ]
    for name in ("train", "heldout"):
        st = cmp[name]
        if st.fixed:
            lines.append(f"Fixed ({name}): {', '.join(st.fixed)}")
        if st.broken:
            lines.append(f"Broken ({name}): {', '.join(st.broken)}")
    return "\n".join(lines) + "\n"


def compare_runs(
    base_dir: Path,
    cand_dir: Path,
    proposals_dir: Path = Path("proposals"),
    t: Thresholds | None = None,
) -> tuple[dict, dict, Path]:
    import json

    from evalgate.manager import load_proposal

    base_id = json.loads((base_dir / "version.json").read_text())["id"]
    cand_id = json.loads((cand_dir / "version.json").read_text())["id"]
    proposal = load_proposal(proposals_dir, cand_id)
    cmp = compare(base_dir, cand_dir)
    decision = decide(cmp, proposal.flags if proposal else [], t)
    out = cand_dir / f"comparison-vs-{base_id}.md"
    out.write_text(render(cmp, decision, base_id, cand_id, proposal))
    if proposal is not None:
        # Remember the outcome so the Manager's next proposal can learn from it.
        from dataclasses import asdict

        proposal.gate_decision = decision["decision"]
        proposal.gate_reasons = decision["reasons"]
        proposal.candidate_run = str(cand_dir)
        proposal.broken = cmp["train"].broken + cmp["heldout"].broken
        proposal.regressions = {
            k: [v["base"], v["candidate"]]
            for k, v in cmp["check_failures"].items()
            if v["candidate"] > v["base"]
        }
        (proposals_dir / f"{cand_id}.json").write_text(json.dumps(asdict(proposal), indent=1))
    return cmp, decision, out
