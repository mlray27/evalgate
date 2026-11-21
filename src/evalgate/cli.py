"""Command line: `evalgate chat` to talk to the agent, `evalgate seed` to reset the database."""

from __future__ import annotations

import argparse
import asyncio
import sys

from evalgate.agent import AgentSession
from evalgate.config import get_settings
from evalgate.runtime import build_llm, default_version, open_db
from evalgate.seed import seed


async def chat(show_tools: bool) -> None:
    settings = get_settings()
    engine, sm = await open_db(settings)
    sess = AgentSession(default_version(settings), build_llm(settings), sm, settings)
    print(f"session {sess.session_id} | agent {sess.version.id} | model {settings.llm_model}")
    print("Type as the caller. Ctrl-D to hang up.\n")
    seen = 0
    try:
        while not sess.done:
            try:
                text = await asyncio.to_thread(input, "caller> ")
            except EOFError:
                break
            print("agent>  ", end="", flush=True)
            async for d in sess.respond(text):
                print(d, end="", flush=True)
            print()
            if show_tools:
                for ev in sess.trace.of_type("tool")[seen:]:
                    print(f"   [{ev['kind']}] {ev['name']}({ev['args']}) -> {ev['result']}")
                seen = len(sess.trace.of_type("tool"))
    finally:
        path = sess.trace.save(settings.trace_dir)
        print(f"\ntrace saved to {path}")
        await engine.dispose()


async def reseed() -> None:
    settings = get_settings()
    engine, sm = await open_db(settings)
    await seed(engine, sm, settings.today, settings.seed)
    await engine.dispose()
    print("database reset to seed state")


async def run_eval(args: argparse.Namespace) -> None:
    from evalgate.evals import run_suite
    from evalgate.runtime import build_sim_llm
    from evalgate.scenarios import load_suite, select_scenarios

    settings = get_settings()
    ids = args.ids.split(",") if args.ids else None
    scenarios = select_scenarios(load_suite(), args.split, args.category, ids)
    if args.limit:
        scenarios = scenarios[: args.limit]
    if not scenarios:
        sys.exit("no scenarios match")
    version = default_version(settings)
    if args.oracle:
        from evalgate.oracle import OracleAgent, OracleCaller

        agent_factory, sim_factory = OracleAgent, lambda _: OracleCaller()
    else:
        agent_llm, sim_llm = build_llm(settings), build_sim_llm(settings)
        agent_factory, sim_factory = (lambda _: agent_llm), (lambda _: sim_llm)
    total = len(scenarios) * args.runs
    done = 0

    def progress(r) -> None:
        nonlocal done
        done += 1
        mark = "pass" if r.success else "FAIL " + ",".join(r.failed_checks or [r.error or ""])
        print(f"[{done}/{total}] {r.scenario_id} r{r.run}: {mark}", flush=True)

    print(f"running {total} conversations, agent {version.id}, model {settings.llm_model}")
    _, out = await run_suite(
        scenarios,
        version,
        agent_factory,
        sim_factory,
        settings,
        runs=args.runs,
        concurrency=args.concurrency,
        progress=progress,
    )
    print()
    print((out / "report.md").read_text().split("## By category")[0])
    print(f"full report: {out / 'report.md'}")


def main() -> None:
    p = argparse.ArgumentParser(prog="evalgate")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chat", help="talk to the agent as a caller")
    c.add_argument("--tools", action="store_true", help="print tool calls and guard results")
    sub.add_parser("seed", help="reset the database to the seed state")
    sub.add_parser("gen-scenarios", help="regenerate scenarios/suite.yaml from templates")
    e = sub.add_parser("eval", help="run simulated callers against the agent")
    e.add_argument("--split", default="train", choices=["train", "heldout", "all"])
    e.add_argument("--category")
    e.add_argument("--ids", help="comma-separated scenario ids")
    e.add_argument("--limit", type=int, help="only the first N matching scenarios")
    e.add_argument("--runs", type=int, default=1, help="runs per scenario")
    e.add_argument("--concurrency", type=int)
    e.add_argument(
        "--oracle",
        action="store_true",
        help="validate the suite with a ground-truth agent (no API key needed)",
    )
    args = p.parse_args()
    if args.cmd == "chat":
        asyncio.run(chat(args.tools))
    elif args.cmd == "seed":
        asyncio.run(reseed())
    elif args.cmd == "gen-scenarios":
        from evalgate.scenarios import generate

        suite = generate(get_settings())
        print(f"wrote {len(suite)} scenarios")
    elif args.cmd == "eval":
        asyncio.run(run_eval(args))
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
