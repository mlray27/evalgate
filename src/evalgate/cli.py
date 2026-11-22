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
    from openai import APIError

    settings = get_settings()
    engine, sm = await open_db(settings)
    sess = AgentSession(default_version(settings), build_llm(settings), sm, settings)
    print(f"session {sess.session_id} | agent {sess.version.id} | model {settings.llm_model}")
    print("Type as the caller. To hang up: Ctrl-D (Mac/Linux) or Ctrl-Z then Enter (Windows).\n")
    seen = 0
    try:
        while not sess.done:
            try:
                text = await asyncio.to_thread(input, "caller> ")
            except EOFError:
                break
            if not text.strip():
                continue  # ignore blank lines
            print("agent>  ", end="", flush=True)
            try:
                async for d in sess.respond(text):
                    print(d, end="", flush=True)
            except APIError as e:
                print(f"\n[provider error] {type(e).__name__}: {getattr(e, 'body', None) or e}")
                print("[the trace was saved; paste this error if you need help]")
                break
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
    import json
    from pathlib import Path

    from evalgate.evals import run_suite
    from evalgate.runtime import build_sim_llm
    from evalgate.scenarios import load_suite, select_scenarios

    settings = get_settings()
    ids = args.ids.split(",") if args.ids else None
    scenarios = select_scenarios(load_suite(), args.split, args.category, ids, args.per_category)
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
        print(f"[{done}] {r.scenario_id} r{r.run}: {mark}", flush=True)

    out_dir = Path(args.resume) if args.resume else None
    agent_model = "oracle" if args.oracle else settings.llm_model
    print(f"{total} conversations, agent {version.id}, model {agent_model}")
    if out_dir:
        print(f"resuming into {out_dir}")
    _, out = await run_suite(
        scenarios,
        version,
        agent_factory,
        sim_factory,
        settings,
        runs=args.runs,
        concurrency=args.concurrency,
        out_dir=out_dir,
        progress=progress,
    )
    print()
    print((out / "report.md").read_text().split("## By category")[0])
    print(f"full report: {out / 'report.md'}")
    summary = json.loads((out / "summary.json").read_text())
    if summary.get("stopped_on_rate_limit"):
        again = [a for a in sys.argv[1:] if not a.startswith("--resume")]
        if "--resume" in sys.argv:
            i = sys.argv.index("--resume")
            again = sys.argv[1:i] + sys.argv[i + 2 :]
        print()
        print("Stopped: a model provider's rate limit didn't clear after waiting.")
        print(f"  {summary.get('rate_limit_error')}")
        print(f"{summary['remaining']} conversations left. When the quota resets, run:")
        print(f"  uv run evalgate {' '.join(again)} --resume {out}")


async def check() -> None:
    """Verify keys, model ids, and tool calling for the agent and simulator endpoints."""
    from openai import APIError, AsyncOpenAI

    st = get_settings()
    endpoints = [
        ("agent", st.llm_base_url, st.llm_api_key, st.llm_model, True),
        (
            "simulator",
            st.sim_base_url or st.llm_base_url,
            st.sim_api_key or st.llm_api_key,
            st.sim_model or st.llm_model,
            False,
        ),
    ]
    ok = True
    for role, url, key, model, needs_tools in endpoints:
        print(f"\n[{role}] {model} @ {url}")
        if not key:
            print("  no API key set")
            ok = False
            continue
        client = AsyncOpenAI(base_url=url, api_key=key, max_retries=1, timeout=30)
        try:
            ids = [m.id async for m in client.models.list()]
            short = [i.split("/")[-1] for i in ids]
            if model in ids or model.split("/")[-1] in short:
                print("  model found")
            else:
                print("  model NOT in this provider's list. Some available ids:")
                for i in ids[:25]:
                    print(f"    {i}")
        except APIError as e:
            print(f"  could not list models ({type(e).__name__}); trying a request anyway")
        try:
            if needs_tools:
                # A real two-turn exchange: the model calls a tool, gets the result, and replies.
                # This is what catches provider quirks a single request won't (e.g. Gemini
                # rejecting a follow-up that drops its thought signature).
                import tempfile

                from evalgate.agent import AgentSession
                from evalgate.db import make_engine, make_sessionmaker
                from evalgate.seed import seed

                with tempfile.TemporaryDirectory() as d:
                    engine = make_engine(f"sqlite+aiosqlite:///{d}/check.db")
                    sm = make_sessionmaker(engine)
                    await seed(engine, sm, st.today, st.seed)
                    sess = AgentSession(default_version(st), build_llm(st), sm, st)
                    try:
                        reply = await sess.say("Hi, my account number is 555-0101.")
                    finally:
                        await engine.dispose()
                tools_used = [e["name"] for e in sess.trace.of_type("tool")]
                if "lookup_customer" in tools_used and reply:
                    print(f"  tool calling works: called {tools_used}, replied {reply[:80]!r}")
                else:
                    print(f"  WARNING: tools {tools_used}, reply {reply[:80]!r}")
                    ok = False
            else:
                r = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": "Say hi in three words."}]
                )
                print(f"  chat works: {r.choices[0].message.content!r}")
        except APIError as e:
            ok = False
            hint = {
                "AuthenticationError": "the API key is wrong",
                "PermissionDeniedError": "the key can't use this model",
                "NotFoundError": "the model id is wrong",
                "RateLimitError": "rate limited; wait a minute or check your quota",
            }.get(type(e).__name__, "")
            print(f"  FAILED: {type(e).__name__} {hint}".rstrip())
            print(f"  {str(getattr(e, 'body', None) or e)[:400]}")
    print("\nall good" if ok else "\nfix the problems above, then run this again")


def main() -> None:
    p = argparse.ArgumentParser(prog="evalgate")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chat", help="talk to the agent as a caller")
    c.add_argument("--tools", action="store_true", help="print tool calls and guard results")
    sub.add_parser("seed", help="reset the database to the seed state")
    sub.add_parser("gen-scenarios", help="regenerate scenarios/suite.yaml from templates")
    sub.add_parser("check", help="test your API keys, model ids and tool calling")
    e = sub.add_parser("eval", help="run simulated callers against the agent")
    e.add_argument("--split", default="train", choices=["train", "heldout", "all"])
    e.add_argument("--category")
    e.add_argument("--ids", help="comma-separated scenario ids")
    e.add_argument("--limit", type=int, help="only the first N matching scenarios")
    e.add_argument("--per-category", type=int, help="first N scenarios from each category")
    e.add_argument("--resume", metavar="RUN_DIR", help="continue a stopped run in this folder")
    e.add_argument("--runs", type=int, default=1, help="runs per scenario")
    e.add_argument("--concurrency", type=int)
    e.add_argument(
        "--oracle",
        action="store_true",
        help="validate the suite with a ground-truth agent (no API key needed)",
    )
    args = p.parse_args()
    try:
        _dispatch(args)
    except RuntimeError as e:  # config problems: show the message, not a traceback
        sys.exit(f"error: {e}")
    except KeyboardInterrupt:
        sys.exit(130)


def _dispatch(args: argparse.Namespace) -> None:
    if args.cmd == "check":
        asyncio.run(check())
    elif args.cmd == "chat":
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
