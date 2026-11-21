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


def main() -> None:
    p = argparse.ArgumentParser(prog="evalgate")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chat", help="talk to the agent as a caller")
    c.add_argument("--tools", action="store_true", help="print tool calls and guard results")
    sub.add_parser("seed", help="reset the database to the seed state")
    args = p.parse_args()
    if args.cmd == "chat":
        asyncio.run(chat(args.tools))
    elif args.cmd == "seed":
        asyncio.run(reseed())
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
