"""FastAPI app: streaming conversations over WebSocket with barge-in."""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from evalgate.agent import AgentSession
from evalgate.config import get_settings
from evalgate.runtime import build_llm, default_version, open_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    engine, sm = await open_db(settings)
    app.state.settings = settings
    app.state.sm = sm
    app.state.version = default_version(settings)
    app.state.llm_factory = getattr(app.state, "llm_factory", None) or (lambda: build_llm(settings))
    app.state.sessions = {}
    yield
    await engine.dispose()


app = FastAPI(title="EvalGate", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "agent_version": app.state.version.id}


@app.get("/sessions/{session_id}/trace")
async def get_trace(session_id: str) -> dict:
    sess = app.state.sessions.get(session_id)
    if not sess:
        raise HTTPException(404)
    return sess.trace.to_dict()


@app.websocket("/ws/session")
async def ws_session(ws: WebSocket) -> None:
    """Protocol: client sends {"text": "..."}; server streams {"type": "delta"} then "turn_end".

    A new caller message while the agent is still talking cancels the current turn (barge-in).
    """
    await ws.accept()
    st = app.state
    sess = AgentSession(st.version, st.llm_factory(), st.sm, st.settings)
    st.sessions[sess.session_id] = sess
    await ws.send_json(
        {"type": "session", "session_id": sess.session_id, "version": sess.version.id}
    )

    async def run_turn(text: str) -> None:
        async for delta in sess.respond(text):
            await ws.send_json({"type": "delta", "text": delta})
        await ws.send_json({"type": "turn_end", "done": sess.done})

    current: asyncio.Task | None = None
    try:
        while True:
            msg = await ws.receive_json()
            if current and not current.done():
                current.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await current
                await ws.send_json({"type": "interrupted"})
            current = asyncio.create_task(run_turn(msg["text"]))
    except WebSocketDisconnect:
        pass
    finally:
        if current and not current.done():
            current.cancel()
        sess.trace.save(st.settings.trace_dir)
