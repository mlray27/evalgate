from fastapi.testclient import TestClient

from evalgate import api, config
from evalgate.llm import ScriptedLLM, Step


def test_websocket_streams_a_turn(settings, monkeypatch):
    monkeypatch.setenv("EVALGATE_DATABASE_URL", settings.database_url)
    monkeypatch.setenv("EVALGATE_TRACE_DIR", settings.trace_dir)
    config.get_settings.cache_clear()
    api.app.state.llm_factory = lambda: ScriptedLLM(
        [
            Step(calls=[("lookup_customer", {"phone": "555-0105"})]),
            Step(text="Hi Grace, how can I help?"),
        ]
    )
    try:
        with TestClient(api.app) as client:
            assert client.get("/health").json()["ok"]
            with client.websocket_connect("/ws/session") as ws:
                hello = ws.receive_json()
                assert hello["type"] == "session"
                ws.send_json({"text": "Hi it's 555-0105"})
                text = ""
                while (msg := ws.receive_json())["type"] == "delta":
                    text += msg["text"]
                assert msg["type"] == "turn_end"
                assert text.strip() == "Hi Grace, how can I help?"
            trace = client.get(f"/sessions/{hello['session_id']}/trace").json()
            assert [e["name"] for e in trace["events"] if e["type"] == "tool"] == [
                "lookup_customer"
            ]
    finally:
        config.get_settings.cache_clear()
        del api.app.state.llm_factory
