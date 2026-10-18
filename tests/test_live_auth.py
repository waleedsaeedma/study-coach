from fastapi.testclient import TestClient

from live.auth import live_token, token_ok
from live.server import app

client = TestClient(app)


def test_laptop_needs_no_token(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    assert live_token() is None
    assert token_ok(None)


def test_online_needs_the_right_token(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "secret")
    assert not token_ok(None)
    assert not token_ok("wrong")
    assert token_ok(live_token())


def test_server_refuses_calls_without_the_token(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "secret")
    assert client.post("/session", json={"name": "sara"}).status_code == 401
    assert client.post("/tool", json={"name": "sara", "tool": "nope"}).status_code == 401
    assert client.post("/tool", json={"name": "sara", "tool": "nope"},
                       headers={"X-Live-Token": "wrong"}).status_code == 401
    ok = client.post("/tool", json={"name": "sara", "tool": "nope"}, headers={"X-Live-Token": live_token()})
    assert ok.status_code == 200 and ok.json()["output"] == "Unknown tool: nope"
