"""The web player (/api/music/*) and voice volume/speed settings (/api/voice)."""
import pytest
from fastapi.testclient import TestClient

import music
import server

OWNER = {"id": "owner", "role": "admin", "username": "daniel", "display_name": "Daniel"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    settings = {}

    class DB:
        def get_settings(self, uid): return dict(settings)
        def set_setting(self, uid, name, value): settings[name] = int(value)
    monkeypatch.setattr(server, "initalize_db", lambda: DB())
    monkeypatch.setattr(server, "voice_style", lambda uid: {"voice": "leo"})
    server.app.dependency_overrides[server.current_user] = lambda: OWNER
    server.app.dependency_overrides[server.admin_only] = lambda: OWNER
    yield TestClient(server.app)
    server.app.dependency_overrides.clear()


def test_voice_volume_and_speed(client):
    assert client.get("/api/voice").json() == {"voice": "rocky", "volume": 100, "speed": 100}
    assert client.put("/api/voice", json={"volume": 55}).json()["volume"] == 55
    assert client.put("/api/voice", json={"speed": 90}).json() == {"voice": "rocky", "volume": 55, "speed": 90}
    assert client.put("/api/voice", json={"volume": 0}).status_code == 422      # 1-100 only
    assert client.put("/api/voice", json={"speed": 200}).status_code == 422


def test_music_now_and_controls(client, monkeypatch):
    monkeypatch.setattr(music, "now_playing", lambda: {"connected": True, "title": "Poor Man's Son", "playing": False})
    assert client.get("/api/music/now").json()["title"] == "Poor Man's Son"
    sent = []
    monkeypatch.setattr(music, "music_control", lambda a: sent.append(a) or type("R", (), {"ok": True})())
    monkeypatch.setattr(music, "_cmd", lambda c, d=None: sent.append((c, d)))
    assert client.post("/api/music/control", json={"action": "resume"}).status_code == 200
    assert client.post("/api/music/control", json={"action": "set_volume", "level": 40}).status_code == 200
    assert client.post("/api/music/control", json={"action": "seek", "seconds": 61}).status_code == 200
    assert client.post("/api/music/control", json={"action": "shutdown"}).status_code == 422
    assert sent == [{"action": "resume", "level": None}, {"action": "set_volume", "level": 40}, ("seekTo", 61.0)]
    monkeypatch.setattr(music, "now_playing", lambda: (_ for _ in ()).throw(RuntimeError("not paired")))
    assert client.get("/api/music/now").json() == {"connected": False, "error": "RuntimeError"}
