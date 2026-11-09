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


def test_new_actions_reach_youtube_music(client, monkeypatch):
    sent = []
    monkeypatch.setattr(music, "_cmd", lambda c, d=None: sent.append((c, d)))
    monkeypatch.setattr(music, "music_output", lambda a: type("R", (), {"ok": True, "data": {"where": a["where"]}})())
    for body in [{"action": "shuffle"}, {"action": "repeat", "repeat": 2}, {"action": "play_index", "index": 7},
                 {"action": "output", "where": "mac"}]:
        assert client.post("/api/music/control", json=body).status_code == 200, body
    assert sent == [("shuffle", None), ("repeatMode", 2), ("playQueueIndex", 7)]


def test_now_playing_is_his_volume_and_patched_at_once(monkeypatch):
    st = {"video": {"title": "Song", "durationSeconds": 200, "thumbnails": []},
          "player": {"trackState": 0, "volume": 20, "videoProgress": 10,
                     "queue": {"selectedItemIndex": 0, "repeatMode": 0,
                               "items": [{"title": f"T{i}", "author": "A", "thumbnails": []} for i in range(40)]}}}
    monkeypatch.setattr(music, "_state", {"value": st, "at": music.time.monotonic()})
    monkeypatch.setattr(music, "_watch", {"last": 0.0, "thread": object()})  # no refresher thread in tests
    monkeypatch.setattr(music, "_duck", {"restore": 70})                      # THURSDAY is talking: music at 20
    n = music.now_playing()
    assert n["volume"] == 70 and n["ducked"] and len(n["up_next"]) == 25 and n["up_next"][0]["title"] == "T1"
    music.patch(playing=True, volume=55, progress=120, repeat=1)
    monkeypatch.setattr(music, "_duck", {"restore": None})
    n = music.now_playing()
    assert n["playing"] and n["volume"] == 55 and n["progress"] == 120 and n["repeat"] == 1


def test_refresher_sleeps_between_reads_instead_of_spinning(monkeypatch):
    """It held the GIL in a busy loop for ~5 s of every 5.2 (2026-11-08): count how often it checks the player."""
    import threading
    import time
    import music
    calls = []
    monkeypatch.setattr(music, "STATE_EVERY_S", 0.2)
    monkeypatch.setattr(music, "WATCH_S", 0.7)
    monkeypatch.setattr(music, "state", lambda fresh=False: calls.append(1) or music._state.update(at=time.monotonic()) or {})
    music._state["at"] = time.monotonic()
    music._watch["last"] = time.monotonic()
    t = threading.Thread(target=music._refresher)
    t.start()
    t.join(2)
    assert 1 <= len(calls) <= 5
