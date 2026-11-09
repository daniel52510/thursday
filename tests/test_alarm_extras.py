"""Wake-up extras (alarms.py, night.py): each one a switch. Training in the briefing, the phone as backup, music
fading in instead of the chime, and the good-night alarm check that a plain yes answers."""
import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

import alarms
import asks
import night

TZ = night._zone()


@pytest.fixture
def prefs(monkeypatch, tmp_path):
    p = {"alarm_training": 1, "alarm_phone": 1, "alarm_music": 0, "alarm_check": 1, "music_query": ""}
    monkeypatch.setattr(alarms, "prefs", lambda uid: p)
    monkeypatch.setattr(asks, "DB", str(tmp_path / "asks.db"))
    return p


ALARM = {"id": "a1", "time": "07:15", "timezone": "America/New_York", "brief": "", "user_id": "owner", "style": "briefing",
         "days": [0], "enabled": True}


def test_training_in_the_briefing_only_when_on(prefs):
    at = datetime(2026, 11, 10, 12, 15, tzinfo=TZ)
    assert "training_today" in alarms.briefing_prompt(ALARM, at)
    prefs["alarm_training"] = 0
    assert "training_today" not in alarms.briefing_prompt(ALARM, at)


def test_phone_backup_once_and_only_when_on(prefs, monkeypatch):
    import approvals
    pushed = []
    monkeypatch.setattr(approvals, "push", lambda title, text, **kw: pushed.append((title, kw["priority"])))
    alarms._phoned.clear()
    alarms._phone(ALARM, ("k",), "not home")
    alarms._phone(ALARM, ("k",), "not home")
    assert pushed == [("Alarm 07:15", 5)]
    prefs["alarm_phone"] = 0
    alarms._phone(ALARM, ("k2",), "x")
    assert len(pushed) == 1


def test_music_fades_in_first_then_he_speaks_without_the_chime(prefs, monkeypatch):
    prefs.update(alarm_music=1, music_query="Morning Coffee playlist")
    calls = []

    async def fade(q):
        calls.append(("music", q))
    monkeypatch.setattr(alarms, "_music_fade_in", fade)
    monkeypatch.setattr(alarms, "MUSIC_LEAD_S", 0.05)
    monkeypatch.setattr(alarms, "ring_prompt", lambda a, at: ("brief me", "briefing"))
    monkeypatch.setattr(alarms, "initalize_db", lambda: SimpleNamespace(create_conversation=lambda t, u: {"id": "c"}))
    monkeypatch.setattr(night, "set_mode", lambda *a: None)

    async def bcast(force=False):
        pass
    monkeypatch.setattr(night, "broadcast", bcast)

    class Sat:
        heard_at = 0.0
        _context = {}

        async def announce(self, **kw):
            calls.append(("announce", kw.get("chime"), kw.get("alarm_try")))
            return True
    monkeypatch.setattr(alarms, "SATELLITES", {Sat()})
    monkeypatch.setattr(alarms, "_briefing_heard", lambda *a: asyncio.sleep(0))
    at = datetime(2026, 11, 10, 12, 15, tzinfo=TZ)

    async def go():
        first = await alarms._ring(ALARM, at, at)
        await asyncio.sleep(0.1)
        second = await alarms._ring(ALARM, at, at)
        return first, second
    alarms._music_started.clear()
    assert asyncio.run(go()) == (False, True)
    assert calls[0] == ("music", "Morning Coffee playlist") and calls[-1] == ("announce", 0.0, 1)


NOW = datetime(2026, 11, 9, 22, 0, tzinfo=TZ)


@pytest.mark.parametrize("alarm,start,want", [
    (NOW.replace(hour=7, minute=45) + timedelta(days=1), NOW.replace(hour=8) + timedelta(days=1), "06:30"),  # too late
    (None, NOW.replace(hour=9, minute=10) + timedelta(days=1), "07:40"),                                   # none set
    (NOW.replace(hour=6) + timedelta(days=1), NOW.replace(hour=9) + timedelta(days=1), None),              # plenty of room
    (None, NOW.replace(hour=14) + timedelta(days=1), None),                                                # afternoon: sleep in
    (None, None, None),                                                                                     # nothing on
])
def test_alarm_check(alarm, start, want):
    out = night.alarm_check(NOW, alarm, "Dentist", start)
    assert (out or {}).get("time") == want


def test_good_night_asks_and_a_plain_yes_sets_the_alarm(prefs, monkeypatch, tmp_path):
    monkeypatch.setattr(night, "FILE", str(tmp_path / "night.json"))
    monkeypatch.setattr(night, "_now", lambda: NOW)
    monkeypatch.setattr(night, "_lights_down", lambda: "dimmed")
    monkeypatch.setattr(night, "_first_alarm", lambda now: NOW.replace(hour=7, minute=45) + timedelta(days=1))
    monkeypatch.setattr(night, "_first_event_at", lambda day: ("Dentist", NOW.replace(hour=8) + timedelta(days=1)))
    import music
    monkeypatch.setattr(music, "state", lambda *a: {})
    done = night.good_night()
    assert done["ask_him"].endswith("Want an alarm at 6:30?")
    created = []
    import agent_loop
    monkeypatch.setattr(agent_loop, "initalize_db", lambda: SimpleNamespace(create_alarm=lambda *a, **kw: created.append((a, kw))))
    heard = asks.voice_answer("yes")
    assert heard and heard[1] == "set"
    assert asks.answer(*heard)["result"] == "alarm set for 6:30 AM"
    assert created[0][0][:2] == ("06:30", []) and created[0][1]["style"] == "wake"   # one-time, she wakes him
    prefs["alarm_check"] = 0
    assert "ask_him" not in night.good_night()
