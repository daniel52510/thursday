"""Night mode (night.py): who decides it's night, what "good night" / "good morning" change, the phrases that start
the routines, and the bedtime nudge's math."""
from datetime import datetime, timedelta

import pytest

import night
import presence


@pytest.fixture
def clock(monkeypatch, tmp_path):
    monkeypatch.setattr(night, "FILE", str(tmp_path / "night.json"))
    monkeypatch.setattr(night, "_woke", lambda now: False)
    t = {"now": datetime(2026, 11, 9, 21, 30, tzinfo=night._zone())}
    monkeypatch.setattr(night, "_now", lambda: t["now"])
    return t


def test_clock_decides_without_an_override(clock):
    assert not night.is_night(clock["now"])                       # 21:30
    assert night.is_night(clock["now"].replace(hour=23))
    assert night.is_night(clock["now"].replace(hour=6)) and not night.is_night(clock["now"].replace(hour=7))
    assert night.quiet(clock["now"].replace(hour=7, minute=30))   # held a little past the night: just up


def test_good_night_early_and_good_morning_early(clock):
    night.set_mode("night", "you said good night")                # at 21:30
    assert night.is_night(clock["now"]) and night.quiet(clock["now"])
    assert night.is_night(clock["now"] + timedelta(hours=10))     # still night at 07:30 if he hasn't said otherwise
    assert not night.is_night(clock["now"] + timedelta(hours=14))  # 11:30: over on its own
    clock["now"] = clock["now"].replace(hour=6) + timedelta(days=1)
    night.set_mode("day", "you said good morning")                # 06:00, inside the clock's night
    assert not night.is_night(clock["now"]) and not night.quiet(clock["now"])
    assert night.is_night(clock["now"].replace(hour=22, minute=30))  # tonight the clock is back in charge


def test_up_in_the_middle_of_the_night_is_three_hours(clock):
    clock["now"] = clock["now"].replace(hour=23)
    night.set_mode("day", "you")
    assert not night.is_night(clock["now"] + timedelta(hours=2)) and night.is_night(clock["now"] + timedelta(hours=4))


def test_watch_ends_the_night(clock, monkeypatch):
    monkeypatch.setattr(night, "_woke", lambda now: True)
    assert not night.is_night(clock["now"].replace(hour=6))


def test_phrases_that_start_the_routines():
    for said in ("Good night", "goodnight Thursday", "okay, I'm going to bed", "heading to sleep now", "bedtime"):
        assert night.GOOD_NIGHT.match(said), said
    for said in ("good morning", "I'm up", "Morning, Thursday"):
        assert night.GOOD_MORNING.match(said), said
    for said in ("what did I do last night?", "is it a good night for stars?", "set an alarm for the morning"):
        assert not night.GOOD_NIGHT.match(said) and not night.GOOD_MORNING.match(said), said


def test_presence_holds_by_night_mode():
    base = {"asleep": False, "meeting": None, "phone": "home", "satellite": True, "hour": 21}
    assert presence.channel({**base, "quiet": False}) == "speak"
    assert presence.channel({**base, "quiet": True}) == "hold"       # he said good night at 9
    assert presence.channel({**base, "hour": 6, "quiet": False}) == "speak"  # he said good morning at 6


def test_bedtime_is_the_sleep_target_before_the_first_alarm(clock, monkeypatch):
    monkeypatch.setattr(night, "_first_alarm", lambda after: (after + timedelta(days=1)).replace(hour=6, minute=30))
    assert night.bedtime(clock["now"]).strftime("%H:%M") == "22:30"
    monkeypatch.setattr(night, "_first_alarm", lambda after: None)
    assert night.bedtime(clock["now"]).strftime("%H:%M") == "23:00"  # THURSDAY_WAKE 07:00, 8 hours


def test_watch_and_alarm_end_a_good_night(clock, monkeypatch):
    night.set_mode("night", "you said good night")                      # 21:30
    clock["now"] = clock["now"].replace(hour=7, minute=40) + timedelta(days=1)
    assert night.is_night(clock["now"])                                  # no sign he's up: still night till 11
    monkeypatch.setattr(night, "_woke", lambda now: True)
    assert not night.is_night(clock["now"]) and not night.quiet(clock["now"])  # his Watch says he's up
