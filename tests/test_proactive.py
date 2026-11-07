"""The proactive engine (proactive.py): it speaks first, once per thing, and keeps the night quiet."""
import asyncio
from datetime import datetime

import pytest

import proactive
import reminders


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(reminders, "DB", str(tmp_path / "r.db"))


def test_once_per_thing():
    assert proactive._first_time("cal:Dentist:2026-11-07T10:00")
    assert not proactive._first_time("cal:Dentist:2026-11-07T10:00")


def _day(i, hrv, rhr):
    return {"date": f"2026-10-{i:02d}", "workouts": [], "hrv": hrv, "resting_heart_rate": rhr}


def test_early_warning_needs_both_signals_on_most_days():
    base = [_day(i, 65, 55) for i in range(1, 25)]
    assert proactive.stressed(base + [_day(25, 50, 61), _day(26, 52, 60), _day(27, 66, 55)])   # 2 of 3 bad
    assert not proactive.stressed(base + [_day(25, 50, 55), _day(26, 52, 56), _day(27, 66, 55)])  # HRV only
    assert not proactive.stressed(base[:5])                                                      # too little data


def test_quiet_hours_hold_non_urgent_pushes(monkeypatch):
    sent = []
    monkeypatch.setattr("approvals.push", lambda title, text, **k: sent.append(title) or True)

    class Night(datetime):
        @classmethod
        def now(cls, tz=None): return datetime(2026, 11, 7, 23, 30, tzinfo=tz)
    monkeypatch.setattr(proactive, "datetime", Night)
    asyncio.run(proactive.deliver("Evening wrap-up", "x"))
    asyncio.run(proactive.deliver("Smoke alarm", "y", urgent=True))
    assert sent == ["Smoke alarm"]  # the wrap-up waits for 8 am

    class Morning(datetime):
        @classmethod
        def now(cls, tz=None): return datetime(2026, 11, 8, 8, 5, tzinfo=tz)
    monkeypatch.setattr(proactive, "datetime", Morning)
    asyncio.run(proactive._release_held())
    assert sent == ["Smoke alarm", "Evening wrap-up"]
