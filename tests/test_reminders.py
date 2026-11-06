"""Reminders that find him, and promises THURSDAY keeps (reminders.py)."""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest

import reminders
from calendar_tool import TURN


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(reminders, "DB", str(tmp_path / "r.db"))
    TURN.set({"user": "owner", "conversation": "c1", "turn": 1, "user_said": "remind me"})


NOW = datetime(2026, 11, 6, 21, 0, tzinfo=timezone.utc)  # 4 pm in New York


def test_parse_when(monkeypatch):
    monkeypatch.setenv("THURSDAY_TZ", "America/New_York")
    assert reminders.parse_when(None, 20, NOW) == NOW + timedelta(minutes=20)
    assert reminders.parse_when("17:00", now=NOW) == datetime(2026, 11, 6, 22, 0, tzinfo=timezone.utc)   # today 5 pm
    assert reminders.parse_when("9:00", now=NOW) == datetime(2026, 11, 7, 14, 0, tzinfo=timezone.utc)    # past: tomorrow
    assert reminders.parse_when("5:30 pm", now=NOW) == datetime(2026, 11, 6, 22, 30, tzinfo=timezone.utc)
    assert reminders.parse_when("2026-11-10 08:15", now=NOW) == datetime(2026, 11, 10, 13, 15, tzinfo=timezone.utc)
    assert reminders.parse_when("someday", now=NOW) is None


def test_set_list_cancel_and_owner_only():
    r = reminders.reminder_set({"text": "Call the bank", "in_minutes": 30})
    assert r.ok and "at" in r.data["when"]
    assert reminders.reminder_set({"text": "x", "at": "nonsense"}).error.startswith("bad_time")
    listed = reminders.reminder_list({}).data["reminders"]
    assert [x["text"] for x in listed] == ["Call the bank"]
    assert reminders.reminder_cancel({"id_or_text": "bank"}).ok
    assert reminders.reminder_list({}).data["reminders"] == []
    TURN.set({"user": "guest", "conversation": "c2", "turn": 1, "user_said": "remind me"})
    assert reminders.reminder_set({"text": "x", "in_minutes": 5}).error == "reminders_are_daniels_only"


def test_promises_are_noted_kept_and_closed():
    reminders.capture("owner", "Great idea. I'll look into the SpaceX tech stacks and get back to you.", set(), "c1")
    reminders.capture("owner", "Yes, I research career paths for you tonight.", set(), "c2")  # Rocky grammar
    reminders.capture("owner", "The weather is nice.", set(), "c3")
    reminders.capture("guest", "I'll look into it.", set(), "c4")                          # owner's only
    note = reminders.prompt_note("owner")
    assert "SpaceX" in note and "career paths" in note and "weather" not in note
    reminders.capture("owner", "Mission started.", {"mission_start"}, "c1")  # he did it: c1's promise is kept
    assert "SpaceX" not in reminders.prompt_note("owner")
    pid = next(line for line in note.splitlines() if "career paths" in line).split("[")[1].split("]")[0]
    assert reminders.promise_done({"id": pid, "dropped": True}).ok
    assert reminders.prompt_note("owner") == ""


def test_due_reminder_is_delivered_once(monkeypatch):
    sent = []

    async def fake_deliver(r):
        sent.append(r["text"])
    monkeypatch.setattr(reminders, "_deliver", fake_deliver)
    reminders.reminder_set({"text": "Oven", "in_minutes": 0.01})
    reminders.reminder_set({"text": "Later", "in_minutes": 60})

    async def one_tick():
        real_sleep = asyncio.sleep
        async def quick(s): await real_sleep(0.01)
        monkeypatch.setattr(reminders.asyncio, "sleep", quick)
        task = asyncio.create_task(reminders.run())
        await real_sleep(1.0)
        task.cancel()
    asyncio.run(one_tick())
    assert sent == ["Oven"]
