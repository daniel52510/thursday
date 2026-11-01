"""Approvals (approvals.py): reminders a mission proposes, and what happens on his OK."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

import approvals


@pytest.fixture
def pushed(monkeypatch):
    sent = []
    monkeypatch.setattr(approvals, "push", lambda *a, **k: sent.append((a, k)) or True)
    return sent


@pytest.fixture
def mission_turn():
    from calendar_tool import TURN
    TURN.set({"user": "owner", "turn": "mission-m1", "timezone": "America/New_York"})


def _local(days: float) -> str:
    return (datetime.now(ZoneInfo("America/New_York")) + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M")


def test_reminder_proposed_and_set_on_approve(pushed, mission_turn):
    r = approvals.propose_reminder({"text": "Buy concert tickets", "at": _local(1), "why": "Sale opens"})
    assert r.ok and "Buy concert tickets" in r.data["waiting_for_his_ok"]
    done = approvals.decide(r.data["approval_id"], "owner", True)
    assert done["status"] == "approved" and done["result"] == "scheduled"
    assert pushed[-1][1].get("at") is not None  # held by ntfy until its time


def test_reminder_too_far_or_past_refused(pushed, mission_turn):
    assert approvals.propose_reminder({"text": "x", "at": _local(6)}).error.startswith("too_far")
    assert approvals.propose_reminder({"text": "x", "at": _local(-1)}).error.startswith("bad_input")


def test_only_missions_propose(pushed):
    from calendar_tool import TURN
    TURN.set({"user": "owner", "turn": "t1"})
    assert approvals.propose_reminder({"text": "x", "at": _local(1)}).error == "only_missions_propose"


def test_decline_sets_nothing(pushed, mission_turn):
    r = approvals.propose_reminder({"text": "Call mom", "at": _local(1)})
    count = len(pushed)
    assert approvals.decide(r.data["approval_id"], "owner", False)["status"] == "declined"
    assert len(pushed) == count  # no reminder pushed
