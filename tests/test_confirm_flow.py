"""Read back first, start on his plain yes (missions.py; used by missions, routines, improvements)."""
import pytest

import missions


class FakeDB:
    def __init__(self):
        self.created = []

    def list_missions(self, *a, **k):
        return []

    def create_mission(self, user, goal, tz, **k):
        self.created.append(goal)
        return {"id": f"m{len(self.created)}"}


@pytest.fixture
def db(monkeypatch):
    fake = FakeDB()
    monkeypatch.setattr(missions, "initalize_db", lambda: fake)
    missions._pending.clear()
    missions._started.clear()
    return fake


def test_first_call_only_reads_back(db, owner_turn):
    owner_turn("look into robot vacuums under 400")
    r = missions.mission_start({"goal": "Best robot vacuum under $400 for a dog"})
    assert r.ok and r.data["status"].startswith("NOT STARTED") and db.created == []


def test_plain_yes_starts_it_and_a_hedge_doesnt(db, owner_turn):
    owner_turn("look into robot vacuums")
    missions.mission_start({"goal": "Best robot vacuum under $400 for a dog"})
    assert missions.confirms_pending("c1", "yes but under 300") is None
    for yes in ("yes", "Uh, yes please", "Hey Thursday, do it", "Start it", "Start the mission", "Run it"):
        assert missions.confirms_pending("c1", yes) == "mission_start"
    owner_turn("Uh, yes please")
    r = missions.mission_start({"confirmed": True})
    assert r.data["status"] == "STARTED" and db.created == ["Best robot vacuum under $400 for a dog"]


def test_the_models_repeat_call_doesnt_start_a_second(db, owner_turn):
    owner_turn("research vacuums")
    missions.mission_start({"goal": "Best robot vacuum under $400 for a dog"})
    owner_turn("yes")
    missions.mission_start({"confirmed": True})
    again = missions.mission_start({"goal": "anything else here", "confirmed": True})
    assert again.data["status"] == "STARTED" and len(db.created) == 1


def test_confirmed_without_a_read_back_is_refused(db, owner_turn):
    owner_turn("do it")
    r = missions.mission_start({"goal": "something to research here", "confirmed": True})
    assert not r.ok and r.error.startswith("not_confirmed") and db.created == []


def test_only_the_owner(db):
    from calendar_tool import TURN
    TURN.set({"user": "guest", "conversation": "g", "user_said": "yes"})
    assert missions.mission_start({"goal": "research this for me"}).error == "missions_are_daniels_only"
