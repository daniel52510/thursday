"""His lifting journal (lifts.py): names land on his program's lifts, sets are kept by day, and the trainer's next
weight follows the rules (up when every set hit the top, hold, back off after two misses, dumbbells cap out)."""
import asyncio
from datetime import datetime

import pytest

import asks
import lifts
import trainer
from calendar_tool import TURN


@pytest.fixture(autouse=True)
def journal(monkeypatch, tmp_path):
    monkeypatch.setattr(lifts, "DB", str(tmp_path / "lifts.db"))
    monkeypatch.setattr(trainer, "profile", lambda: {"weight_lb": 300, "home_equipment": "apartment gym"})
    TURN.set({"user": "owner", "conversation": "c", "turn": 1, "user_said": "x"})


def test_his_words_land_on_his_gym_lifts():
    assert lifts.canonical("bench") == "Dumbbell bench press (3 s down)"
    assert lifts.canonical("goblet squats") == "Goblet squat (3 s down)"
    assert lifts.canonical("lat pulldown") == "Cable lat pulldown"
    assert lifts.canonical("zercher carries") == "Zercher carries"   # not in the program: kept as he said it


def test_parse_prescriptions():
    assert lifts.parse("4 x 6-8, heavy") == (4, 6, 8) and lifts.parse("3 x 12 each leg") == (3, 12, 12)
    assert lifts.parse("45 s") is None and lifts.parse("10 min easy") is None


def _log(ex, day, weight, reps):
    return lifts.add(ex, [{"reps": r, "weight_lb": weight} for r in reps], day)


def test_progression_up_hold_and_back_off():
    ex = "Cable lat pulldown"                                              # 4 x 8-12, upper body: +5
    assert lifts.next_weight(ex, 4, 8, 12)["weight_lb"] is None            # no history yet
    _log(ex, "2026-11-02", 120, [12, 12, 12, 12])
    assert lifts.next_weight(ex, 4, 8, 12)["weight_lb"] == 125             # every set at the top: up
    _log(ex, "2026-11-05", 125, [10, 9, 8, 8])
    assert lifts.next_weight(ex, 4, 8, 12)["weight_lb"] == 125             # in range: same weight, more reps
    _log(ex, "2026-11-08", 125, [8, 7, 6, 6])
    assert lifts.next_weight(ex, 4, 8, 12)["weight_lb"] == 125             # one miss: hold
    _log(ex, "2026-11-11", 125, [7, 6, 6, 5])
    nxt = lifts.next_weight(ex, 4, 8, 12)
    assert nxt["weight_lb"] == 110 and "back off" in nxt["why"]            # two misses running: -10 %, to 5 lb


def test_lower_body_jumps_more_and_dumbbells_cap_at_his_rack():
    _log("Seated cable row", "2026-11-02", 150, [10, 10, 10, 10])
    assert lifts.next_weight("Seated cable row", 4, 10, 10)["weight_lb"] == 155
    _log("Goblet squat (3 s down)", "2026-11-02", 50, [12, 12, 12, 12])     # his heaviest dumbbell
    capped = lifts.next_weight("Goblet squat (3 s down)", 4, 10, 12)
    assert capped["weight_lb"] == 50 and "slower" in capped["why"]


def test_personal_best_and_bodyweight():
    _log("Dumbbell bench press (3 s down)", "2026-11-02", 45, [10, 10])
    assert lifts.add("bench", [{"reps": 8, "weight_lb": 50}])["personal_best"]
    assert not lifts.add("bench", [{"reps": 5, "weight_lb": 45}])["personal_best"]
    _log("Push-ups", "2026-11-02", 0, [20, 18])
    assert lifts.next_weight("Push-ups", 3, 15, 20)["weight_lb"] == 0


def test_plan_and_log_as_planned(monkeypatch):
    monday = datetime(2026, 11, 9, 18, tzinfo=lifts._today().tzinfo)          # push day
    monkeypatch.setattr(lifts, "_today", lambda: monday)
    _log("Dumbbell bench press (3 s down)", "2026-11-02", 45, [12, 12, 12, 12])
    p = lifts.plan()
    bench = next(x for x in p["lifts"] if x["exercise"].startswith("Dumbbell bench"))
    assert bench["weight_lb"] == 50 and bench["sets"] == 4 and "Last:" not in bench["why"]
    assert all(not x["exercise"].lower().startswith("core") for x in p["lifts"])
    saved = lifts.log_plan()
    assert [s["exercise"] for s in saved] == ["Dumbbell bench press (3 s down)"]  # only lifts with a weight
    assert saved[0]["sets"] == [{"reps": 12, "weight_lb": 50.0}] * 4 and lifts.logged_today()


def test_voice_tool_logs_uneven_sets_and_undo():
    r = lifts.lift_log({"exercise": "lat pulldown", "sets_detail": [{"reps": 10, "weight_lb": 120}, {"reps": 8, "weight_lb": 130}]})
    assert r.ok and r.data["exercise"] == "Cable lat pulldown" and len(r.data["sets"]) == 2
    assert lifts.lift_undo({"exercise": "lat pulldown"}).data["removed_sets"] == 2
    TURN.set({"user": "guest", "conversation": "c", "turn": 2, "user_said": "x"})
    assert lifts.lift_log({"exercise": "bench", "reps": 5}).error == "training_is_daniels_only"


def test_he_is_asked_to_log_in_the_evening(monkeypatch, tmp_path):
    monkeypatch.setattr(asks, "DB", str(tmp_path / "asks.db"))
    evening = datetime(2026, 11, 9, 20, 15, tzinfo=lifts._today().tzinfo)
    monkeypatch.setattr(lifts, "_today", lambda: evening)
    import health
    monkeypatch.setattr(health, "DB", str(tmp_path / "health.db"))
    asked = []

    async def fake_ask(kind, q, options, data, **kw):
        asked.append((kind, q, [o["key"] for o in options]))
    monkeypatch.setattr(asks, "ask", fake_ask)
    asyncio.run(asks.lift_source())
    assert asked and asked[0][0] == "lifts" and asked[0][2] == ["tell", "no"]  # no history: he tells it
    _log("Dumbbell bench press (3 s down)", "2026-11-02", 45, [12, 12, 12, 12])
    asked.clear()
    asyncio.run(asks.lift_source())
    assert asked[0][2] == ["log", "no"] and "bench press 4x12 at 50" in asked[0][1]
    assert asks.HANDLERS["lifts"]("log", {"day": "2026-11-09"}) == "logged 1 lifts"
    asked.clear()
    asyncio.run(asks.lift_source())                                           # logged: nothing more to ask
    assert not asked


def test_library_has_the_presets_and_picks_stay_as_named():
    names = [n for group in lifts.LIBRARY.values() for n in group]
    assert 100 <= len(names) <= 150 and len(names) == len(set(names))
    assert lifts.canonical("Hip thrusts") == "Hip thrusts" and lifts.canonical("Arnold press") == "Arnold press"


def test_custom_exercise_is_kept_for_next_time_and_a_workout_saves_in_one_go():
    saved = lifts.save_workout("2026-11-09", [
        {"exercise": "Landmine squat to press", "custom": True, "category": "Full body",
         "sets": [{"reps": 10, "weight_lb": 45}, {"reps": 8, "weight_lb": 55}]},
        {"exercise": "Hip thrusts", "sets": [{"reps": 12, "weight_lb": 135}] * 3},
        {"exercise": "Nothing", "sets": [{"reps": 0, "weight_lb": 10}]}])           # no reps: skipped
    assert [s["exercise"] for s in saved] == ["Landmine squat to press", "Hip thrusts"]
    assert lifts.library()["My exercises"] == ["Landmine squat to press"]
    assert lifts.canonical("landmine squat to press") == "Landmine squat to press"
    last = lifts.last_sets("Landmine squat to press")
    assert last["last"] == [{"reps": 10, "weight_lb": 45.0}, {"reps": 8, "weight_lb": 55.0}] and last["last_day"] == "2026-11-09"


def test_history_spans_a_lifts_name_variants():
    _log("Dumbbell bench press", "2026-11-02", 45, [12, 12, 12, 12])               # picked from the library
    assert lifts.next_weight("Dumbbell bench press (3 s down)", 4, 8, 12)["weight_lb"] == 50  # the program's name
