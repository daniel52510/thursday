"""The coach's brain (coach.py) and the rules built on it: calories follow the session he did, short sleep eases
the deficit and (two nights running) the weights, a missed lifting day moves to today, an under-fed day stops a push,
the meal plan resizes to today's numbers, and the morning briefing says why."""
from datetime import datetime, timedelta

import pytest

import coach
import food
import lifts
import mealplan
import trainer
from calendar_tool import TURN

TZ = food._now().tzinfo


@pytest.fixture(autouse=True)
def setup(monkeypatch, tmp_path):
    for mod, attr, name in ((food, "DB", "food.db"), (food, "PREFS", "food.json"), (lifts, "DB", "lifts.db"),
                            (mealplan, "FILE", "plans.json")):
        monkeypatch.setattr(mod, attr, str(tmp_path / name))
    monkeypatch.setattr(trainer, "profile", lambda: {"weight_lb": 300.0, "height": "6 ft 5 in", "age": "27",
                                                     "home_equipment": "apartment gym"})
    monkeypatch.setattr(food, "_active_kcal", lambda: None)
    monkeypatch.setattr(coach, "_health_days", lambda n: {})
    TURN.set({"user": "owner", "conversation": "c", "turn": 1, "user_said": "x"})


def _at(monkeypatch, when):
    monkeypatch.setattr(food, "_now", lambda: when)
    monkeypatch.setattr(lifts, "_today", lambda: when)
    monkeypatch.setattr(trainer, "_today", lambda: when)


def test_calories_follow_the_session_he_did(monkeypatch):
    mon = datetime(2026, 11, 9, 12, tzinfo=TZ)                         # a lifting day
    _at(monkeypatch, mon)
    planned = food.plan_today()["kcal"]
    _at(monkeypatch, mon.replace(hour=20))                             # 8 pm, nothing recorded: skipped
    skipped = food.plan_today()
    assert skipped["kcal"] == planned - 300 and "no lift session recorded" in skipped["why"]
    lifts.add("Cable lat pulldown", [{"reps": 10, "weight_lb": 120}], day="2026-11-09")
    assert food.plan_today()["kcal"] == planned                        # he lifted after all
    assert coach.effective_kind(mon + timedelta(days=1)) == ("lift", "planned")


def test_short_sleep_eases_the_deficit_and_two_nights_drop_the_weights(monkeypatch):
    mon = datetime(2026, 11, 9, 9, tzinfo=TZ)
    _at(monkeypatch, mon)
    rested = food.plan_today()["kcal"]
    nights = {"2026-11-09": {"sleep_minutes": 330}, "2026-11-08": {"sleep_minutes": 343}}
    monkeypatch.setattr(coach, "_health_days", lambda n: nights)
    short = food.plan_today()
    assert short["kcal"] == rested + 200 and "short night (5 h 30)" in short["why"]
    lifts.add("Dumbbell bench press (3 s down)", [{"reps": 12, "weight_lb": 45}] * 4, day="2026-11-02")
    import health
    monkeypatch.setattr(health, "readiness", lambda *a: None)
    monkeypatch.setattr(health, "daily", lambda days, tz=None: [])
    out = trainer.training_today({}).data
    bench = next(x for x in out["loads"] if x["exercise"].startswith("Dumbbell bench"))
    assert "50 lb" not in bench["do"] and "45 lb" in bench["do"] and "Two short nights" in out["why"]


def test_a_missed_lifting_day_moves_to_today(monkeypatch):
    import health
    monkeypatch.setattr(health, "readiness", lambda *a: None)
    monkeypatch.setattr(health, "daily", lambda days, tz=None: [])
    wed = datetime(2026, 11, 11, 9, tzinfo=TZ)                         # intervals day; Tuesday (pull) skipped
    _at(monkeypatch, wed)
    lifts.add("Cable lat pulldown", [{"reps": 10, "weight_lb": 120}], day="2026-11-09")   # Monday was done
    out = trainer.training_today({}).data
    assert out["moved_from"] == "Tuesday" and out["session"].startswith("Pull") and "moves to today" in out["why"]
    assert any(x["exercise"] == "Cable lat pulldown" for x in out["loads"])


def test_an_underfed_day_stops_a_push(monkeypatch):
    import health
    monkeypatch.setattr(health, "readiness", lambda *a: {"score": 90, "label": "primed", "drivers": [], "date": "x"})
    monkeypatch.setattr(health, "daily", lambda days, tz=None: [])
    tue = datetime(2026, 11, 10, 9, tzinfo=TZ)
    _at(monkeypatch, tue)
    lifts.add("Cable lat pulldown", [{"reps": 10, "weight_lb": 120}], day="2026-11-09")
    assert trainer.training_today({}).data["intensity"] == "push"
    food.add([{"name": "Pizza", "qty": "", "kcal": 1500, "protein": 60, "carbs": 150, "fat": 70}], "dinner", day="2026-11-09")
    out = trainer.training_today({}).data
    assert out["intensity"] == "as planned" and "no max efforts" in out["why"]


def test_the_plan_resizes_to_todays_numbers(monkeypatch):
    mon = datetime(2026, 11, 9, 20, tzinfo=TZ)
    _at(monkeypatch, mon)
    per = {"kcal": 300, "protein": 25, "carbs": 30, "fat": 8}
    meals = [mealplan._with_macros({"slot": s, "name": s, "how": "", "ingredients": [{"item": "x", "grams": 200, "per100": dict(per)}]})
             for s, _ in mealplan.SLOTS]                                  # 600 each, 3,000 a day
    day = {"date": "2026-11-09", "weekday": "Monday", "kind": "lift", "theme": "plant-based", "meals": meals,
           "targets": {"kcal": 3400, "protein": 245, "carbs": 330, "fat": 85}, "totals": mealplan._totals(meals), "eaten": [0, 1]}
    mealplan._save({mealplan.key(mealplan.week_start("this")): {"week": "x", "days": [day], "groceries": []}})
    why = mealplan.refresh_today()                                        # 8 pm, no lift: the day drops 300
    d = mealplan.today()
    want = food.plan_today()["kcal"]
    assert why and abs(d["totals"]["kcal"] - want) <= 30 and d["meals"][0]["kcal"] == 600   # eaten meals untouched
    assert mealplan.refresh_today() is None                               # once it fits, nothing more


def test_sleep_pattern_and_the_briefing(monkeypatch):
    mon = datetime(2026, 11, 9, 9, tzinfo=TZ)
    _at(monkeypatch, mon)
    monkeypatch.setattr(coach, "_health_days", lambda n: {"2026-11-09": {"sleep_minutes": 423}, "2026-11-08": {"sleep_minutes": 343}})
    import health
    monkeypatch.setattr(health, "readiness", lambda *a: {"score": 85, "label": "primed", "drivers": [], "date": "x"})
    monkeypatch.setattr(health, "daily", lambda days, tz=None: [])
    b = coach.briefing()
    assert b["text"].startswith("7 h 03 of sleep (you've been averaging 6 h 23)") and "readiness 85" in b["text"]
    assert "Today: push day, go heavy, starting with Dumbbell bench press 4 x 8-12" in b["text"] and "kcal" in b["text"]
