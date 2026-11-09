"""Waterson mode (food.py) and the weekly meal plan (mealplan.py): feedings and timing, the alcohol rule, the
morning check-in moving the day's intensity, meal plans to each day's numbers, and "Ate it" logging exact macros."""
import asyncio
from datetime import datetime, timedelta

import pytest

import asks
import food
import mealplan
import trainer
from calendar_tool import TURN


@pytest.fixture(autouse=True)
def setup(monkeypatch, tmp_path):
    for mod, attr, name in ((food, "DB", "food.db"), (food, "PREFS", "food.json"), (mealplan, "FILE", "plans.json"),
                            (asks, "DB", "asks.db")):
        monkeypatch.setattr(mod, attr, str(tmp_path / name))
    monkeypatch.setattr(trainer, "profile", lambda: {"weight_lb": 300.0, "height": "6 ft 5 in", "age": "27"})
    monkeypatch.setattr(food, "_active_kcal", lambda: None)
    TURN.set({"user": "owner", "conversation": "c", "turn": 1, "user_said": "x"})


def test_feedings_and_next_one_due(monkeypatch):
    now = datetime(2026, 11, 10, 13, 0, tzinfo=food._now().tzinfo)
    monkeypatch.setattr(food, "_now", lambda: now)
    meal = {"name": "x", "qty": "", "kcal": 500, "protein": 50, "carbs": 40, "fat": 15}
    with food._connect() as conn:
        for i, h in enumerate((7.0, 7.3, 10.0)):   # breakfast in two entries, then a snack: two feedings
            at = now.replace(hour=int(h), minute=int(h % 1 * 60)).timestamp()
            conn.execute("INSERT INTO entries VALUES (?,?,?,?,?,?,?,?,?,?,?)", (str(i), "2026-11-10", at, "snack", "x", "", 500, 50, 40, 15, "t"))
    f = food.fuel()
    assert f["feedings"] == 2 and f["next_by"] == "1:00 PM" and f["protein_per_feeding_g"] == 49 and f["theme"] == "fish"


def test_alcohol_rule_weekdays_only():
    beer = [{"name": "Guinness pint"}]
    assert food.drinks_note(beer, datetime(2026, 11, 10)) and not food.drinks_note(beer, datetime(2026, 11, 14))  # Tue / Sat
    food.save_prefs(waterson={"alcohol": False})
    assert food.drinks_note(beer, datetime(2026, 11, 10)) is None


def test_morning_check_in_moves_the_day(monkeypatch):
    import health
    monkeypatch.setattr(health, "readiness", lambda *a: {"score": 75, "label": "good", "drivers": [], "date": "x"})
    monkeypatch.setattr(health, "daily", lambda days, tz=None: [])
    assert trainer.training_today({}).data["intensity"] == "as planned"
    food.record_checkin("low")
    assert trainer.training_today({}).data["intensity"] == "lighter"
    food.record_checkin("great")
    out = trainer.training_today({}).data
    assert out["intensity"] == "push" and out["feeling_this_morning"] == "great"


def _fake_meals(t_kcal):
    """Five meals of one ingredient whose per-100 g values make each meal t_kcal / 5 at 200 g."""
    per = {"kcal": t_kcal / 5 / 2, "protein": 24.5, "carbs": 30, "fat": 7.5}
    return [mealplan._with_macros({"slot": s, "name": f"{s} dish", "how": "",
                                   "ingredients": [{"item": "Chicken breast", "grams": 200, "per100": dict(per)}]})
            for s, _ in mealplan.SLOTS]


def test_meal_numbers_come_from_the_ingredients_not_the_model():
    m = mealplan._with_macros({"ingredients": [{"item": "Sirloin", "grams": 175, "per100": {"kcal": 200, "protein": 30, "carbs": 0, "fat": 8}},
                                               {"item": "Potato", "grams": 200, "per100": {"kcal": 87, "protein": 2, "carbs": 20, "fat": 0.1}}]})
    assert (m["kcal"], m["protein"], m["carbs"], m["fat"]) == (524, 56, 40, 14)
    scaled = mealplan._scale([m], 2.0)[0]                       # capped at 1.3x
    assert scaled["ingredients"][0]["grams"] == 230 and scaled["kcal"] == round(230 * 2 + 260 * 0.87)


def test_a_day_is_planned_to_its_numbers_and_replanned_once_if_off(monkeypatch):
    calls = []
    day = datetime(2026, 11, 13, 12, tzinfo=food._now().tzinfo)          # Friday: a lifting day, red meat
    want = food.plan_today(day)["kcal"]

    def llm(prompt):
        calls.append(prompt)
        return _fake_meals(1500) if len(calls) == 1 else _fake_meals(want)
    monkeypatch.setattr(mealplan, "_ask_llm", llm)
    d = mealplan.plan_day(day, "no mushrooms", ["Tuesday's chili"])
    assert len(calls) == 2 and "actually add up to" in calls[1] and abs(d["totals"]["kcal"] - want) < want * 0.05
    assert d["theme"] == "red meat" and d["kind"] == "lift" and "no mushrooms" in calls[0] and "Tuesday's chili" in calls[0]


def test_week_groceries_and_ate_it_logs_exact_numbers(monkeypatch):
    monkeypatch.setattr(mealplan, "_ask_llm", lambda prompt: _fake_meals(3000))
    week = mealplan.generate("next")
    planned = sum(i["grams"] for d in week["days"] for m in d["meals"] for i in m["ingredients"])
    assert len(week["days"]) == 7 and week["groceries"] == [{"item": "Chicken breast", "grams": planned}]
    assert all(abs(d["totals"]["kcal"] - d["targets"]["kcal"]) <= d["targets"]["kcal"] * 0.05 for d in week["days"])
    d = week["days"][0]
    saved = mealplan.eat(d["date"], 2)
    assert saved["kcal"] == d["meals"][2]["kcal"] and saved["source"] == "plan" and saved["meal"] == "lunch"
    assert 2 in mealplan.get("next")["days"][0]["eaten"]


def test_sources_check_in_fuel_nudge_and_sunday_plan(monkeypatch):
    import night
    import proactive
    monkeypatch.setattr(night, "is_night", lambda now=None: False)
    tue_8am = datetime(2026, 11, 10, 8, 0, tzinfo=food._now().tzinfo)
    monkeypatch.setattr(food, "_now", lambda: tue_8am)
    asked, told, started = [], [], []

    async def fake_ask(kind, q, options, data, **kw):
        asked.append(kind)

    async def fake_deliver(title, text, **kw):
        told.append(text)
    monkeypatch.setattr(asks, "ask", fake_ask)
    monkeypatch.setattr(proactive, "deliver", fake_deliver)
    asyncio.run(asks.checkin_source())
    assert asked == ["checkin"]
    food.record_checkin("good")
    asyncio.run(asks.checkin_source())
    assert asked == ["checkin"]                                      # answered: not again
    at_9 = tue_8am.replace(hour=14)
    monkeypatch.setattr(food, "_now", lambda: at_9)
    food.add([{"name": "Eggs", "qty": "", "kcal": 300, "protein": 25, "carbs": 2, "fat": 20}], day="2026-11-10")
    with food._connect() as conn:
        conn.execute("UPDATE entries SET at = ?", ((at_9 - timedelta(hours=5)).timestamp(),))
    asyncio.run(asks.fuel_source()); asyncio.run(asks.fuel_source())
    assert len(told) == 1 and "5 hours since you ate" in told[0]   # once per gap
    sunday = datetime(2026, 11, 15, 18, 0, tzinfo=food._now().tzinfo)
    monkeypatch.setattr(food, "_now", lambda: sunday)
    monkeypatch.setattr(mealplan, "start", lambda which="this", notify=True: started.append(which) or "started")
    asyncio.run(asks.mealplan_source()); asyncio.run(asks.mealplan_source())
    assert started == ["next"]
