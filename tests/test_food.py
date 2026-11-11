"""His food journal (food.py): targets from the trainer, estimates parsed and bounded, totals and what's left,
recent foods re-added without an estimate, his own calorie target, owner only."""
import json

import pytest

import food
import trainer
from calendar_tool import TURN


@pytest.fixture(autouse=True)
def journal(monkeypatch, tmp_path):
    monkeypatch.setattr(food, "DB", str(tmp_path / "food.db"))
    monkeypatch.setattr(food, "PREFS", str(tmp_path / "food.json"))
    monkeypatch.setattr(trainer, "profile", lambda: {"weight_lb": 300.0, "height": "6 ft 5 in", "age": "27"})
    monkeypatch.setattr(food, "_day_kind", lambda day: "cardio")
    monkeypatch.setattr(food, "_active_kcal", lambda: None)
    # The planned day stands whatever the clock says: after 7 pm with nothing logged, coach.effective_kind turns it
    # into a rest day (-150 kcal), and these tests failed every evening (found 2026-11-10, 7:46 pm).
    import coach
    monkeypatch.setattr(coach, "SKIP_AFTER_HOUR", 24)
    TURN.set({"user": "owner", "conversation": "c", "turn": 1, "user_said": "x"})


def test_targets_are_worked_out_from_his_body_and_distance_to_goal(monkeypatch):
    t = food.targets()                     # BMR 2,453 x 1.55 = 3,803; 55 lb to go: -750 → 3,050
    assert (t["kcal"], t["protein"], t["maintenance"]) == (3050, 245, 3802)
    assert t["fat"] == round(max(3050 * 0.25 / 9, 0.3 * 245)) and t["carbs"] == round((3050 - 245 * 4 - t["fat"] * 9) / 4)
    monkeypatch.setattr(food, "_day_kind", lambda day: "lift")
    lift = food.targets()
    assert lift["kcal"] == 3200 and lift["carbs"] > t["carbs"] and lift["protein"] == 245    # lifting day: +150 carbs
    monkeypatch.setattr(food, "_day_kind", lambda day: "cardio")
    food.log_weight(262)                   # 17 lb to go: ~1 lb a week; and a lighter body burns less
    t2 = food.targets()
    assert t2["weight_lb"] == 262 and t2["maintenance"] < 3802 and t2["kcal"] == round((t2["maintenance"] - 500) / 25) * 25
    food.log_weight(246)                   # at goal: maintenance, no deficit
    assert food.targets()["kcal"] == round(food.targets()["maintenance"] / 25) * 25
    assert food.set_target(2800)["kcal"] == 2800 and food.set_target(None)["kcal"] != 2800


def test_watch_activity_and_floors(monkeypatch):
    monkeypatch.setattr(food, "_active_kcal", lambda: 900)            # discounted active calories from the Watch
    t = food.targets()
    assert t["maintenance"] == round(2453.2 * 1.2 + 900) and "Watch" in t["why"]
    monkeypatch.setattr(trainer, "profile", lambda: {"weight_lb": 150.0, "height": "5 ft 2 in", "age": "60"})
    monkeypatch.setattr(trainer, "targets", lambda p: {"goal_weight_lb": 120})
    assert food.targets()["kcal"] >= food.MIN_KCAL                   # never under the floor


def test_adaptive_maintenance_learns_from_his_logs(monkeypatch):
    from datetime import date, timedelta
    today = food._now().date()
    for i in range(20):                    # 20 logged days at 3,000 while losing 0.2 lb a day (700 kcal/day)
        d = (today - timedelta(days=20 - i)).isoformat()
        food.add([{"name": "Day", "qty": "", "kcal": 3000, "protein": 240, "carbs": 300, "fat": 100}], "lunch", day=d)
        food.log_weight(300 - 0.2 * i, d)
    burn, n = food._adaptive_tdee()
    assert n == 20 and 3600 <= burn <= 3800                         # 3,000 eaten + ~700 a day from the scale
    assert "your logs and weigh-ins say" in food.targets()["why"]


def test_estimate_is_parsed_and_bounded(monkeypatch):
    import httpx
    reply = {"items": [{"name": "Scrambled eggs", "qty": "2 large", "kcal": 180, "protein": 12, "carbs": 2, "fat": 14},
                       {"name": "", "kcal": 50}, {"name": "Toast", "qty": "1 slice", "kcal": "80", "protein": 3, "carbs": 15, "fat": 1},
                       {"name": "Bogus", "kcal": 99999}]}
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: type("R", (), {"json": lambda self: {"message": {"content": json.dumps(reply)}}})())
    items = food.estimate("two eggs and toast")
    assert [i["name"] for i in items] == ["Scrambled eggs", "Toast", "Bogus"] and items[2]["kcal"] == 5000


def test_totals_left_recent_and_again():
    food.add([{"name": "Eggs", "qty": "2", "kcal": 180, "protein": 12, "carbs": 2, "fat": 14}], "breakfast")
    food.add([{"name": "Protein shake", "qty": "1 scoop", "kcal": 120, "protein": 25, "carbs": 3, "fat": 1}], "snack")
    d = food.day()
    assert d["totals"] == {"kcal": 300, "protein": 37, "carbs": 5, "fat": 15} and d["left"]["kcal"] == 2750
    shake = next(e for e in food.recent() if e["name"] == "Protein shake")
    food.readd(shake["id"])
    assert food.day()["totals"]["protein"] == 62 and len(food.recent()) == 2   # same food once in recents
    assert food.remove(shake["id"]) and food.day()["totals"]["protein"] == 37


def test_voice_tools(monkeypatch):
    monkeypatch.setattr(food, "estimate", lambda text: [{"name": "Chicken burrito", "qty": "1", "kcal": 1000, "protein": 50, "carbs": 110, "fat": 35}])
    r = food.food_log({"description": "a chicken burrito", "meal": "lunch"})
    assert r.ok and r.data["calories_left"] == 2050 and r.data["protein_left_g"] == 195
    assert food.food_today({}).data["eaten_kcal"] == 1000
    assert food.food_undo({"name": "burrito"}).data["removed"] == "Chicken burrito"
    assert not food.food_undo({}).ok
    TURN.set({"user": "guest", "conversation": "c", "turn": 2, "user_said": "x"})
    assert food.food_log({"description": "x"}).error == "food_journal_is_daniels_only"


def test_weight_by_voice_and_the_weekly_summary(monkeypatch):
    import asyncio
    import asks
    import proactive
    from datetime import datetime
    r = food.weight_log({"pounds": 296.4})
    assert r.ok and r.data["logged_lb"] == 296.4 and r.data["seven_day_average_lb"] == 296.4
    assert not food.weight_log({"pounds": 5}).ok
    sunday = datetime(2026, 11, 8, 11, 0, tzinfo=food._now().tzinfo)
    monkeypatch.setattr(food, "_now", lambda: sunday)
    monkeypatch.setattr(asks, "DB", str(__import__("pathlib").Path(food.DB).with_name("asks.db")))
    told = []

    async def deliver(title, text, **kw):
        told.append(text)
    monkeypatch.setattr(proactive, "deliver", deliver)
    asyncio.run(asks.nutrition_source())
    asyncio.run(asks.nutrition_source())                              # once a week
    assert len(told) == 1 and "Today's target is" in told[0] and "maintenance" in told[0]


def test_day_kinds_follow_the_week():
    import importlib
    from datetime import datetime, timedelta
    fresh = importlib.reload(importlib.import_module("food"))       # the real _day_kind, not the fixture's
    monday = datetime(2026, 11, 9, 12, tzinfo=fresh._now().tzinfo)
    kinds = [fresh._day_kind(monday + timedelta(days=i)) for i in range(7)]
    assert kinds == ["lift", "lift", "cardio", "lift", "lift", "cardio", "rest"]
