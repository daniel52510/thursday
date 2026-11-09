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
    TURN.set({"user": "owner", "conversation": "c", "turn": 1, "user_said": "x"})


def test_targets_are_the_trainers_cut():
    t = food.targets()
    assert t["kcal"] == 3200 and t["protein"] == 245 and t["fat"] == round(3200 * 0.28 / 9)
    assert t["carbs"] == round((3200 - 245 * 4 - t["fat"] * 9) / 4)
    assert food.set_target(2800)["kcal"] == 2800 and food.set_target(None)["kcal"] == 3200


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
    assert d["totals"] == {"kcal": 300, "protein": 37, "carbs": 5, "fat": 15} and d["left"]["kcal"] == 2900
    shake = next(e for e in food.recent() if e["name"] == "Protein shake")
    food.readd(shake["id"])
    assert food.day()["totals"]["protein"] == 62 and len(food.recent()) == 2   # same food once in recents
    assert food.remove(shake["id"]) and food.day()["totals"]["protein"] == 37


def test_voice_tools(monkeypatch):
    monkeypatch.setattr(food, "estimate", lambda text: [{"name": "Chicken burrito", "qty": "1", "kcal": 1000, "protein": 50, "carbs": 110, "fat": 35}])
    r = food.food_log({"description": "a chicken burrito", "meal": "lunch"})
    assert r.ok and r.data["calories_left"] == 2200 and r.data["protein_left_g"] == 195
    assert food.food_today({}).data["eaten_kcal"] == 1000
    assert food.food_undo({"name": "burrito"}).data["removed"] == "Chicken burrito"
    assert not food.food_undo({}).ok
    TURN.set({"user": "guest", "conversation": "c", "turn": 2, "user_said": "x"})
    assert food.food_log({"description": "x"}).error == "food_journal_is_daniels_only"


def test_the_trainer_reads_the_journal_and_weight(monkeypatch):
    from datetime import date, timedelta
    t = food.targets()["kcal"]
    start = date(2026, 10, 26)
    for i in range(12):                                   # 12 full days, on target, good protein
        food.add([{"name": "Day", "qty": "", "kcal": t, "protein": 240, "carbs": 300, "fat": 100}], "lunch",
                 day=(start + timedelta(days=i)).isoformat())
    monkeypatch.setattr(food, "history", lambda days=14: [{"day": "x", "kcal": t, "protein": 240}] * 12)
    monkeypatch.setattr(food, "_weights", lambda days=28: [("2026-10-26", 300.0), ("2026-11-07", 299.8)])
    c = food.coach()
    assert c["lb_per_week"] == -0.1 and c["suggested_kcal"] == t - 200 and "flat" in c["advice"][-1]
    monkeypatch.setattr(food, "_weights", lambda days=28: [("2026-10-26", 300.0), ("2026-11-07", 295.0)])
    assert food.coach()["suggested_kcal"] == t + 200                        # 2.9 lb a week: too fast
    monkeypatch.setattr(food, "_weights", lambda days=28: [("2026-10-26", 300.0), ("2026-11-07", 298.0)])
    assert food.coach()["suggested_kcal"] is None                           # ~1.2 lb a week: right on plan
    monkeypatch.setattr(food, "history", lambda days=14: [{"day": "x", "kcal": t, "protein": 150}] * 12)
    assert any("Protein averages 150" in a for a in food.coach()["advice"])


def test_weekly_check_in_asks_and_a_yes_sets_the_target(monkeypatch):
    import asyncio
    import asks
    from datetime import datetime
    sunday = datetime(2026, 11, 8, 11, 0, tzinfo=food._now().tzinfo)
    monkeypatch.setattr(food, "_now", lambda: sunday)
    monkeypatch.setattr(food, "coach", lambda days=14: {"suggested_kcal": 3000, "advice": ["Flat for two weeks: time to drop to 3000 kcal."]})
    asked = []

    async def fake_ask(kind, q, options, data, **kw):
        asked.append((kind, q, data))
    monkeypatch.setattr(asks, "ask", fake_ask)
    asyncio.run(asks.nutrition_source())
    assert asked and asked[0][0] == "nutrition" and "3,000" in asked[0][1]
    assert asks.HANDLERS["nutrition"]("set", {"kcal": 3000}) == "daily target now 3,000 kcal"
    assert food.targets()["kcal"] == 3000
