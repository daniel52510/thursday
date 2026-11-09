"""THURSDAY as his nutritionist: a weekly meal plan for the Casino Royale cut.

Each day is planned to that day's numbers (food.plan_today: a lifting, cardio or rest day; his current weight; the
deficit), over five feedings a few hours apart with protein in each (Waterson's "the right food at the right time"),
on that day's protein theme (food.ROTATION: plant-based, fish, white meat, vegetarian, red meat Fridays), the
"Bond breakfast" (rye, eggs, avocado, kimchi) a couple of times a week, simple apartment cooking, and his notes
(likes, dislikes, allergies, budget). Every meal has ingredients in grams and its macros; a day whose total misses
its target by more than 10 % is planned again once. A grocery list sums the week.

Made on Sunday evening for the week ahead (asks.mealplan_source), or when he asks; one LLM call per day, in the
background (about a minute), with a push when it's ready. "Ate it" logs a planned meal with its exact numbers.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict

logger = logging.getLogger("thursday.voice.mealplan")  # under voice_ws's logger: its handler reaches docker logs
FILE = os.path.join(os.getenv("THURSDAY_BRAIN_DIR", "./brain"), "mealplans.json")
SLOTS = [("Breakfast", "breakfast"), ("Mid-morning", "snack"), ("Lunch", "lunch"), ("Afternoon", "snack"),
         ("Dinner", "dinner")]
TOLERANCE = 0.10
_lock = threading.Lock()


def _load() -> dict:
    try:
        with open(FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save(data: dict) -> None:
    tmp = FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, FILE)


def week_start(which: str = "this") -> datetime:
    import food
    now = food._now()
    monday = (now - timedelta(days=now.weekday())).replace(hour=12, minute=0, second=0, microsecond=0)
    return monday + timedelta(days=7) if which == "next" else monday


def key(start: datetime) -> str:
    return start.strftime("%G-W%V")


def get(which: str = "this") -> dict | None:
    with _lock:
        return _load().get(key(week_start(which)))


def _day_prompt(t: dict, theme: str, weekday: str, notes: str, avoid: list[str], bond_breakfast: bool) -> str:
    slots = ", ".join(s for s, _ in SLOTS)
    workout = {"lift": "a lifting day: put more of the carbs in the meals around training (lunch and the afternoon "
                       "feeding)", "cardio": "a conditioning/cardio day", "rest": "a rest day: carbs lower, still "
                                                                                    "full protein"}[t["day_kind"]]
    return (
        f"You are a sports nutritionist planning one day of eating for a 6'5\", ~{t.get('weight_lb') or 300:.0f} lb man "
        f"cutting toward a lean, athletic build (Daniel Craig in Casino Royale). {weekday} is {workout}.\n"
        f"Hit these totals within 5 %: {t['kcal']} kcal, {t['protein']} g protein, {t['carbs']} g carbs, "
        f"{t['fat']} g fat.\n"
        f"Five feedings: {slots}, a few hours apart, with real protein in every one (about "
        f"{round(t['protein'] / 5)} g each).\n"
        f"Today's protein theme is {theme}: make it the protein of lunch and dinner; breakfast and the snacks vary "
        f"(eggs, Greek yogurt, cottage cheese, fish, poultry, legumes, whey). Whole foods, nothing banned, no alcohol.\n"
        + ("Breakfast is the Bond breakfast: rye bread, poached or scrambled eggs, avocado, kimchi (scale it to the "
           "numbers).\n" if bond_breakfast else "")
        + "Simple apartment cooking: 30 minutes or less, batch-friendly, ordinary US grocery store ingredients.\n"
        + (f"His notes (follow them): {notes}\n" if notes else "")
        + (f"Don't repeat these from yesterday: {', '.join(avoid)}.\n" if avoid else "")
        + "For every ingredient give its grams and its standard nutrition per 100 g (USDA values, as prepared: cooked "
          "rice, raw meat weight, etc.); the meal totals are computed from those, so get them right.\n"
        + 'Answer only JSON: {"meals": [{"slot": "Breakfast", "name": "...", "how": "one short line", "ingredients": '
          '[{"item": "Whole eggs", "grams": 150, "kcal_100g": 143, "protein_100g": 12.6, "carbs_100g": 0.7, '
          '"fat_100g": 9.5}]}]} with the five slots in order.')


def _ask_llm(prompt: str) -> list[dict]:
    import httpx
    from agent_stream import CHAT_URL, MODEL, OLLAMA_KEEP_ALIVE, OLLAMA_OPTIONS
    r = httpx.post(CHAT_URL, timeout=120, json={
        "model": MODEL, "stream": False, "think": False, "format": "json", "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {**OLLAMA_OPTIONS, "temperature": 0.6}, "messages": [{"role": "user", "content": prompt}]})
    meals = json.loads((r.json().get("message") or {}).get("content") or "{}").get("meals") or []
    out = []
    for m in meals[:len(SLOTS)]:
        try:
            ings = []
            for i in (m.get("ingredients") or [])[:12]:
                if not i.get("item"):
                    continue
                per = {k: max(0.0, min(950.0 if k == "kcal" else 100.0, float(i.get(f"{k}_100g") or 0)))
                       for k in ("kcal", "protein", "carbs", "fat")}
                ings.append({"item": str(i["item"])[:40], "grams": max(0, round(float(i.get("grams") or 0))), "per100": per})
            out.append(_with_macros({"slot": str(m.get("slot") or "")[:20], "how": str(m.get("how") or "")[:160],
                                     "name": " ".join(str(m.get("name") or "").split())[:80], "ingredients": ings}))
        except (TypeError, ValueError, AttributeError):
            continue
    return out


def _with_macros(meal: dict) -> dict:
    """A meal's numbers from its ingredients' grams and per-100 g values: computed here, not taken from the model
    (asked for meal totals, it made every meal fit the target exactly, 2026-11-09)."""
    for k in ("kcal", "protein", "carbs", "fat"):
        meal[k] = round(sum(i["grams"] / 100 * i["per100"][k] for i in meal["ingredients"]))
    return meal


def _scale(meals: list[dict], factor: float) -> list[dict]:
    """Every portion by the same factor (kept between 0.75 and 1.3), rounded to 5 g, then the numbers again."""
    factor = max(0.75, min(1.3, factor))
    for m in meals:
        for i in m["ingredients"]:
            i["grams"] = max(5, round(i["grams"] * factor / 5) * 5)
        _with_macros(m)
    return meals


def _totals(meals: list[dict]) -> dict:
    return {k: sum(m[k] for m in meals) for k in ("kcal", "protein", "carbs", "fat")}


def _off(got: dict, t: dict) -> bool:
    """Worth one more try: calories off by more than TOLERANCE, protein under 90 % or over 125 %, fat under 80 %."""
    return (abs(got["kcal"] - t["kcal"]) > t["kcal"] * TOLERANCE or not 0.9 * t["protein"] <= got["protein"] <= 1.25 * t["protein"]
            or got["fat"] < 0.8 * t["fat"])


def plan_day(day: datetime, notes: str, avoid: list[str]) -> dict:
    """One day: its targets, theme and meals; planned again once if the total misses by more than TOLERANCE."""
    import food
    t = food.plan_today(day)
    theme = food.protein_theme(day) if food.prefs()["waterson"]["rotation"] else "your choice, varied"
    prompt = _day_prompt(t, theme, day.strftime("%A"), notes, avoid, bond_breakfast=day.weekday() in (0, 3, 5))
    meals = _ask_llm(prompt)
    if not meals or _off(_totals(meals), t):
        got = _totals(meals) if meals else {"kcal": 0, "protein": 0, "carbs": 0, "fat": 0}
        meals = _ask_llm(prompt + f"\nYour last plan's ingredients actually add up to {got['kcal']} kcal, "
                                  f"{got['protein']} g protein, {got['carbs']} g carbs, {got['fat']} g fat. The day must "
                                  f"total {t['kcal']} kcal, {t['protein']} g protein, {t['carbs']} g carbs, {t['fat']} g fat:"
                                  " change the portions (and swap foods if a macro is far off).") or meals
    if meals and abs(_totals(meals)["kcal"] - t["kcal"]) > t["kcal"] * 0.05:
        meals = _scale(meals, t["kcal"] / max(1, _totals(meals)["kcal"]))  # the last few percent: portion sizes
    return {"date": day.date().isoformat(), "weekday": day.strftime("%A"), "kind": t["day_kind"], "theme": theme,
            "targets": {k: t[k] for k in ("kcal", "protein", "carbs", "fat")}, "meals": meals,
            "totals": _totals(meals), "eaten": []}


def groceries(days: list[dict]) -> list[dict]:
    """The week's ingredients, summed by item."""
    out: dict[str, float] = {}
    names: dict[str, str] = {}
    for d in days:
        for m in d["meals"]:
            for i in m["ingredients"]:
                k = re.sub(r"\s+", " ", i["item"].lower().strip())
                out[k] = out.get(k, 0) + i["grams"]
                names.setdefault(k, i["item"].strip())
    return sorted(({"item": names[k], "grams": round(v)} for k, v in out.items()), key=lambda x: x["item"].lower())


def generate(which: str = "this") -> dict:
    """The whole week, day by day (in the caller's thread). Today and later only, for this week."""
    import food
    start = week_start(which)
    today = food._now().date()
    notes = food.prefs()["notes"]
    days, avoid = [], []
    for i in range(7):
        d = start + timedelta(days=i)
        if which == "this" and d.date() < today:
            continue
        plan = plan_day(d, notes, avoid)
        days.append(plan)
        avoid = [m["name"] for m in plan["meals"]]
    week = {"week": key(start), "made": time.time(), "status": "ready", "days": days, "groceries": groceries(days)}
    with _lock:
        data = _load()
        data[key(start)] = week
        for old in sorted(data)[:-6]:  # keep about six weeks
            data.pop(old, None)
        _save(data)
    return week


_running: dict[str, threading.Thread] = {}


def start(which: str = "this", notify: bool = True) -> str:
    """Plan the week in the background; "started" or "already_running"."""
    k = key(week_start(which))
    if (th := _running.get(k)) and th.is_alive():
        return "already_running"

    def run() -> None:
        t0 = time.monotonic()
        try:
            week = generate(which)
            logger.info("meal plan %s ready in %.0f s (%d days)", k, time.monotonic() - t0, len(week["days"]))
            if notify:
                import approvals
                approvals.push("Your meal plan is ready", f"{len(week['days'])} days planned, with a grocery list "
                               f"of {len(week['groceries'])} items. It's on the Food tab.", tags=("green_salad",))
        except Exception:
            logger.exception("meal plan %s", k)
    _running[k] = th = threading.Thread(target=run, daemon=True, name=f"mealplan-{k}")
    th.start()
    return "started"


REFRESH_KCAL = 150


def refresh_today() -> str | None:
    """Today's plan follows today's numbers: when they've moved more than REFRESH_KCAL (a skipped session, a short
    night, a new weigh-in), the meals he hasn't eaten are resized to what's left. Why it changed, or None."""
    import food
    now = food._now()
    with _lock:
        data = _load()
        week = data.get(key(week_start("this")))
        d = next((x for x in (week or {}).get("days", []) if x["date"] == now.date().isoformat()), None)
        if not d:
            return None
        t = food.plan_today(now)
        if abs(t["kcal"] - d["targets"]["kcal"]) <= REFRESH_KCAL:
            return None
        eaten = set(d.get("eaten", []))
        left = [m for i, m in enumerate(d["meals"]) if i not in eaten]
        if not left:
            return None
        done = sum(m["kcal"] for i, m in enumerate(d["meals"]) if i in eaten)
        _scale(left, (t["kcal"] - done) / max(1, sum(m["kcal"] for m in left)))
        before = d["targets"]["kcal"]
        d["targets"] = {k: t[k] for k in ("kcal", "protein", "carbs", "fat")}
        d["totals"] = _totals(d["meals"])
        d["adjusted"] = f"{before:,} → {t['kcal']:,} kcal: {t['why']}"
        _save(data)
    logger.info("meal plan %s resized: %s", d["date"], d["adjusted"])
    return d["adjusted"]


def today() -> dict | None:
    import food
    week = get("this")
    d = food._now().date().isoformat()
    return next((x for x in (week or {}).get("days", []) if x["date"] == d), None)


def eat(date: str, index: int) -> dict | None:
    """He ate a planned meal: logged with the plan's exact numbers (no estimate), marked on the plan."""
    import food
    with _lock:
        data = _load()
        for week in data.values():
            for d in week.get("days", []):
                if d["date"] == date and 0 <= index < len(d["meals"]):
                    m = d["meals"][index]
                    slot = dict(SLOTS).get(m.get("slot"), None)
                    saved = food.add([{"name": m["name"], "qty": "as planned", **{k: m[k] for k in ("kcal", "protein", "carbs", "fat")}}],
                                     slot, source="plan", day=date)
                    d.setdefault("eaten", []).append(index)
                    _save(data)
                    return saved[0]
    return None


# ---------- his tools ----------

def meal_plan(args: Dict[str, Any]):
    from food import _owner_check
    from tools import ToolResult
    if err := _owner_check("meal_plan"):
        return err
    which = "next" if str(args.get("week") or "").lower() == "next" else "this"
    week = get(which)
    if args.get("make") or not week:
        state = start(which)
        return ToolResult(ok=True, tool_name="meal_plan", data={
            "status": "planning" if state == "started" else "already planning",
            "note": "Tell him you're putting the week's meals together (about a minute) and it'll be on the Food tab "
                    "with a grocery list; his phone gets a ping."})
    d = today() if which == "this" else None
    return ToolResult(ok=True, tool_name="meal_plan", data={
        "week": week["week"], "today": {"theme": d["theme"], "kind": d["kind"], "targets": d["targets"],
                                        "meals": [f"{m['slot']}: {m['name']} ({m['kcal']} kcal, {m['protein']} g protein)"
                                                  for m in d["meals"]]} if d else None,
        "days": [f"{x['weekday']}: {x['theme']}, {x['targets']['kcal']} kcal" for x in week["days"]],
        "grocery_items": len(week["groceries"]),
        "note": "Answer what he asked: today's meals, or the week in a sentence. The full plan and grocery list are on "
                "the Food tab."})


def meal_ate(args: Dict[str, Any]):
    """'I ate the lunch from the plan': logs today's planned meal by slot or name."""
    from food import _owner_check, _left_line, day as food_day
    from tools import ToolResult
    if err := _owner_check("meal_ate"):
        return err
    d = today()
    if not d:
        return ToolResult(ok=False, tool_name="meal_ate", error="no plan for today: offer to make one (meal_plan)")
    want = str(args.get("meal") or "").lower()
    i = next((i for i, m in enumerate(d["meals"]) if want and (want in m["slot"].lower() or want in m["name"].lower())), None)
    if i is None:
        return ToolResult(ok=False, tool_name="meal_ate", error="which meal: " + ", ".join(m["slot"] for m in d["meals"]))
    saved = eat(d["date"], i)
    return ToolResult(ok=True, tool_name="meal_ate", data={"logged": f"{saved['name']}: {saved['kcal']:.0f} kcal",
                                                           **_left_line(food_day())})


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "meal_plan",
        "description": "His weekly meal plan as his nutritionist: 'what should I eat today?', 'plan my meals this "
                       "week', 'what's for dinner?', 'make next week's plan'. make=true to (re)plan it.",
        "parameters": {"type": "object", "properties": {
            "week": {"type": "string", "enum": ["this", "next"]}, "make": {"type": "boolean"}}}}},
    {"type": "function", "function": {
        "name": "meal_ate", "description": "He ate a meal from today's plan ('I had the planned lunch', 'ate the "
                                           "breakfast'): logs it with the plan's exact numbers.",
        "parameters": {"type": "object", "properties": {"meal": {"type": "string", "description": "The slot or dish"}},
                       "required": ["meal"]}}},
]
