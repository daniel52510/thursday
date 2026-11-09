"""The coach's brain: what actually happened (sleep, the session he did or skipped, what he ate, his weight), read in
one place, so training, nutrition, the meal plan and the morning briefing all decide from the same facts.

  sleep      last night (Apple Health, health.daily), the run of short nights, and his real pattern (when he falls
             asleep, how long he sleeps) for the bedtime nudge
  training   what he did today (his lifting journal, or a Watch workout), versus what was planned; a lifting session
             he missed in the last two days moves to today when today isn't a lifting day
  fuel       yesterday's eating against yesterday's target
and the rules built on them (all evidence-informed, none of them diagnosis):
  - calories follow the session he did, not the plan: no session by 7 pm drops the training-day bump
  - under 6 h of sleep: the deficit eases by 200 kcal that day (short sleep on a diet costs more muscle)
  - two short nights running: the main lifts drop 5 %
  - a big shortfall yesterday (under 70 % of calories or protein): no pushing today, and a fueling note
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict

SHORT_NIGHT_MIN = 360
SKIP_AFTER_HOUR = 19
SHORT_SLEEP_KCAL = 200
UNDERFED = 0.7
_STRENGTH = re.compile(r"strength|functional|weight|core training|cross training|hiit", re.I)
_CARDIO = re.compile(r"run|cycl|bike|elliptical|row|swim|walk|hik|stair|cardio|tennis|basketball|soccer|boxing", re.I)


def _now() -> datetime:
    import food
    return food._now()


def _health_days(n: int) -> dict[str, dict]:
    try:
        import health
        return {r["date"]: r for r in health.daily(n)}
    except Exception:
        return {}


# ---------- sleep ----------

def last_night(day: datetime | None = None) -> int | None:
    day = day or _now()
    return _health_days(2).get(day.date().isoformat(), {}).get("sleep_minutes")


def short_streak(day: datetime | None = None) -> int:
    """Short nights in a row, ending last night."""
    day = day or _now()
    rows = _health_days(10)
    n = 0
    for i in range(10):
        m = rows.get((day - timedelta(days=i)).date().isoformat(), {}).get("sleep_minutes")
        if m is None or m >= SHORT_NIGHT_MIN:
            break
        n += 1
    return n


def sleep_pattern(days: int = 14) -> dict:
    """His real nights: average length, and when he usually falls asleep (from when each night's sleep starts)."""
    try:
        import health
        since = (_now() - timedelta(days=days)).astimezone().isoformat()
        with health._connect() as conn:
            rows = conn.execute("SELECT start, end FROM samples WHERE metric IN ('sleep_core','sleep_deep','sleep_rem') "
                                "AND start >= ?", (since,)).fetchall()
    except Exception:
        rows = []
    zone = _now().tzinfo
    starts: dict[str, datetime] = {}
    for start, end in rows:
        s, e = datetime.fromisoformat(start).astimezone(zone), datetime.fromisoformat(end).astimezone(zone)
        night = e.date().isoformat()
        if night not in starts or s < starts[night]:
            starts[night] = s
    lengths = [r.get("sleep_minutes") for r in _health_days(days).values() if r.get("sleep_minutes")]
    out: dict[str, Any] = {"nights": len(lengths)}
    if lengths:
        out["avg_minutes"] = round(sum(lengths) / len(lengths))
    if starts:
        mins = [(s.hour + (24 if s.hour < 12 else 0)) * 60 + s.minute for s in starts.values()]  # past midnight counts late
        avg = round(sum(mins) / len(mins)) % (24 * 60)
        out["usual_onset"] = f"{avg // 60:02d}:{avg % 60:02d}"
    return out


# ---------- training ----------

def did(day: datetime) -> dict:
    """What he did that day: {"kind": "lift" | "cardio" | None, "what": ...}."""
    d = day.date().isoformat()
    try:
        import lifts
        with lifts._connect() as conn:
            n = conn.execute("SELECT COUNT(DISTINCT exercise) FROM sets WHERE day = ?", (d,)).fetchone()[0]
        if n:
            return {"kind": "lift", "what": f"{n} lifts logged"}
    except Exception:
        pass
    workouts = [w for w in _health_days(10).get(d, {}).get("workouts", []) if w.get("minutes", 0) >= 15]
    for w in workouts:
        if _STRENGTH.search(w.get("type", "")):
            return {"kind": "lift", "what": f"{w['type']} {round(w['minutes'])} min"}
    for w in workouts:
        if _CARDIO.search(w.get("type", "")) and (w["minutes"] >= 30 or "walk" not in w["type"].lower()):
            return {"kind": "cardio", "what": f"{w['type']} {round(w['minutes'])} min"}
    return {"kind": None, "what": ""}


def effective_kind(day: datetime) -> tuple[str, str]:
    """The day's kind for its calories, and why: what he did; or, today, the plan until it's clearly skipped."""
    import food
    planned = food._day_kind(day)
    today = _now()
    if day.date() > today.date():
        return planned, "planned"
    done = did(day)
    if done["kind"]:
        return done["kind"], f"did: {done['what']}"
    if planned != "rest" and (day.date() < today.date() or today.hour >= SKIP_AFTER_HOUR):
        return "rest", f"no {planned} session recorded"
    return planned, "planned"


def missed_session(day: datetime | None = None) -> dict | None:
    """A lifting day in the last two days with nothing recorded, when today isn't a lifting day: do it today."""
    import food
    import trainer
    day = day or _now()
    if food._day_kind(day) == "lift":
        return None
    for back in (1, 2):
        d = day - timedelta(days=back)
        if food._day_kind(d) == "lift" and not did(d)["kind"]:
            return {"from": d.strftime("%A"), "weekday": d.weekday(), "session": trainer.WEEK[d.weekday()][0], "day": d}
    return None


# ---------- fuel ----------

def fuel_yesterday(day: datetime | None = None) -> dict:
    import food
    y = (day or _now()) - timedelta(days=1)
    d = food.day(y.date().isoformat())
    t = food.plan_today(y)
    out = {"logged": d["totals"]["kcal"] >= food.FULL_DAY_KCAL, "kcal": d["totals"]["kcal"], "kcal_target": t["kcal"],
           "protein": d["totals"]["protein"], "protein_target": t["protein"]}
    out["underfed"] = out["logged"] and (out["kcal"] < UNDERFED * t["kcal"] or out["protein"] < UNDERFED * t["protein"])
    return out


# ---------- the morning briefing ----------

def briefing() -> dict:
    """Everything that shapes today, and today's plan with the reasons: one dict, and a few spoken sentences."""
    import food
    import mealplan
    import trainer
    from calendar_tool import TURN
    from accounts import OWNER
    now = _now()
    token = TURN.set({"user": OWNER, "conversation": "coach", "turn": "coach", "user_said": ""})
    try:
        t = trainer.training_today({}).data
    finally:
        TURN.reset(token)
    n = food.targets()
    sleep = last_night(now)
    pattern = sleep_pattern()
    fy = fuel_yesterday(now)
    yesterday = did(now - timedelta(days=1))
    c = food.coach()
    plan = mealplan.today()
    parts = []
    if sleep:
        parts.append(f"{sleep // 60} h {sleep % 60:02d} of sleep" + (f" (you've been averaging {pattern['avg_minutes'] // 60} h "
                                                                    f"{pattern['avg_minutes'] % 60:02d})" if pattern.get("avg_minutes") else ""))
    if (t.get("readiness") or {}).get("score") is not None:
        parts.append(f"readiness {t['readiness']['score']}")
    if yesterday["kind"]:
        parts.append(f"yesterday: {yesterday['what']}")
    if fy["logged"]:
        parts.append(f"you ate {fy['kcal']:,} of {fy['kcal_target']:,} kcal and {fy['protein']} g protein yesterday")
    if c.get("lb_per_week") is not None:
        parts.append(f"weight {c['lb_per_week']:+} lb a week")
    first = (t.get("loads") or [{}])[0]
    how = {"push": "go heavy", "as planned": "as planned", "lighter": "lighter today", "recover": "recovery only"}
    session = (f"Today: {t['session'].split(':')[0].lower()} day, {how.get(t['intensity'], t['intensity'])}"
               + (f" ({t['moved_from']}'s session)" if t.get("moved_from") else "")
               + (f", starting with {re.sub(r' [(].*?[)]', '', first['exercise'])} {first['do']}" if first.get("exercise") else ""))
    eat = f"{n['kcal']:,} kcal and {n['protein']} g protein"
    if plan:
        nxt = next((m for i, m in enumerate(plan["meals"]) if i not in plan.get("eaten", [])), None)
        if nxt:
            eat += f"; first up, {nxt['name'].lower()}"
    text = (("; ".join(parts).capitalize() + ". ") if parts else "") + f"{session}. Eat {eat}. {t['why']}"
    return {"text": text.replace("..", "."), "training": {k: t.get(k) for k in ("session", "intensity", "why", "moved_from")},
            "nutrition": {k: n[k] for k in ("kcal", "protein", "carbs", "fat", "why")}, "sleep_minutes": sleep,
            "sleep_pattern": pattern, "yesterday_trained": yesterday, "yesterday_ate": fy, "weight": c.get("weight_lb")}


def coach_today(args: Dict[str, Any]):
    from food import _owner_check
    from tools import ToolResult
    if err := _owner_check("coach_today"):
        return err
    b = briefing()
    return ToolResult(ok=True, tool_name="coach_today", data={**b, "note": "You're his coach: say today's plan and the "
                      "one or two reasons that matter most, in three sentences or fewer."})


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "coach_today",
        "description": "His day as his coach sees it: sleep, recovery, what he trained and ate yesterday, his weight "
                       "trend, and today's session and food with the reasons. 'What's the plan today?', 'how am I "
                       "doing?', 'morning briefing'.",
        "parameters": {"type": "object", "properties": {}}}},
]
