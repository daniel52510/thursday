"""THURSDAY as his personal trainer: the "Casino Royale" program, coached day by day.

Modeled on the publicly described prep for Daniel Craig's Casino Royale (Simon Waterson): heavy compound lifting
for size, fast full-body circuits and conditioning to get lean, and recovery treated as part of the plan. One week
repeats; every day's session is adjusted to his readiness (health.readiness: HRV, resting heart rate, sleep) so a
bad night means lighter work, not a skipped plan. He asks "what's my workout today?"; the morning briefing has it too.

General fitness coaching, not medical advice: the tool's note says so, and to stop on pain.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict

PROGRAM = "Casino Royale"
GOAL = ("Lean, athletic and visibly muscular (Daniel Craig in Casino Royale): build the chest, shoulders and back "
        "with heavy compound lifts, get lean with circuits and conditioning, and recover well.")

# Monday = 0. Each exercise: (name, sets x reps or time).
WEEK = {
    0: ("Push: chest, shoulders, triceps", [
        ("Barbell bench press", "4 x 6-8, heavy"), ("Incline dumbbell press", "3 x 8-10"),
        ("Standing overhead press", "3 x 6-8"), ("Weighted dips", "3 x as many as clean"),
        ("Lateral raises", "3 x 12-15"), ("Core: hanging knee raises + plank", "3 rounds")]),
    1: ("Pull: back and biceps (the V-taper)", [
        ("Pull-ups (weighted when 10+ are easy)", "4 x 6-10"), ("Bent-over barbell row", "4 x 6-8"),
        ("Single-arm dumbbell row", "3 x 10 each"), ("Face pulls", "3 x 15"),
        ("Barbell curls", "3 x 8-10"), ("Core: ab wheel", "3 x 10")]),
    2: ("Conditioning: intervals", [
        ("Warm-up jog", "10 min easy"), ("Intervals (run, bike or rower)", "8 x 1 min hard / 1 min easy"),
        ("Cool-down + mobility (hips, thoracic spine, shoulders)", "15 min")]),
    3: ("Legs and power", [
        ("Back squat", "4 x 6-8"), ("Romanian deadlift", "3 x 8"), ("Walking lunges", "3 x 12 each leg"),
        ("Box jumps", "4 x 5, full rest"), ("Calf raises", "3 x 15"), ("Core: hanging leg raises", "3 x 10")]),
    4: ("Bond circuit: full body, fast (the classic Craig session)", [
        ("5 rounds, 45 s work / 15 s move, 90 s between rounds", ""),
        ("Push-ups", "45 s"), ("Pull-ups or inverted rows", "45 s"), ("Dumbbell thrusters", "45 s"),
        ("Renegade rows", "45 s"), ("Burpees", "45 s"), ("Plank", "45 s")]),
    5: ("Long easy cardio (zone 2: you can still talk)", [
        ("Run, swim or bike", "45-60 min easy"), ("Stretch", "10 min")]),
    6: ("Rest and recover", [
        ("Walk", "30 min"), ("Mobility flow", "15 min"), ("Plan the week's meals", "")]),
}

NUTRITION = ["Protein at every meal: about 1 g per pound of body weight a day.",
             "A small calorie deficit (about 300-500 a day) to get lean without losing muscle.",
             "Mostly whole foods; carbs around training; easy on alcohol (it costs recovery).",
             "7.5+ hours of sleep: it's when the muscle is built."]

# Heavier athletes swap the high-impact moves for ones that train the same thing without pounding the knees and
# ankles (until they're under JOINT_FRIENDLY_LB). Same session, same effort.
JOINT_FRIENDLY_LB = 260
SWAPS = {"Box jumps": ("Weighted step-ups (knee height)", "4 x 6 each leg"),
         "Burpees": ("Dumbbell squat to press", "45 s"),
         "Intervals (run, bike or rower)": ("Intervals on the rower or bike (run them once under 260 lb)",
                                            "8 x 1 min hard / 1 min easy"),
         "Run, swim or bike": ("Incline walk, swim or bike", "45-60 min easy")}


# His apartment gym (photo, 2026-11-07): dumbbells to about 50+ lb, adjustable bench, cable machine, treadmills,
# ellipticals, upright bike; no barbell, rack, dip station or rower. Same muscles, same intent; with lighter
# dumbbells than a barbell allows, the main lifts go by reps and slow tempo instead of load.
APARTMENT = {
    "Barbell bench press": ("Dumbbell bench press (3 s down)", "4 x 8-12"),
    "Standing overhead press": ("Seated dumbbell shoulder press", "3 x 8-12"),
    "Weighted dips": ("Cable triceps pushdown", "3 x 12-15"),
    "Pull-ups (weighted when 10+ are easy)": ("Cable lat pulldown", "4 x 8-12"),
    "Bent-over barbell row": ("Seated cable row", "4 x 10"),
    "Face pulls": ("Cable face pulls", "3 x 15"),
    "Barbell curls": ("Dumbbell or cable curls", "3 x 10-12"),
    "Back squat": ("Goblet squat (3 s down)", "4 x 10-12"),
    "Romanian deadlift": ("Dumbbell Romanian deadlift", "3 x 10"),
    "Walking lunges": ("Dumbbell split squats", "3 x 10 each leg"),
    "Core: ab wheel": ("Core: cable crunches", "3 x 15"),
    "Core: hanging knee raises + plank": ("Core: cable woodchops + plank", "3 rounds"),
    "Core: hanging leg raises": ("Core: dead bugs", "3 x 10 each side"),
    "Pull-ups or inverted rows": ("Cable rows (fast)", "45 s"),
    "Intervals on the rower or bike (run them once under 260 lb)": ("Intervals on the upright bike or elliptical",
                                                                    "8 x 1 min hard / 1 min easy"),
    "Intervals (run, bike or rower)": ("Intervals on the bike, elliptical or treadmill", "8 x 1 min hard / 1 min easy"),
}


def profile() -> dict:
    """His stats from his facts (Settings → Memory): weight, height, background, goal."""
    from accounts import OWNER
    from agent_stream import initalize_db
    db = initalize_db()
    out = {k: db.get_fact(k, OWNER) for k in ("weight_lb", "height", "training_background", "fitness_goal",
                                             "goal_weight_lb", "age", "home_equipment")}
    try:
        out["weight_lb"] = float(str(out["weight_lb"]).split()[0]) if out.get("weight_lb") else None
    except ValueError:
        out["weight_lb"] = None
    return out


def targets(p: dict) -> dict:
    """Protein from the goal weight (not today's): Craig's Casino Royale build at 6'5" is about 245 lb."""
    goal = float(p.get("goal_weight_lb") or 0) or (245.0 if p.get("weight_lb") else None)
    if not goal:
        return {}
    out = {"goal_weight_lb": round(goal), "protein_g_per_day": round(goal),
           "pace": "about 1-1.5 lb a week down while lifting heavy (faster loses muscle)",
           "calories": "a 500-750 kcal daily deficit from maintenance (the scale tunes it: if he's not losing ~1 lb a "
                       "week after 2-3 weeks, take off another 200)"}
    kcal = maintenance(p)
    if kcal:
        out.update(maintenance_kcal=kcal, eat_kcal=f"{round(kcal - 750, -1):.0f}-{round(kcal - 500, -1):.0f}")
    return out


def maintenance(p: dict) -> int | None:
    """Mifflin-St Jeor (men) x 1.55 for training ~5 days a week. None without weight, height and age."""
    import re
    try:
        kg = float(p["weight_lb"]) * 0.4536
        ft, inch = (int(x) for x in re.findall(r"\d+", str(p["height"]))[:2])
        cm = (ft * 12 + inch) * 2.54
        age = int(str(p["age"]).split()[0])
    except (KeyError, TypeError, ValueError):
        return None
    return round((10 * kg + 6.25 * cm - 5 * age + 5) * 1.55, -1)


def _session(day: int, weight: float | None, apartment: bool = False):
    title, exercises = WEEK[day]
    if weight and weight >= JOINT_FRIENDLY_LB:
        exercises = [SWAPS.get(n, (n, r)) if n in SWAPS else (n, r) for n, r in exercises]
    if apartment:
        exercises = [APARTMENT.get(n, (n, r)) for n, r in exercises]
    return title, exercises


# Readiness (health.readiness) decides how hard today is.
def adjust(score: int | None) -> tuple[str, str]:
    if score is None:
        return "as planned", "No readiness data today: train as planned and go by feel."
    if score >= 80:
        return "push", "Readiness is high: go heavier on the main lift or add a set. A good day to beat last week."
    if score >= 60:
        return "as planned", "Readiness is good: train as planned."
    if score >= 40:
        return "lighter", ("Readiness is so-so: keep the main lifts but cut the volume by about a third, or swap "
                           "intervals for an easy zone-2 session.")
    return "recover", "Readiness is low: skip the hard work today. A walk and mobility; the plan resumes tomorrow."


def _owner() -> bool:
    from accounts import OWNER
    from calendar_tool import TURN
    turn = TURN.get()
    return bool(turn and turn.get("user") == OWNER)


def _today():
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo(os.getenv("THURSDAY_TZ", "America/New_York")))


def training_today(args: Dict[str, Any]):
    from tools import ToolResult
    import health
    if not _owner():
        return ToolResult(ok=False, tool_name="training_today", error="training_is_daniels_only")
    day = _today() + timedelta(days=1 if args.get("tomorrow") else 0)
    me = profile()
    title, exercises = _session(day.weekday(), me.get("weight_lb"), "apartment" in str(me.get("home_equipment") or ""))
    ready = health.readiness() if not args.get("tomorrow") else None
    mode, why = adjust(ready["score"] if ready else None)
    if args.get("tomorrow"):
        why = "Tomorrow's intensity depends on how he sleeps tonight; it's set in the morning."
    if mode == "recover":
        title, exercises = WEEK[6]
    # This week so far, from his synced Watch workouts.
    rows = health.daily(7)
    done = [w for d in rows for w in d.get("workouts", []) if w.get("minutes", 0) >= 8]
    import lifts  # his journal sets the weights: what he lifted last time, and whether it's time to go up
    loads = {} if mode == "recover" else {x["exercise"]: x for x in lifts.plan(day)["lifts"]}
    if mode == "lighter":
        for x in loads.values():
            x["sets"] = max(2, round(x["sets"] * 2 / 3))
    return ToolResult(ok=True, tool_name="training_today", data={
        "program": PROGRAM, "day": day.strftime("%A"), "session": title, "intensity": mode, "why": why,
        "exercises": [f"{n}: {r}" if r else n for n, r in exercises], "readiness": ready,
        "loads": [{"exercise": x["exercise"], "do": f"{x['sets']} x {x['reps']}" + (
            f" at {x['weight_lb']:g} lb" if x["weight_lb"] else " bodyweight" if x["weight_lb"] == 0 else ""),
            "last_time": x["last"], "why": x["why"]} for x in loads.values()],
        "workouts_last_7_days": len(done), "him": me, "targets": targets(me),
        "note": "You're his personal trainer. Say the session in a sentence or two (the focus and the first couple "
                "of moves with their weights from loads), the intensity and why, and offer the full list. Encouraging, "
                "specific, no lecture. If a lift went up since last time, say so. "
                "General fitness coaching, not medical advice: stop on sharp pain."})


def training_plan(args: Dict[str, Any]):
    from tools import ToolResult
    if not _owner():
        return ToolResult(ok=False, tool_name="training_plan", error="training_is_daniels_only")
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return ToolResult(ok=True, tool_name="training_plan", data={
        "program": PROGRAM, "goal": GOAL, "week": {names[d]: t for d, (t, _) in WEEK.items()},
        "nutrition": NUTRITION, "him": (me := profile()), "targets": targets(me),
        "joint_friendly": bool(me.get("weight_lb") and me["weight_lb"] >= JOINT_FRIENDLY_LB), "how_it_adapts": "Each day is adjusted to his readiness (HRV, resting heart rate, sleep).",
        "note": "Explain it like a coach: the idea, the week in a sentence, and the two habits that matter most. "
                "When he asks about food or protein, give his own numbers from targets (protein in grams a day, the "
                "deficit, the goal weight and pace) in plain words."})


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "training_today",
        "description": "Today's (or tomorrow's) workout from his Casino Royale training program, adjusted to his "
                       "readiness. 'What's my workout today?', 'what am I training tomorrow?', 'should I lift today?'.",
        "parameters": {"type": "object", "properties": {"tomorrow": {"type": "boolean"}}}}},
    {"type": "function", "function": {
        "name": "training_plan",
        "description": "His whole training program: the goal, the weekly split, nutrition and how it adapts. "
                       "'What's my plan?', 'how do I get in Bond shape?', 'what should I eat?'.",
        "parameters": {"type": "object", "properties": {}}}},
]
