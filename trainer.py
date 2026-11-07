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
    title, exercises = WEEK[day.weekday()]
    ready = health.readiness() if not args.get("tomorrow") else None
    mode, why = adjust(ready["score"] if ready else None)
    if mode == "recover":
        title, exercises = WEEK[6]
    # This week so far, from his synced Watch workouts.
    rows = health.daily(7)
    done = [w for d in rows for w in d.get("workouts", []) if w.get("minutes", 0) >= 8]
    return ToolResult(ok=True, tool_name="training_today", data={
        "program": PROGRAM, "day": day.strftime("%A"), "session": title, "intensity": mode, "why": why,
        "exercises": [f"{n}: {r}" if r else n for n, r in exercises], "readiness": ready,
        "workouts_last_7_days": len(done),
        "note": "You're his personal trainer. Say the session in a sentence or two (the focus and the first couple "
                "of moves), the intensity and why, and offer the full list. Encouraging, specific, no lecture. "
                "General fitness coaching, not medical advice: stop on sharp pain."})


def training_plan(args: Dict[str, Any]):
    from tools import ToolResult
    if not _owner():
        return ToolResult(ok=False, tool_name="training_plan", error="training_is_daniels_only")
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return ToolResult(ok=True, tool_name="training_plan", data={
        "program": PROGRAM, "goal": GOAL, "week": {names[d]: t for d, (t, _) in WEEK.items()},
        "nutrition": NUTRITION, "how_it_adapts": "Each day is adjusted to his readiness (HRV, resting heart rate, sleep).",
        "note": "Explain it like a coach: the idea, the week in a sentence, and the two habits that matter most."})


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
