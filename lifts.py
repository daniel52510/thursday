"""His lifting journal: every set he does (exercise, date, reps, weight), and what THURSDAY's trainer makes of it.

Getting it in, with the least typing:
  - by voice: "bench, 3 sets of 8 at 185", "I did goblet squats 4 x 10 with 50s" (lift_log)
  - after a workout: when his Watch logs a strength session, or in the evening if it didn't, THURSDAY asks
    (asks.py) "Log today's push session as planned: ...?"; one yes logs it, and he corrects any set by voice
  - the Lifts section of the Health page: confirm today's plan, add a set, remove a wrong one

What the trainer does with it (next_weight), per exercise, from his last two sessions of it:
  - every working set hit the top of the rep range: add weight (+5 lb upper body, +10 lb lower; dumbbells per hand)
  - he missed the bottom of the range: hold the weight; missed it two sessions running: back off 10 %
  - otherwise: same weight, more reps
Dumbbells stop at DUMBBELL_MAX_LB (his apartment gym): past that it progresses by reps and a slower tempo instead.
training_today puts the numbers in today's plan ("Dumbbell bench press 4 x 8-12 at 50 lb: last time 50 x 12, 12,
11, 10"), so the routine follows what he actually lifted.

Weights in pounds; 0 means bodyweight. Dumbbell weights are per hand.
"""
from __future__ import annotations

import difflib
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict

DB = os.path.join(os.getenv("THURSDAY_BRAIN_DIR", "./brain"), "lifts.db")
DUMBBELL_MAX_LB = float(os.getenv("LIFT_DUMBBELL_MAX_LB", "50"))
_PRESCRIPTION = re.compile(r"(\d+)\s*x\s*(\d+)(?:\s*-\s*(\d+))?", re.I)
_LOWER = re.compile(r"squat|deadlift|lunge|leg press|step-up|hip thrust|calf", re.I)
_DUMBBELL = re.compile(r"dumbbell|goblet|\bdb\b", re.I)

# The picker's preset library (Health → Lifts → Log a workout): the common lifts by muscle group. Anything else he
# adds as a custom exercise, kept for next time (custom table).
LIBRARY: dict[str, list[str]] = {
    "Chest": ["Barbell bench press", "Incline barbell bench press", "Decline barbell bench press", "Dumbbell bench press",
              "Incline dumbbell press", "Decline dumbbell press", "Dumbbell fly", "Incline dumbbell fly",
              "Cable crossover", "Low-to-high cable fly", "Machine chest press", "Pec deck", "Push-ups",
              "Weighted push-ups", "Chest dips", "Smith machine bench press", "Floor press", "Svend press"],
    "Back": ["Deadlift", "Pull-ups", "Weighted pull-ups", "Chin-ups", "Cable lat pulldown", "Close-grip lat pulldown",
             "Straight-arm pulldown", "Bent-over barbell row", "Pendlay row", "Single-arm dumbbell row",
             "Chest-supported dumbbell row", "Seated cable row", "T-bar row", "Machine row", "Inverted rows",
             "Rack pulls", "Back extensions", "Meadows row", "Kroc row", "Single-arm cable row"],
    "Shoulders": ["Standing overhead press", "Seated dumbbell shoulder press", "Arnold press", "Push press",
                  "Machine shoulder press", "Lateral raises", "Cable lateral raises", "Front raises",
                  "Rear delt fly", "Reverse pec deck", "Cable face pulls", "Upright rows", "Barbell shrugs",
                  "Dumbbell shrugs", "Landmine press", "Y-raises"],
    "Biceps": ["Barbell curls", "EZ-bar curls", "Dumbbell curls", "Hammer curls", "Incline dumbbell curls",
               "Preacher curls", "Cable curls", "Concentration curls", "Spider curls", "Reverse curls",
               "Bayesian cable curls"],
    "Triceps": ["Close-grip bench press", "Triceps dips", "Skull crushers", "Overhead dumbbell extension",
                "Cable triceps pushdown", "Rope pushdown", "Overhead cable extension", "Dumbbell kickbacks",
                "Diamond push-ups", "JM press"],
    "Quads": ["Back squat", "Front squat", "Goblet squat", "Hack squat", "Leg press", "Leg extensions",
              "Bulgarian split squats", "Dumbbell split squats", "Walking lunges", "Reverse lunges", "Step-ups",
              "Smith machine squat", "Pistol squats", "Sissy squats", "Belt squat"],
    "Hamstrings & glutes": ["Romanian deadlift", "Dumbbell Romanian deadlift", "Stiff-leg deadlift",
                            "Sumo deadlift", "Trap bar deadlift", "Good mornings", "Lying leg curls",
                            "Seated leg curls", "Nordic curls", "Hip thrusts", "Glute bridges", "Cable pull-throughs",
                            "Kettlebell swings", "Single-leg Romanian deadlift", "Hip abduction machine",
                            "Cable kickbacks"],
    "Calves": ["Standing calf raises", "Seated calf raises", "Single-leg calf raises", "Leg press calf raises"],
    "Core": ["Plank", "Side plank", "Hanging leg raises", "Hanging knee raises", "Ab wheel rollouts",
             "Cable crunches", "Cable woodchops", "Pallof press", "Dead bugs", "Russian twists", "Decline sit-ups",
             "Weighted crunches", "Mountain climbers", "Dragon flags"],
    "Full body": ["Power clean", "Hang clean", "Clean and jerk", "Snatch", "Thrusters", "Dumbbell thrusters",
                  "Farmer's carry", "Suitcase carry", "Turkish get-up", "Burpees", "Sled push", "Battle ropes",
                  "Medicine ball slams", "Renegade rows"],
}


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB) or ".", exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS sets (id TEXT PRIMARY KEY, day TEXT NOT NULL, exercise TEXT NOT NULL,
        set_no INTEGER NOT NULL, reps INTEGER NOT NULL, weight_lb REAL NOT NULL, source TEXT, created REAL NOT NULL)""")
    conn.execute("CREATE INDEX IF NOT EXISTS sets_exercise_day ON sets (exercise, day)")
    conn.execute("CREATE TABLE IF NOT EXISTS custom (name TEXT PRIMARY KEY COLLATE NOCASE, category TEXT, created REAL)")
    return conn


def library() -> dict[str, list[str]]:
    """The presets by group, his custom exercises first under "My exercises"."""
    with _connect() as conn:
        mine = [r[0] for r in conn.execute("SELECT name FROM custom ORDER BY created DESC")]
    return {"My exercises": mine, **LIBRARY} if mine else dict(LIBRARY)


def add_custom(name: str, category: str = "") -> str:
    name = " ".join(str(name or "").split())[:60]
    if name:
        with _connect() as conn:
            conn.execute("INSERT OR IGNORE INTO custom VALUES (?, ?, ?)", (name, category[:30], time.time()))
    return name


def _today() -> datetime:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo(os.getenv("THURSDAY_TZ", "America/New_York")))


def known_exercises() -> list[str]:
    """Every lift in his program, as written there and as his apartment gym swaps it."""
    import trainer
    names = {n for _, ex in trainer.WEEK.values() for n, _ in ex}
    names |= {n for n, _ in trainer.SWAPS.values()} | {n for n, _ in trainer.APARTMENT.values()}
    return sorted(n for n in names if not n.lower().startswith(("core:", "warm", "cool", "walk", "stretch", "plan")))


def canonical(name: str) -> str:
    """His words → the program's name for the lift when it's clearly the same one ("bench" → "Dumbbell bench
    press (3 s down)" in his gym), else his words, tidied."""
    said = " ".join(str(name or "").split())[:60]
    if not said:
        return ""
    low = said.lower()
    names = known_exercises()
    exact = names + [n for group in library().values() for n in group]  # picked from the library: kept as is
    for n in exact:
        if n.lower() == low:
            return n
    # Prefer what his gym has (the apartment swaps), then the program's own names.
    import trainer
    gym = [n for n, _ in trainer.APARTMENT.values()]
    for pool in (gym, names):
        words = [w[:-1] if len(w) > 4 and w.endswith("s") else w for w in re.findall(r"[a-z]+", low) if len(w) > 2]
        hits = [n for n in pool if words and all(w in n.lower() for w in words)]
        if len(hits) == 1 or (hits and len(low) >= 4):
            return min(hits, key=len)
    close = difflib.get_close_matches(low, [n.lower() for n in names], n=1, cutoff=0.75)
    return next((n for n in names if n.lower() == close[0]), said.title()) if close else said[0].upper() + said[1:]


def _est_1rm(weight: float, reps: int) -> float:
    return weight * (1 + reps / 30) if weight and reps else 0.0  # Epley


def add(exercise: str, sets: list[dict], day: str | None = None, source: str = "voice") -> dict:
    """Saves sets [{reps, weight_lb}] for one exercise on one day (today by default). Returns what was saved, and
    whether it's a personal best (estimated one-rep max)."""
    ex = canonical(exercise)
    day = day or _today().date().isoformat()
    clean = [{"reps": max(0, min(100, int(s.get("reps") or 0))), "weight_lb": max(0.0, min(1500.0, float(s.get("weight_lb") or 0)))}
             for s in sets if int(s.get("reps") or 0) > 0][:20]
    if not ex or not clean:
        return {"error": "need an exercise and at least one set with reps"}
    with _connect() as conn:
        before = max((_est_1rm(r["weight_lb"], r["reps"]) for r in conn.execute(
            "SELECT exercise, reps, weight_lb FROM sets") if _base(r["exercise"]) == _base(ex)), default=0.0)
        n0 = conn.execute("SELECT COALESCE(MAX(set_no), 0) FROM sets WHERE exercise = ? AND day = ?", (ex, day)).fetchone()[0]
        for i, s in enumerate(clean, 1):
            conn.execute("INSERT INTO sets VALUES (?,?,?,?,?,?,?,?)",
                         (uuid.uuid4().hex[:10], day, ex, n0 + i, s["reps"], s["weight_lb"], source, time.time()))
    best = max(_est_1rm(s["weight_lb"], s["reps"]) for s in clean)
    return {"exercise": ex, "day": day, "sets": clean, "personal_best": bool(before and best > before + 0.5)}


def save_workout(day: str | None, exercises: list[dict]) -> list[dict]:
    """A whole session from the workout builder: [{exercise, sets: [{reps, weight_lb}], custom}] in one go."""
    saved = []
    for e in exercises[:30]:
        if e.get("custom"):
            add_custom(str(e.get("exercise") or ""), str(e.get("category") or ""))
        out = add(str(e.get("exercise") or ""), list(e.get("sets") or []), day, source="builder")
        if "error" not in out:
            saved.append(out)
    return saved


def last_sets(exercise: str) -> dict:
    """What to prefill when he adds an exercise: last session's sets, and the trainer's next weight."""
    ex = canonical(exercise)
    hist = sessions(ex, 1)
    sets = [{"reps": s["reps"], "weight_lb": s["weight_lb"]} for s in hist[0]["sets"]] if hist else []
    nxt = next_weight(ex, max(3, len(sets)), 8, 12) if hist else None
    return {"exercise": ex, "last": sets, "last_day": hist[0]["day"] if hist else None,
            "suggested_weight_lb": nxt["weight_lb"] if nxt else None}


def remove(set_id: str) -> bool:
    with _connect() as conn:
        return conn.execute("DELETE FROM sets WHERE id = ?", (set_id,)).rowcount > 0


def remove_last(exercise: str, day: str | None = None) -> int:
    """Removes one exercise's sets on a day (his "scratch that, I didn't do curls")."""
    ex, day = canonical(exercise), day or _today().date().isoformat()
    with _connect() as conn:
        return conn.execute("DELETE FROM sets WHERE exercise = ? AND day = ?", (ex, day)).rowcount


def _base(name: str) -> str:
    """"Dumbbell bench press (3 s down)" and "Dumbbell bench press" are the same lift for his history."""
    return re.sub(r"\s*\(.*?\)", "", name or "").strip().lower()


def sessions(exercise: str, limit: int = 2) -> list[dict]:
    """His last `limit` days of one exercise (any of its names), newest first: [{day, sets: [{reps, weight_lb}]}]."""
    with _connect() as conn:
        names = [r[0] for r in conn.execute("SELECT DISTINCT exercise FROM sets") if _base(r[0]) == _base(exercise)]
        if not names:
            return []
        marks = ",".join("?" * len(names))
        days = [r[0] for r in conn.execute(f"SELECT DISTINCT day FROM sets WHERE exercise IN ({marks}) "
                                           "ORDER BY day DESC LIMIT ?", (*names, limit))]
        return [{"day": d, "sets": [dict(r) for r in conn.execute(
            f"SELECT id, reps, weight_lb FROM sets WHERE exercise IN ({marks}) AND day = ? ORDER BY created, set_no",
            (*names, d))]} for d in days]


def journal(days: int = 30) -> list[dict]:
    """Day by day, newest first: [{day, exercises: [{exercise, sets: [{id, reps, weight_lb}]}]}]."""
    since = (_today() - timedelta(days=days)).date().isoformat()
    out: dict[str, dict[str, list]] = {}
    with _connect() as conn:
        for r in conn.execute("SELECT * FROM sets WHERE day >= ? ORDER BY day DESC, created, set_no", (since,)):
            out.setdefault(r["day"], {}).setdefault(r["exercise"], []).append(
                {"id": r["id"], "reps": r["reps"], "weight_lb": r["weight_lb"]})
    return [{"day": d, "exercises": [{"exercise": e, "sets": s} for e, s in ex.items()]} for d, ex in out.items()]


def _say_sets(sets: list[dict]) -> str:
    w = {s["weight_lb"] for s in sets}
    reps = ", ".join(str(s["reps"]) for s in sets)
    if len(w) == 1:
        lb = next(iter(w))
        return f"{lb:g} lb x {reps}" if lb else f"bodyweight x {reps}"
    return ", ".join(f"{s['weight_lb']:g}x{s['reps']}" for s in sets)


def parse(prescription: str) -> tuple[int, int, int] | None:
    """"4 x 6-8, heavy" → (4, 6, 8); "3 x 12 each leg" → (3, 12, 12); anything timed or unnumbered → None."""
    m = _PRESCRIPTION.search(prescription or "")
    if not m or re.match(r"\s*(?:min|sec|s\b)", prescription[m.end():]):
        return None  # "8 x 1 min hard / 1 min easy" is intervals, not sets of reps
    lo = int(m[2])
    return int(m[1]), lo, int(m[3] or lo)


def next_weight(exercise: str, sets: int, lo: int, hi: int) -> dict:
    """The trainer's call for today: {"weight_lb", "reps", "why", "last"}; weight None when there's no history."""
    hist = sessions(exercise, 2)
    if not hist:
        return {"weight_lb": None, "reps": f"{lo}-{hi}" if hi != lo else str(lo), "last": None,
                "why": f"First time logged: pick a weight you can do for {hi} with two reps to spare."}
    last = hist[0]
    top = max(s["weight_lb"] for s in last["sets"])
    work = [s for s in last["sets"] if s["weight_lb"] == top]
    said_last = f"{_say_sets(last['sets'])} on {datetime.fromisoformat(last['day']).strftime('%b %-d')}"
    rep_text = f"{lo}-{hi}" if hi != lo else str(lo)
    dumbbell = bool(_DUMBBELL.search(exercise))
    step = 10.0 if _LOWER.search(exercise) and not dumbbell else 5.0
    if top == 0:  # bodyweight: reps are the progression
        return {"weight_lb": 0, "reps": rep_text, "last": said_last,
                "why": "Bodyweight: beat last time's reps; add weight once the top of the range is easy."}
    if len(work) >= sets and all(s["reps"] >= hi for s in work):
        nxt = top + step
        if dumbbell and nxt > DUMBBELL_MAX_LB:
            return {"weight_lb": top, "reps": rep_text, "last": said_last,
                    "why": f"Topped out the {top:g}s: same weight, slower (3 s down) and a pause at the bottom, "
                           "or add a set."}
        return {"weight_lb": nxt, "reps": rep_text, "last": said_last,
                "why": f"Hit {hi} on every set last time: up {step:g} lb."}
    missed = any(s["reps"] < lo for s in work)
    if missed and len(hist) > 1:
        prev_top = max(s["weight_lb"] for s in hist[1]["sets"])
        prev_missed = any(s["reps"] < lo for s in hist[1]["sets"] if s["weight_lb"] == prev_top)
        if prev_top == top and prev_missed:
            back = max(0.0, round(top * 0.9 / 5) * 5)
            return {"weight_lb": back, "reps": rep_text, "last": said_last,
                    "why": f"Missed {lo} two sessions running at {top:g}: back off to {back:g} and build again."}
    return {"weight_lb": top, "reps": rep_text, "last": said_last,
            "why": "Same weight: one more rep than last time on each set." if not missed
            else f"Short of {lo} last time: stay at {top:g} and own it."}


def plan(day: datetime | None = None) -> dict:
    """Today's lifts from his program (his gym's version), each with the trainer's weight from his journal."""
    import trainer
    day = day or _today()
    me = trainer.profile()
    title, exercises = trainer._session(day.weekday(), me.get("weight_lb"),
                                        "apartment" in str(me.get("home_equipment") or ""))
    lifts = []
    for name, rx in exercises:
        p = parse(rx)
        if not p or name.lower().startswith("core"):
            continue
        sets, lo, hi = p
        lifts.append({"exercise": name, "sets": sets, **next_weight(name, sets, lo, hi)})
    return {"day": day.date().isoformat(), "session": title, "lifts": lifts}


def log_plan(day: str | None = None) -> list[dict]:
    """"Yes, as planned": each planned lift with a weight, as prescribed (top of the range). Lifts with no weight
    yet are left for him to say."""
    p = plan()
    saved = []
    for lift in p["lifts"]:
        if lift["weight_lb"] is None:
            continue
        reps = int(str(lift["reps"]).split("-")[-1])
        saved.append(add(lift["exercise"], [{"reps": reps, "weight_lb": lift["weight_lb"]}] * lift["sets"],
                         day or p["day"], source="confirmed"))
    return saved


def logged_today() -> bool:
    with _connect() as conn:
        return conn.execute("SELECT 1 FROM sets WHERE day = ?", (_today().date().isoformat(),)).fetchone() is not None


# ---------- his tools ----------

def _owner_check(name: str):
    from accounts import OWNER
    from calendar_tool import TURN
    from tools import ToolResult
    turn = TURN.get()
    if not turn or turn.get("user") != OWNER:
        return ToolResult(ok=False, tool_name=name, error="training_is_daniels_only")
    return None


def lift_log(args: Dict[str, Any]):
    from tools import ToolResult
    if err := _owner_check("lift_log"):
        return err
    if isinstance(args.get("sets_detail"), list) and args["sets_detail"]:
        sets = [{"reps": s.get("reps"), "weight_lb": s.get("weight_lb", args.get("weight_lb", 0))}
                for s in args["sets_detail"] if isinstance(s, dict)]
    else:
        n = max(1, min(20, int(args.get("sets") or 1)))
        sets = [{"reps": args.get("reps"), "weight_lb": args.get("weight_lb", 0)}] * n
    out = add(str(args.get("exercise") or ""), sets, args.get("day") or None)
    if "error" in out:
        return ToolResult(ok=False, tool_name="lift_log", error=out["error"] + ": ask him what he did")
    return ToolResult(ok=True, tool_name="lift_log", data={
        **out, "note": "Confirm in a few words (the lift, sets x reps at the weight)" +
        (", and that it's a personal best" if out["personal_best"] else "") + ". If the name isn't what he said, "
        "say the name you logged it under."})


def lift_undo(args: Dict[str, Any]):
    from tools import ToolResult
    if err := _owner_check("lift_undo"):
        return err
    n = remove_last(str(args.get("exercise") or ""), args.get("day") or None)
    return ToolResult(ok=bool(n), tool_name="lift_undo", data={"removed_sets": n},
                      **({} if n else {"error": "nothing_logged_for_that_today"}))


def lift_history(args: Dict[str, Any]):
    from tools import ToolResult
    if err := _owner_check("lift_history"):
        return err
    if args.get("exercise"):
        ex = canonical(str(args["exercise"]))
        hist = sessions(ex, 6)
        best = max((_est_1rm(s["weight_lb"], s["reps"]) for h in hist for s in h["sets"]), default=0)
        return ToolResult(ok=True, tool_name="lift_history", data={
            "exercise": ex, "sessions": [{"day": h["day"], "sets": _say_sets(h["sets"])} for h in hist],
            "estimated_one_rep_max_lb": round(best) or None,
            "note": "Sum up the trend in a sentence or two (up, flat, down) and what's next."})
    rows = journal(int(args.get("days") or 14))
    return ToolResult(ok=True, tool_name="lift_history", data={
        "days": [{"day": d["day"], "lifts": [f"{e['exercise']}: {_say_sets(e['sets'])}" for e in d["exercises"]]}
                 for d in rows], "note": "Highlights only: how many sessions, what moved up."})


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "lift_log",
        "description": "Log lifting he did into his journal: 'bench 3 sets of 8 at 185', 'goblet squats 4 x 10 with "
                       "the 50s', 'pull-ups 3 x 8'. Dumbbell weights are per hand; bodyweight is 0. Use sets_detail "
                       "when sets differ ('185 for 8, 8, then 7 and 6').",
        "parameters": {"type": "object", "properties": {
            "exercise": {"type": "string"}, "sets": {"type": "integer"}, "reps": {"type": "integer"},
            "weight_lb": {"type": "number"},
            "sets_detail": {"type": "array", "items": {"type": "object", "properties": {
                "reps": {"type": "integer"}, "weight_lb": {"type": "number"}}}},
            "day": {"type": "string", "description": "YYYY-MM-DD if not today ('yesterday I did...')"}},
            "required": ["exercise"]}}},
    {"type": "function", "function": {
        "name": "lift_undo", "description": "Remove an exercise he logged by mistake (today unless day is given).",
        "parameters": {"type": "object", "properties": {"exercise": {"type": "string"}, "day": {"type": "string"}},
                       "required": ["exercise"]}}},
    {"type": "function", "function": {
        "name": "lift_history",
        "description": "His lifting journal: one exercise's last sessions and estimated max ('how's my bench going?'), "
                       "or the last couple of weeks ('what did I lift this week?').",
        "parameters": {"type": "object", "properties": {"exercise": {"type": "string"}, "days": {"type": "integer"}}}}},
]
