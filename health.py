"""His Apple Health data, kept on the Mac (brain/health.db) for THURSDAY to answer from. Nothing here leaves it.

Apple keeps Health data on the phone (no Mac or iCloud API), so the phone sends it: an iOS Shortcut (or the Health
Auto Export app) POSTs samples to /api/health/ingest with a device token from Settings → Devices, and the one-time
"Export All Data" zip fills in the history (scripts/import_health_export.py). Both land in one table, deduplicated,
so re-sending the same samples is harmless.

Metric names are short and stable ("heart_rate", "sleep_deep", ...); HealthKit's identifiers and the export
apps' names map onto them in METRICS. Sleep is stored as one sample per stage interval, value = minutes.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable

DB = os.path.join(os.getenv("THURSDAY_BRAIN_DIR", "./brain"), "health.db")

# HealthKit identifier / export-app name -> (our metric, unit we store)
METRICS: dict[str, tuple[str, str]] = {
    "HKQuantityTypeIdentifierStepCount": ("steps", "count"),
    "HKQuantityTypeIdentifierHeartRate": ("heart_rate", "bpm"),
    "HKQuantityTypeIdentifierRestingHeartRate": ("resting_heart_rate", "bpm"),
    "HKQuantityTypeIdentifierWalkingHeartRateAverage": ("walking_heart_rate", "bpm"),
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": ("hrv", "ms"),
    "HKQuantityTypeIdentifierOxygenSaturation": ("blood_oxygen", "%"),
    "HKQuantityTypeIdentifierRespiratoryRate": ("respiratory_rate", "breaths/min"),
    "HKQuantityTypeIdentifierActiveEnergyBurned": ("active_energy", "kcal"),
    "HKQuantityTypeIdentifierAppleExerciseTime": ("exercise_minutes", "min"),
    "HKQuantityTypeIdentifierAppleStandTime": ("stand_minutes", "min"),
    "HKQuantityTypeIdentifierDistanceWalkingRunning": ("distance", "km"),
    "HKQuantityTypeIdentifierFlightsClimbed": ("flights", "count"),
    "HKQuantityTypeIdentifierVO2Max": ("vo2max", "mL/kg/min"),
    "HKQuantityTypeIdentifierBodyMass": ("weight", "kg"),
    "HKQuantityTypeIdentifierAppleSleepingWristTemperature": ("wrist_temperature", "degC"),
    "HKQuantityTypeIdentifierTimeInDaylight": ("daylight_minutes", "min"),
}
# Health Auto Export / Shortcut names for the same things
ALIASES = {"step_count": "steps", "heart_rate_variability": "hrv", "resting_heart_rate": "resting_heart_rate",
           "walking_heart_rate_average": "walking_heart_rate", "active_energy": "active_energy",
           "apple_exercise_time": "exercise_minutes", "apple_stand_time": "stand_minutes",
           "walking_running_distance": "distance", "flights_climbed": "flights", "vo2_max": "vo2max",
           "weight_body_mass": "weight", "body_mass": "weight", "respiratory_rate": "respiratory_rate",
           "blood_oxygen_saturation": "blood_oxygen", "oxygen_saturation": "blood_oxygen",
           "apple_sleeping_wrist_temperature": "wrist_temperature", "time_in_daylight": "daylight_minutes",
           "heart_rate": "heart_rate", "steps": "steps", "hrv": "hrv"}
SLEEP = {"HKCategoryValueSleepAnalysisAsleepCore": "sleep_core", "HKCategoryValueSleepAnalysisAsleepDeep": "sleep_deep",
         "HKCategoryValueSleepAnalysisAsleepREM": "sleep_rem", "HKCategoryValueSleepAnalysisAwake": "sleep_awake",
         "HKCategoryValueSleepAnalysisInBed": "sleep_in_bed", "HKCategoryValueSleepAnalysisAsleepUnspecified": "sleep_asleep",
         "HKCategoryValueSleepAnalysisAsleep": "sleep_asleep",
         "core": "sleep_core", "deep": "sleep_deep", "rem": "sleep_rem", "awake": "sleep_awake",
         "in bed": "sleep_in_bed", "inbed": "sleep_in_bed", "asleep": "sleep_asleep"}
KNOWN = {m for m, _ in METRICS.values()} | set(SLEEP.values()) | {"workout"}


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB) or ".", exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS samples (
        metric TEXT NOT NULL, start TEXT NOT NULL, end TEXT NOT NULL, value REAL NOT NULL, unit TEXT,
        source TEXT, note TEXT, UNIQUE (metric, start, end, source))""")
    conn.execute("CREATE INDEX IF NOT EXISTS samples_metric_start ON samples (metric, start)")
    return conn


def _iso(when: Any) -> str | None:
    """Anything a phone sends as a time -> UTC ISO ("2026-11-06 07:12:00 -0500", ISO strings, epoch seconds)."""
    if when in (None, ""):
        return None
    try:
        if isinstance(when, (int, float)):
            dt = datetime.fromtimestamp(float(when), timezone.utc)
        else:
            s = str(when).strip().replace("Z", "+00:00")
            for fmt in ("%Y-%m-%d %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z"):
                try:
                    dt = datetime.strptime(s, fmt)
                    break
                except ValueError:
                    continue
            else:
                dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.astimezone()  # the Mac's zone, like the phone's
        return dt.astimezone(timezone.utc).isoformat(timespec="seconds")
    except (ValueError, OSError, OverflowError):
        return None


def store(rows: Iterable[tuple]) -> int:
    """(metric, start, end, value, unit, source, note) rows; duplicates are ignored. Returns how many were new."""
    with _connect() as conn:
        before = conn.total_changes
        conn.executemany("INSERT OR IGNORE INTO samples VALUES (?,?,?,?,?,?,?)", rows)
        return conn.total_changes - before


def _metric(name: str) -> str | None:
    n = (name or "").strip()
    if n in METRICS:
        return METRICS[n][0]
    n = n.lower().replace(" ", "_")
    return ALIASES.get(n) or (n if n in KNOWN else None)


def parse_payload(body: dict) -> list[tuple]:
    """Both shapes phones send:
    - ours (the Shortcut): {"samples": [{"type", "start", "end", "value", "unit"?, "source"?}, ...]},
      sleep as type "sleep" with value = the stage ("Core", "Deep", "REM", "Awake", "In Bed");
    - Health Auto Export: {"data": {"metrics": [{"name", "units", "data": [{"date", "qty", ...}]}], "workouts": [...]}}.
    Unknown types are skipped, not stored."""
    out: list[tuple] = []
    for s in body.get("samples") or []:
        start, end = _iso(s.get("start") or s.get("date")), _iso(s.get("end") or s.get("start") or s.get("date"))
        if not start:
            continue
        kind, src = str(s.get("type", "")), str(s.get("source") or "shortcut")
        if kind.lower() in ("sleep", "sleep_analysis", "hkcategorytypeidentifiersleepanalysis"):
            stage = SLEEP.get(str(s.get("value", "")).strip()) or SLEEP.get(str(s.get("value", "")).strip().lower())
            if stage and end:
                minutes = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 60
                out.append((stage, start, end, round(minutes, 2), "min", src, None))
            continue
        metric = _metric(kind)
        try:
            value = float(s.get("value"))
        except (TypeError, ValueError):
            continue
        if metric:
            out.append((metric, start, end or start, value, s.get("unit"), src, None))
    data = body.get("data") or {}
    for m in data.get("metrics") or []:
        name = str(m.get("name", ""))
        if name == "sleep_analysis":
            for d in m.get("data") or []:  # one row a night, minutes or hours per stage
                start, end = _iso(d.get("sleepStart") or d.get("date")), _iso(d.get("sleepEnd") or d.get("date"))
                scale = 60 if (m.get("units") or "hr").startswith("hr") else 1
                for key, stage in (("core", "sleep_core"), ("deep", "sleep_deep"), ("rem", "sleep_rem"),
                                   ("awake", "sleep_awake"), ("inBed", "sleep_in_bed")):
                    if start and d.get(key) is not None:
                        out.append((stage, start, end or start, float(d[key]) * scale, "min", "auto-export", "night"))
            continue
        metric = _metric(name)
        if not metric:
            continue
        for d in m.get("data") or []:
            when = _iso(d.get("date"))
            value = d.get("qty", d.get("Avg"))
            if when and value is not None:
                out.append((metric, when, when, float(value), m.get("units"), "auto-export", None))
    for w in data.get("workouts") or []:
        start, end = _iso(w.get("start")), _iso(w.get("end"))
        if start and end:
            minutes = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 60
            out.append(("workout", start, end, round(minutes, 1), "min", "auto-export", str(w.get("name") or "")[:60]))
    return out


def summary() -> dict:
    """What's stored, per metric: how many samples and the date range (the health page and a sanity check)."""
    with _connect() as conn:
        rows = conn.execute("SELECT metric, COUNT(*), MIN(start), MAX(start) FROM samples GROUP BY metric "
                            "ORDER BY metric").fetchall()
    return {"metrics": [{"metric": m, "samples": n, "first": a, "last": b} for m, n, a, b in rows],
            "total": sum(r[1] for r in rows)}


def delete_all() -> int:
    with _connect() as conn:
        n = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        conn.execute("DELETE FROM samples")
    return n


SUM = {"steps", "active_energy", "exercise_minutes", "stand_minutes", "distance", "flights", "daylight_minutes"}
AVG = {"heart_rate", "resting_heart_rate", "walking_heart_rate", "hrv", "respiratory_rate", "blood_oxygen",
       "wrist_temperature"}
LAST = {"weight", "vo2max"}
STAGES = ("sleep_core", "sleep_deep", "sleep_rem", "sleep_awake")


def daily(days: int = 30, tz: str | None = None) -> list[dict]:
    """One row per local day, oldest first: sums (steps, energy...), averages (HRV, resting heart rate...),
    the latest weight / VO2 max, sleep by stage counted on the morning he woke up, and that day's workouts."""
    from collections import defaultdict
    from datetime import timedelta
    from zoneinfo import ZoneInfo
    zone = ZoneInfo(tz or os.getenv("THURSDAY_TZ", "America/New_York"))
    since = (datetime.now(timezone.utc) - timedelta(days=days + 1)).isoformat()
    with _connect() as conn:
        rows = conn.execute("SELECT metric, start, end, value, note FROM samples WHERE start >= ? ORDER BY start",
                            (since,)).fetchall()
    out: dict[str, dict] = defaultdict(lambda: {"workouts": []})
    avg: dict[tuple, list] = defaultdict(list)
    for metric, start, end, value, note in rows:
        when = end if metric.startswith("sleep") else start  # a night belongs to the morning he woke up
        day = datetime.fromisoformat(when).astimezone(zone).date().isoformat()
        d = out[day]
        if metric in SUM:
            d[metric] = d.get(metric, 0) + value
        elif metric in AVG:
            avg[(day, metric)].append(value)
        elif metric in LAST:
            d[metric] = value
        elif metric in STAGES:
            d[metric] = d.get(metric, 0) + value
        elif metric == "workout":
            d["workouts"].append({"type": note or "Workout", "minutes": value,
                                  "start": datetime.fromisoformat(start).astimezone(zone).strftime("%H:%M")})
    for (day, metric), vals in avg.items():
        out[day][metric] = sum(vals) / len(vals)
    today = datetime.now(zone).date()
    result = []
    for i in range(days - 1, -1, -1):
        day = (today - timedelta(days=i)).isoformat()
        d = {"date": day, **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in out.get(day, {"workouts": []}).items()}}
        asleep = sum(d.get(s, 0) for s in STAGES if s != "sleep_awake")
        if asleep:
            d["sleep_minutes"] = round(asleep)
        result.append(d)
    return result


# ---------------------------------------------------------------- the trainer
# Readiness is explainable on purpose: three of his own signals against his own baseline, each adding or taking
# points, so THURSDAY can say why ("HRV is 12% over your baseline, sleep was short"), not just a number.
SLEEP_GOAL_MIN = 7 * 60


def _avg(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def readiness(days: list[dict] | None = None) -> dict | None:
    """Today's readiness, 0-100, from HRV and resting heart rate against his 30-day baseline and last night's
    sleep. None until there's enough data (a few days of HRV or resting heart rate)."""
    days = days if days is not None else daily(31)
    if len(days) < 2:
        return None
    today, base = days[-1], days[:-1]
    latest = lambda k: today.get(k) if today.get(k) is not None else next((d[k] for d in reversed(base[-2:]) if d.get(k) is not None), None)
    hrv, hrv_base = latest("hrv"), _avg(base, "hrv")
    rhr, rhr_base = latest("resting_heart_rate"), _avg(base, "resting_heart_rate")
    sleep = latest("sleep_minutes")
    if (hrv is None or hrv_base is None) and (rhr is None or rhr_base is None):
        return None
    score, drivers = 70.0, []
    if hrv is not None and hrv_base:
        pct = (hrv - hrv_base) / hrv_base * 100
        pts = max(-15, min(15, pct * 0.75))
        score += pts
        drivers.append({"signal": "HRV", "today": round(hrv), "baseline": round(hrv_base), "unit": "ms",
                        "change_pct": round(pct), "points": round(pts)})
    if rhr is not None and rhr_base:
        diff = rhr - rhr_base
        pts = max(-12, min(8, -diff * 2))
        score += pts
        drivers.append({"signal": "resting heart rate", "today": round(rhr), "baseline": round(rhr_base), "unit": "bpm",
                        "change": round(diff), "points": round(pts)})
    if sleep is not None:
        pts = max(-12, min(8, (sleep - SLEEP_GOAL_MIN) / 60 * 6))
        score += pts
        drivers.append({"signal": "sleep", "last_night_min": round(sleep), "goal_min": SLEEP_GOAL_MIN, "points": round(pts)})
    score = round(max(0, min(100, score)))
    label = ("primed: a good day to push" if score >= 80 else "good: train as planned" if score >= 60
             else "so-so: keep it moderate" if score >= 40 else "low: rest or go easy")
    return {"score": score, "label": label, "drivers": drivers, "date": today["date"]}


def health_check(args: dict) -> "ToolResult":
    """The trainer's eyes: his numbers for a topic, against his own baseline, for THURSDAY to coach from."""
    from accounts import OWNER
    from calendar_tool import TURN
    from tools import ToolResult
    turn = TURN.get()
    if not turn or turn.get("user") != OWNER:
        return ToolResult(ok=False, tool_name="health_check", error="health_data_is_daniels_only")
    topic = str(args.get("topic") or "overview")
    span = max(1, min(int(args.get("days") or 7), 90))
    days = daily(max(31, span + 1))
    recent, before = days[-span:], days[-31:-span] or days[:-span]
    if not any(len(d) > 2 for d in days):
        return ToolResult(ok=False, tool_name="health_check",
                          error="no_health_data: Health Auto Export hasn't synced yet; ask him to open the app")
    pick = {"sleep": ("sleep_minutes", "sleep_deep", "sleep_rem", "sleep_core", "sleep_awake"),
            "heart": ("resting_heart_rate", "hrv", "walking_heart_rate", "heart_rate", "blood_oxygen", "respiratory_rate"),
            "activity": ("steps", "active_energy", "exercise_minutes", "stand_minutes", "distance", "flights"),
            "weight": ("weight",), "recovery": ("hrv", "resting_heart_rate", "sleep_minutes", "wrist_temperature")}
    keys = pick.get(topic) or tuple(k for ks in pick.values() for k in ks) + ("vo2max",)
    summary = {}
    for k in keys:
        now_avg, then_avg = _avg(recent, k), _avg(before, k)
        if now_avg is not None:
            summary[k] = {"avg": round(now_avg, 1), **({"baseline_avg": round(then_avg, 1)} if then_avg is not None else {})}
    per_day = [{"date": d["date"], **{k: d[k] for k in keys if d.get(k) is not None}} for d in recent]
    workouts = [{"date": d["date"], **w} for d in recent for w in d.get("workouts", [])]
    data = {"topic": topic, "days": span, "averages_vs_previous_weeks": summary, "per_day": per_day[-14:],
            "workouts": workouts, "readiness_today": readiness(days),
            "units": {"sleep_*": "minutes", "hrv": "ms", "resting_heart_rate": "bpm", "active_energy": "kcal",
                      "distance": "km", "weight": "kg (say pounds if he uses them: x 2.2046)"},
            "note": "You're his trainer and coach, not a doctor. Lead with what matters for today, compare with his own "
                    "baseline, then one concrete suggestion (train hard, keep it easy, sleep earlier...). Never "
                    "diagnose; if something looks off for days (resting HR up, HRV down, poor sleep), suggest rest "
                    "and, if it persists, checking with a doctor. Missing data: say what's missing, don't guess."}
    if topic == "weight" and "weight" not in summary:
        data["note"] += " No weight data yet: weight needs a smart scale or manual entries in Apple Health, " \
                        "and 'Weight & Body Mass' turned on in Health Auto Export."
    return ToolResult(ok=True, tool_name="health_check", data=data)


TOOL_SCHEMAS = [{"type": "function", "function": {
    "name": "health_check",
    "description": "His Apple Watch health data as his personal trainer: sleep, recovery and readiness, heart "
                   "(resting HR, HRV), activity, workouts, weight, compared with his own baseline. Use for 'how did "
                   "I sleep', 'am I recovered', 'should I train hard today', 'how active was I this week', 'my "
                   "resting heart rate', 'last workout'.",
    "parameters": {"type": "object", "properties": {
        "topic": {"type": "string", "enum": ["overview", "sleep", "recovery", "heart", "activity", "workouts", "weight"],
                  "description": "What he's asking about; 'recovery' covers readiness and whether to train"},
        "days": {"type": "integer", "description": "How far back: 1 for last night/today, 7 for this week, 30 for a month"}},
        "required": ["topic"]}}}]
