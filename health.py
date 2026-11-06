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
