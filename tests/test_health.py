"""Apple Health ingest (health.py): both payload shapes land as clean, deduplicated samples."""
import pytest

import health


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "DB", str(tmp_path / "health.db"))


AUTO_EXPORT = {"data": {
    "metrics": [
        {"name": "heart_rate_variability", "units": "ms", "data": [{"date": "2026-11-06 06:58:00 -0500", "qty": 48.2}]},
        {"name": "resting_heart_rate", "units": "count/min", "data": [{"date": "2026-11-06 00:00:00 -0500", "qty": 54}]},
        {"name": "step_count", "units": "count", "data": [{"date": "2026-11-06 09:00:00 -0500", "qty": 1200}]},
        {"name": "sleep_analysis", "units": "hr", "data": [{"date": "2026-11-06 00:00:00 -0500",
            "sleepStart": "2026-11-05 23:40:00 -0500", "sleepEnd": "2026-11-06 06:50:00 -0500",
            "core": 3.9, "deep": 1.1, "rem": 1.6, "awake": 0.4, "inBed": 7.2}]},
        {"name": "something_new", "units": "x", "data": [{"date": "2026-11-06 09:00:00 -0500", "qty": 1}]},
    ],
    "workouts": [{"name": "Outdoor Run", "start": "2026-11-05 17:00:00 -0500", "end": "2026-11-05 17:32:00 -0500"}],
}}


def test_auto_export_payload():
    rows = health.parse_payload(AUTO_EXPORT)
    got = {r[0]: r for r in rows}
    assert got["hrv"][3] == 48.2 and got["resting_heart_rate"][3] == 54 and got["steps"][3] == 1200
    assert got["sleep_deep"][3] == pytest.approx(66) and got["sleep_rem"][3] == pytest.approx(96)  # hours -> minutes
    assert got["workout"][3] == 32 and got["workout"][6] == "Outdoor Run"
    assert "something_new" not in got                       # unknown metrics are skipped
    assert got["hrv"][1] == "2026-11-06T11:58:00+00:00"    # stored in UTC


def test_store_is_idempotent_and_summarized():
    rows = health.parse_payload(AUTO_EXPORT)
    assert health.store(rows) == len(rows)
    assert health.store(rows) == 0                          # the app re-sends overlapping windows: harmless
    s = health.summary()
    assert s["total"] == len(rows) and {m["metric"] for m in s["metrics"]} >= {"hrv", "sleep_deep", "workout"}
    assert health.delete_all() == len(rows) and health.summary()["total"] == 0


def test_shortcut_payload_with_sleep_stages():
    rows = health.parse_payload({"samples": [
        {"type": "HKQuantityTypeIdentifierHeartRate", "start": "2026-11-06T07:00:00-05:00", "value": 61},
        {"type": "sleep", "value": "Deep", "start": "2026-11-06T02:00:00-05:00", "end": "2026-11-06T02:45:00-05:00"},
        {"type": "steps", "start": "bad date", "value": 5},
    ]})
    assert [(r[0], r[3]) for r in rows] == [("heart_rate", 61.0), ("sleep_deep", 45.0)]


def test_import_apple_export(tmp_path):
    import importlib.util, zipfile
    xml = """<?xml version="1.0"?><HealthData>
 <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Watch" unit="count/min" startDate="2025-03-01 08:05:00 -0500" endDate="2025-03-01 08:05:00 -0500" value="60"/>
 <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Watch" unit="count/min" startDate="2025-03-01 08:35:00 -0500" endDate="2025-03-01 08:35:00 -0500" value="80"/>
 <Record type="HKQuantityTypeIdentifierOxygenSaturation" sourceName="Watch" unit="%" startDate="2025-03-01 03:00:00 -0500" endDate="2025-03-01 03:00:00 -0500" value="0.97"/>
 <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Watch" startDate="2025-03-01 02:00:00 -0500" endDate="2025-03-01 02:30:00 -0500" value="HKCategoryValueSleepAnalysisAsleepDeep"/>
 <Workout workoutActivityType="HKWorkoutActivityTypeRunning" sourceName="Watch" startDate="2025-03-01 17:00:00 -0500" endDate="2025-03-01 17:40:00 -0500"/>
</HealthData>"""
    z = tmp_path / "export.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("apple_health_export/export.xml", xml)
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "imp", Path(__file__).resolve().parent.parent / "scripts" / "import_health_export.py")
    imp = importlib.util.module_from_spec(spec); spec.loader.exec_module(imp)
    imp.main(str(z))
    got = {m["metric"]: m["samples"] for m in health.summary()["metrics"]}
    assert got == {"heart_rate": 1, "blood_oxygen": 1, "sleep_deep": 1, "workout": 1}  # 2 HR readings -> 1 hourly avg
    import sqlite3
    c = sqlite3.connect(health.DB)
    assert c.execute("SELECT value FROM samples WHERE metric='heart_rate'").fetchone()[0] == 70
    assert c.execute("SELECT value FROM samples WHERE metric='blood_oxygen'").fetchone()[0] == 97
    assert c.execute("SELECT note FROM samples WHERE metric='workout'").fetchone()[0] == "Running"


def test_daily_rollup():
    health.store(health.parse_payload(AUTO_EXPORT))
    rows = health.daily(3650, tz="America/New_York")
    by = {r["date"]: r for r in rows}
    d = by["2026-11-06"]
    assert d["steps"] == 1200 and d["hrv"] == 48.2 and d["resting_heart_rate"] == 54
    assert d["sleep_deep"] == 66 and d["sleep_minutes"] == round((3.9 + 1.1 + 1.6) * 60)  # awake not counted
    assert by["2026-11-05"]["workouts"][0]["type"] == "Outdoor Run"


def _day(date, **kw):
    return {"date": date, "workouts": [], **kw}


def test_readiness_explains_itself():
    base = [_day(f"2026-10-{d:02d}", hrv=60, resting_heart_rate=55, sleep_minutes=420) for d in range(1, 30)]
    good = health.readiness(base + [_day("2026-10-30", hrv=72, resting_heart_rate=52, sleep_minutes=480)])
    bad = health.readiness(base + [_day("2026-10-30", hrv=45, resting_heart_rate=62, sleep_minutes=330)])
    assert good["score"] >= 80 and "push" in good["label"]
    assert bad["score"] < 40 and "rest" in bad["label"]
    assert {d["signal"] for d in bad["drivers"]} == {"HRV", "resting heart rate", "sleep"}
    assert all(d["points"] < 0 for d in bad["drivers"])
    assert health.readiness([_day("2026-10-30")]) is None  # not enough data


def test_health_check_is_his_only(monkeypatch):
    from calendar_tool import TURN
    TURN.set({"user": "guest", "conversation": "c", "turn": 1, "user_said": "how did I sleep"})
    assert health.health_check({"topic": "sleep"}).error == "health_data_is_daniels_only"
    TURN.set({"user": "owner", "conversation": "c", "turn": 1, "user_said": "how did I sleep"})
    assert "no_health_data" in health.health_check({"topic": "sleep"}).error  # empty test db
    health.store(health.parse_payload(AUTO_EXPORT))
    monkeypatch.setattr(health, "daily", lambda days, tz=None: [_day("2026-11-05", hrv=50, sleep_minutes=400),
                                                                _day("2026-11-06", hrv=48.2, sleep_minutes=396, sleep_deep=66)])
    r = health.health_check({"topic": "sleep", "days": 1})
    assert r.ok and r.data["averages_vs_previous_weeks"]["sleep_minutes"]["avg"] == 396
