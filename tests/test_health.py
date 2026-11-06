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
