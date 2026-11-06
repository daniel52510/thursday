"""Imports his full Apple Health history from the Health app's "Export All Data" zip into brain/health.db, so
trends and baselines have years behind them from day one. The Health Auto Export app keeps it current after.

export.xml is often several GB: it's streamed (iterparse), never loaded whole. Heart rate is reduced to hourly
averages (the app sends hourly too); everything else is kept as recorded. Re-running is harmless: samples are
deduplicated.

    docker cp ~/Downloads/export.zip thursday-core:/tmp/export.zip
    docker exec thursday-core python scripts/import_health_export.py /tmp/export.zip
"""
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from xml.etree.ElementTree import iterparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import health  # noqa: E402

HOURLY = {"heart_rate"}  # thousands a day raw; an hourly average is what coaching needs
BATCH = 20_000


def _xml(path: str):
    if path.endswith(".zip"):
        z = zipfile.ZipFile(path)
        name = next(n for n in z.namelist() if n.endswith("export.xml") and "cda" not in n)
        return z.open(name)
    return open(path, "rb")


def main(path: str) -> None:
    rows, hourly, seen, new = [], defaultdict(list), 0, 0

    def flush() -> None:
        nonlocal rows, new
        new += health.store(rows)
        rows = []

    for _, el in iterparse(_xml(path), events=("end",)):
        tag = el.tag
        if tag == "Record":
            a = el.attrib
            kind, start, end = a.get("type", ""), health._iso(a.get("startDate")), health._iso(a.get("endDate"))
            src = (a.get("sourceName") or "health")[:40]
            if kind == "HKCategoryTypeIdentifierSleepAnalysis":
                stage = health.SLEEP.get(a.get("value", ""))
                if stage and start and end:
                    minutes = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 60
                    rows.append((stage, start, end, round(minutes, 2), "min", src, None))
            elif kind in health.METRICS and start:
                metric, unit = health.METRICS[kind]
                try:
                    value = float(a.get("value"))
                except (TypeError, ValueError):
                    value = None
                if value is not None:
                    if metric == "blood_oxygen" and value <= 1:
                        value *= 100  # stored as a fraction in the export
                    if metric in HOURLY:
                        hourly[(metric, start[:13], src)].append(value)
                    else:
                        rows.append((metric, start, end or start, value, a.get("unit") or unit, src, None))
            seen += 1
        elif tag == "Workout":
            a = el.attrib
            start, end = health._iso(a.get("startDate")), health._iso(a.get("endDate"))
            if start and end:
                minutes = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 60
                kind = a.get("workoutActivityType", "").removeprefix("HKWorkoutActivityType")
                rows.append(("workout", start, end, round(minutes, 1), "min", (a.get("sourceName") or "")[:40], kind))
        if tag in ("Record", "Workout"):
            el.clear()  # keep memory flat on a multi-GB file
            if len(rows) >= BATCH:
                flush()
                print(f"  {seen:,} records read, {new:,} new so far", flush=True)
    for (metric, hour, src), values in hourly.items():
        rows.append((metric, f"{hour}:00:00+00:00", f"{hour}:59:59+00:00", round(sum(values) / len(values), 1),
                     "bpm", src, f"hourly avg of {len(values)}"))
        if len(rows) >= BATCH:
            flush()
    flush()
    print(f"done: {seen:,} records read, {new:,} new samples stored")
    for m in health.summary()["metrics"]:
        print(f"  {m['metric']:<20} {m['samples']:>9,}  {m['first'][:10]} -> {m['last'][:10]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
