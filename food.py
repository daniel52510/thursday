"""His food journal: what he ate, its calories and macros, and what's left for the day.

Getting it in:
  - by voice or chat: "I had two eggs and toast with butter" (food_log). One LLM call splits it into items and
    estimates each (calories, protein, carbs, fat), shown as estimates; nothing is looked up online
  - the Food tab of the Health page: type it the same way, re-add a recent food in one tap (its numbers are kept,
    no LLM), or enter exact numbers off a label

Targets come from the trainer (trainer.targets: maintenance by Mifflin-St Jeor, minus ~600 kcal for the cut;
protein 1 g per lb of goal weight), unless he sets his own calories (brain/food.json). Fat is ~28 % of calories,
carbs the rest. Days are his local days.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime
from typing import Any, Dict

DB = os.path.join(os.getenv("THURSDAY_BRAIN_DIR", "./brain"), "food.db")
PREFS = os.path.join(os.getenv("THURSDAY_BRAIN_DIR", "./brain"), "food.json")
CUT_KCAL = 600            # under maintenance: the middle of the trainer's 500-750
FAT_SHARE = 0.28
MEALS = ("breakfast", "lunch", "dinner", "snack")


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB) or ".", exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS entries (id TEXT PRIMARY KEY, day TEXT NOT NULL, at REAL NOT NULL,
        meal TEXT, name TEXT NOT NULL, qty TEXT, kcal REAL NOT NULL, protein REAL NOT NULL, carbs REAL NOT NULL,
        fat REAL NOT NULL, source TEXT)""")
    conn.execute("CREATE INDEX IF NOT EXISTS entries_day ON entries (day)")
    return conn


def _now() -> datetime:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo(os.getenv("THURSDAY_TZ", "America/New_York")))


def _meal(now: datetime) -> str:
    return "breakfast" if now.hour < 11 else "lunch" if now.hour < 16 else "dinner" if now.hour < 21 else "snack"


def targets() -> dict:
    """{kcal, protein, carbs, fat, why}: his own calories if he set them, else the trainer's cut."""
    import trainer
    me = trainer.profile()
    t = trainer.targets(me)
    try:
        with open(PREFS) as f:
            own = int(json.load(f).get("kcal") or 0)
    except (OSError, ValueError):
        own = 0
    kcal = own or (round((t["maintenance_kcal"] - CUT_KCAL) / 50) * 50 if t.get("maintenance_kcal") else 2500)
    protein = int(t.get("protein_g_per_day") or round(kcal * 0.3 / 4))
    fat = round(kcal * FAT_SHARE / 9)
    carbs = max(0, round((kcal - protein * 4 - fat * 9) / 4))
    why = ("your own target" if own else
           f"maintenance about {t['maintenance_kcal']:.0f} minus {CUT_KCAL} for the cut" if t.get("maintenance_kcal")
           else "a default: add your weight, height and age in Settings → Memory for a real number")
    return {"kcal": kcal, "protein": protein, "carbs": carbs, "fat": fat, "why": why}


def set_target(kcal: int | None) -> dict:
    with open(PREFS, "w") as f:
        json.dump({"kcal": int(kcal) if kcal else 0}, f)
    return targets()


def estimate(text: str) -> list[dict]:
    """His words → [{name, qty, kcal, protein, carbs, fat}], one LLM call. Raises on no answer."""
    import httpx
    from agent_stream import CHAT_URL, MODEL, OLLAMA_KEEP_ALIVE, OLLAMA_OPTIONS
    prompt = (
        "Split what he ate into separate foods and estimate each one's nutrition for the amount he said (a typical "
        "US portion if he didn't say). Use standard nutrition data; restaurant or packaged items at their usual "
        "published values. Calories must roughly equal protein*4 + carbs*4 + fat*9.\n"
        f"He ate: {text}\n"
        'Answer only JSON: {"items": [{"name": "Scrambled eggs", "qty": "2 large", "kcal": 180, "protein": 12, '
        '"carbs": 2, "fat": 14}]} (grams for protein, carbs, fat; whole numbers).')
    r = httpx.post(CHAT_URL, timeout=60, json={
        "model": MODEL, "stream": False, "think": False, "format": "json", "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {**OLLAMA_OPTIONS, "temperature": 0.1}, "messages": [{"role": "user", "content": prompt}]})
    items = json.loads((r.json().get("message") or {}).get("content") or "{}").get("items") or []
    out = []
    for it in items[:15]:
        try:
            row = {"name": " ".join(str(it.get("name") or "").split())[:60], "qty": str(it.get("qty") or "")[:40],
                   **{k: max(0.0, min(5000.0, float(it.get(k) or 0))) for k in ("kcal", "protein", "carbs", "fat")}}
        except (TypeError, ValueError):
            continue
        if row["name"] and row["kcal"] > 0:
            out.append(row)
    return out


def add(items: list[dict], meal: str | None = None, source: str = "voice", day: str | None = None) -> list[dict]:
    now = _now()
    day, meal = day or now.date().isoformat(), meal if meal in MEALS else _meal(now)
    saved = []
    with _connect() as conn:
        for it in items:
            eid = uuid.uuid4().hex[:10]
            row = {"id": eid, "day": day, "at": time.time(), "meal": meal, "name": it["name"], "qty": it.get("qty", ""),
                   **{k: round(float(it.get(k) or 0), 1) for k in ("kcal", "protein", "carbs", "fat")}, "source": source}
            conn.execute("INSERT INTO entries VALUES (:id,:day,:at,:meal,:name,:qty,:kcal,:protein,:carbs,:fat,:source)", row)
            saved.append(row)
    return saved


def remove(entry_id: str) -> bool:
    with _connect() as conn:
        return conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,)).rowcount > 0


def readd(entry_id: str) -> list[dict]:
    """A recent food again, its numbers as they were (no estimate)."""
    with _connect() as conn:
        r = conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
    return add([dict(r)], source="again") if r else []


def day(day: str | None = None) -> dict:
    """The day: entries, totals, targets, and what's left."""
    d = day or _now().date().isoformat()
    with _connect() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM entries WHERE day = ? ORDER BY at", (d,))]
    totals = {k: round(sum(r[k] for r in rows)) for k in ("kcal", "protein", "carbs", "fat")}
    t = targets()
    return {"day": d, "entries": rows, "totals": totals, "targets": t,
            "left": {k: t[k] - totals[k] for k in ("kcal", "protein", "carbs", "fat")}}


def recent(limit: int = 12) -> list[dict]:
    """His last different foods, newest first, for one-tap re-adding."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM entries ORDER BY at DESC LIMIT 200").fetchall()
    seen, out = set(), []
    for r in rows:
        key = (r["name"].lower(), r["qty"])
        if key not in seen:
            seen.add(key)
            out.append(dict(r))
        if len(out) >= limit:
            break
    return out


def history(days: int = 14) -> list[dict]:
    with _connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT day, ROUND(SUM(kcal)) kcal, ROUND(SUM(protein)) protein, ROUND(SUM(carbs)) carbs, "
            "ROUND(SUM(fat)) fat FROM entries GROUP BY day ORDER BY day DESC LIMIT ?", (days,))]


# ---------- his tools ----------

def _owner_check(name: str):
    from accounts import OWNER
    from calendar_tool import TURN
    from tools import ToolResult
    turn = TURN.get()
    if not turn or turn.get("user") != OWNER:
        return ToolResult(ok=False, tool_name=name, error="food_journal_is_daniels_only")
    return None


def _left_line(d: dict) -> dict:
    l = d["left"]
    return {"calories_left": l["kcal"], "protein_left_g": l["protein"], "carbs_left_g": l["carbs"],
            "fat_left_g": l["fat"], "eaten_kcal": d["totals"]["kcal"], "target_kcal": d["targets"]["kcal"]}


def food_log(args: Dict[str, Any]):
    from tools import ToolResult
    if err := _owner_check("food_log"):
        return err
    text = " ".join(str(args.get("description") or "").split())[:400]
    if not text:
        return ToolResult(ok=False, tool_name="food_log", error="ask what he ate")
    try:
        items = estimate(text)
    except Exception as exc:
        return ToolResult(ok=False, tool_name="food_log", error=f"couldn't estimate it ({type(exc).__name__})")
    if not items:
        return ToolResult(ok=False, tool_name="food_log", error="couldn't tell what food that was: ask him")
    saved = add(items, str(args.get("meal") or "").lower() or None)
    return ToolResult(ok=True, tool_name="food_log", data={
        "logged": [f"{s['name']} ({s['qty']}): {s['kcal']:.0f} kcal, {s['protein']:.0f} g protein" for s in saved],
        **_left_line(day()),
        "note": "Say the total for this in a few words and what's left today (calories, and protein if it's "
                "behind). They're estimates; no lecture."})


def food_today(args: Dict[str, Any]):
    from tools import ToolResult
    if err := _owner_check("food_today"):
        return err
    d = day()
    return ToolResult(ok=True, tool_name="food_today", data={
        **_left_line(d), "foods": [f"{e['meal']}: {e['name']} {e['kcal']:.0f} kcal" for e in d["entries"]][-12:],
        "why_target": d["targets"]["why"],
        "note": "Lead with calories left and protein left; one suggestion if protein is far behind."})


def food_undo(args: Dict[str, Any]):
    """Removes the most recent entry, or the latest one matching a name ("take off the toast")."""
    from tools import ToolResult
    if err := _owner_check("food_undo"):
        return err
    want = str(args.get("name") or "").lower().strip()
    rows = [e for e in reversed(day()["entries"]) if not want or want in e["name"].lower()]
    if not rows or not remove(rows[0]["id"]):
        return ToolResult(ok=False, tool_name="food_undo", error="nothing like that logged today")
    return ToolResult(ok=True, tool_name="food_undo", data={"removed": rows[0]["name"], **_left_line(day())})


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "food_log",
        "description": "Log what he ate or drank into his food journal ('I had two eggs and toast', 'a chicken "
                       "burrito for lunch', 'protein shake'). It estimates calories and macros and says what's left.",
        "parameters": {"type": "object", "properties": {
            "description": {"type": "string", "description": "What he ate, in his words, with amounts if he said them"},
            "meal": {"type": "string", "enum": list(MEALS)}}, "required": ["description"]}}},
    {"type": "function", "function": {
        "name": "food_today", "description": "His calories and macros so far today and what's left ('how many "
                                             "calories do I have left?', 'how's my protein?').",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "food_undo", "description": "Take a food back off today's journal ('remove the toast', 'undo that').",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}}}},
]


# ---------- what the trainer makes of it ----------

FULL_DAY_KCAL = 1000      # a day with less logged is a day he didn't log everything: left out of averages


def _weights(days: int = 28) -> list[tuple[str, float]]:
    """His weigh-ins (Apple Health via health.py), in pounds, oldest first."""
    try:
        import health
        rows = [(r["date"], float(r["weight"])) for r in health.daily(days) if r.get("weight")]
    except Exception:
        return []
    import trainer
    profile_lb = trainer.profile().get("weight_lb") or 0
    # Health exports kilograms on some setups: a reading under half his profile weight is kg.
    return [(d, w * 2.20462 if profile_lb and w < profile_lb * 0.6 else w) for d, w in rows]


def coach(days: int = 14) -> dict:
    """His last two weeks of eating against the plan, his weight trend, and the trainer's call."""
    t = targets()
    full = [h for h in history(days) if h["kcal"] >= FULL_DAY_KCAL]
    out: dict[str, Any] = {"target_kcal": t["kcal"], "target_protein_g": t["protein"], "days_logged": len(full)}
    if full:
        out["avg_kcal"] = round(sum(h["kcal"] for h in full) / len(full))
        out["avg_protein_g"] = round(sum(h["protein"] for h in full) / len(full))
    w = _weights(28)
    if len(w) >= 2:
        span = (datetime.fromisoformat(w[-1][0]) - datetime.fromisoformat(w[0][0])).days
        if span >= 10:
            out["lb_per_week"] = round((w[-1][1] - w[0][1]) / span * 7, 1)
            out["weight_lb"] = round(w[-1][1], 1)
    advice, change = [], None
    if len(full) < 5:
        advice.append("Not enough logged days yet to judge the plan: log most meals for a week.")
    else:
        over = out["avg_kcal"] - t["kcal"]
        if over > 250:
            advice.append(f"Eating about {over} kcal over the target on average: the plan is fine, the gap is intake.")
        if out["avg_protein_g"] < t["protein"] * 0.85:
            advice.append(f"Protein averages {out['avg_protein_g']} g against {t['protein']} g: that's the lever "
                          "that keeps the muscle on a cut.")
        rate = out.get("lb_per_week")
        if rate is not None and len(full) >= 10 and abs(over) <= 150:
            if rate > -0.5:
                change = t["kcal"] - 200
                advice.append(f"On target for two weeks and the scale is flat ({rate:+} lb a week): time to drop to "
                              f"{change} kcal.")
            elif rate < -2.0:
                change = t["kcal"] + 200
                advice.append(f"Losing {abs(rate)} lb a week is faster than muscle likes: up to {change} kcal.")
    out["advice"], out["suggested_kcal"] = advice, change
    return out


def trainer_view() -> dict:
    """For training_today: today's fuel so far and the last two weeks."""
    d = day()
    return {"today": {"eaten_kcal": d["totals"]["kcal"], "kcal_left": d["left"]["kcal"],
                      "protein_g": d["totals"]["protein"], "protein_left_g": d["left"]["protein"],
                      "logged_anything": bool(d["entries"])},
            "last_two_weeks": coach()}
