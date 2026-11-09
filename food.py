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
from datetime import datetime, timedelta
from typing import Any, Dict

DB = os.path.join(os.getenv("THURSDAY_BRAIN_DIR", "./brain"), "food.db")
PREFS = os.path.join(os.getenv("THURSDAY_BRAIN_DIR", "./brain"), "food.json")
# The daily calculation (targets): see plan_today.
PACE = ((30, 1.5), (15, 1.0), (3, 0.5))   # lb from goal → lb a week to lose; closer than 3 lb: maintain
MAX_DEFICIT = 0.25        # of maintenance, at most
MIN_KCAL = 2000           # never below, whatever the math says (6'5", lifting)
WATCH_DISCOUNT = 0.75     # watches overstate active calories by ~25-30 %
DAY_SHIFT = {"lift": 150, "cardio": 0, "rest": -150}   # carbs follow the training: a week nets about zero
FAT_MIN_SHARE, FAT_MIN_G_PER_LB = 0.25, 0.3
MEALS = ("breakfast", "lunch", "dinner", "snack")


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB) or ".", exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS entries (id TEXT PRIMARY KEY, day TEXT NOT NULL, at REAL NOT NULL,
        meal TEXT, name TEXT NOT NULL, qty TEXT, kcal REAL NOT NULL, protein REAL NOT NULL, carbs REAL NOT NULL,
        fat REAL NOT NULL, source TEXT)""")
    conn.execute("CREATE INDEX IF NOT EXISTS entries_day ON entries (day)")
    conn.execute("CREATE TABLE IF NOT EXISTS weights (day TEXT PRIMARY KEY, lb REAL NOT NULL, at REAL NOT NULL)")
    return conn


def _now() -> datetime:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo(os.getenv("THURSDAY_TZ", "America/New_York")))


def _meal(now: datetime) -> str:
    return "breakfast" if now.hour < 11 else "lunch" if now.hour < 16 else "dinner" if now.hour < 21 else "snack"


def _own_kcal() -> int:
    try:
        with open(PREFS) as f:
            return int(json.load(f).get("kcal") or 0)
    except (OSError, ValueError):
        return 0


def log_weight(lb: float, day: str | None = None) -> dict:
    """A weigh-in he tells THURSDAY or types (the Watch/scale ones come from Apple Health)."""
    day = day or _now().date().isoformat()
    with _connect() as conn:
        conn.execute("INSERT OR REPLACE INTO weights VALUES (?, ?, ?)", (day, round(float(lb), 1), time.time()))
    return {"day": day, "lb": round(float(lb), 1), "now": current_weight()}


def weigh_ins(days: int = 42) -> list[tuple[str, float]]:
    """Every weigh-in, his and Apple Health's, one per day (his wins), in pounds, oldest first."""
    since = (_now().date() - timedelta(days=days)).isoformat()
    out: dict[str, float] = {}
    try:
        import health
        import trainer
        profile_lb = trainer.profile().get("weight_lb") or 0
        for r in health.daily(days):
            if r.get("weight"):
                w = float(r["weight"])  # Health exports kilograms on some setups: under 60 % of him is kg
                out[r["date"]] = w * 2.20462 if profile_lb and w < profile_lb * 0.6 else w
    except Exception:
        pass
    with _connect() as conn:
        for r in conn.execute("SELECT day, lb FROM weights WHERE day >= ?", (since,)):
            out[r["day"]] = r["lb"]
    return sorted(out.items())


def current_weight() -> float | None:
    """The average of his last week of weigh-ins (one reading swings 2-4 lb with water); else his profile weight."""
    import trainer
    w = weigh_ins(14)
    if w:
        last = datetime.fromisoformat(w[-1][0])
        recent = [lb for d, lb in w if (last - datetime.fromisoformat(d)).days < 7]
        return round(sum(recent) / len(recent), 1)
    return trainer.profile().get("weight_lb")


def _day_kind(day: datetime) -> str:
    """From the session's own title: conditioning and easy cardio days, the rest day, and lifting otherwise
    (push, pull, legs, the Bond circuit)."""
    import trainer
    title = trainer.WEEK[day.weekday()][0].lower()
    if "rest" in title:
        return "rest"
    return "cardio" if re.search(r"conditioning|cardio|zone 2", title) else "lift"


def _active_kcal() -> float | None:
    """His Watch's active calories, the average of the last 7 full days, discounted."""
    try:
        import health
        days = [r["active_energy"] for r in health.daily(8)[:-1] if r.get("active_energy")]
    except Exception:
        return None
    return round(sum(days) / len(days) * WATCH_DISCOUNT) if len(days) >= 3 else None


def _adaptive_tdee() -> tuple[float | None, int]:
    """What he actually burns, from his own numbers: average intake minus the energy of his weight change, over
    the last 3 weeks of fully logged days and weigh-ins. (estimate, days of data)."""
    since = (_now().date() - timedelta(days=21)).isoformat()
    full = [h for h in history(21) if h["kcal"] >= FULL_DAY_KCAL and h["day"] >= since]
    w = [x for x in weigh_ins(21)]
    if len(full) < 10 or len(w) < 4:
        return None, len(full)
    span = (datetime.fromisoformat(w[-1][0]) - datetime.fromisoformat(w[0][0])).days
    if span < 10:
        return None, len(full)
    k = max(1, len(w) // 3)  # thirds of the readings, averaged, so one salty day doesn't swing it
    start, end = sum(lb for _, lb in w[:k]) / k, sum(lb for _, lb in w[-k:]) / k
    when = lambda part: sum(datetime.fromisoformat(d).toordinal() for d, _ in part) / len(part)
    per_day = (end - start) / max(1.0, when(w[-k:]) - when(w[:k]))  # between the two groups' middles, not the ends
    intake = sum(h["kcal"] for h in full) / len(full)
    return round(intake - per_day * 3500), len(full)


def plan_today(day: datetime | None = None) -> dict:
    """Today's calories and macros, worked out fresh each day:
    maintenance = BMR (Mifflin-St Jeor, at his current weight) x 1.2 + his Watch's active calories (discounted),
    blended toward what his own intake and weigh-ins show he burns as that data builds up; minus a deficit set by
    how far he is from his goal (1.5 / 1 / 0.5 lb a week, never over 25 % of maintenance, never under 2,000);
    plus or minus for the day's training. Protein holds at his goal weight in grams; fat at least 25 % of calories;
    carbs take the rest and move with training."""
    import re
    import trainer
    day = day or _now()
    me = trainer.profile()
    t = trainer.targets(me)
    weight = current_weight() or 0
    goal = float(t.get("goal_weight_lb") or weight or 0)
    try:
        ft, inch = (int(x) for x in re.findall(r"\d+", str(me["height"]))[:2])
        age = int(str(me["age"]).split()[0])
        bmr = 10 * weight * 0.4536 + 6.25 * (ft * 12 + inch) * 2.54 - 5 * age + 5
    except (KeyError, TypeError, ValueError):
        bmr = None
    why: list[str] = []
    if not bmr or not weight:
        kcal, tdee = 2500, None
        why.append("A default: add your weight, height and age in Settings → Memory for real numbers.")
    else:
        active = _active_kcal()
        formula = bmr * 1.2 + active if active is not None else bmr * 1.55
        why.append(f"BMR {bmr:,.0f} at {weight:g} lb" + (f", plus {active:,.0f} active from your Watch" if active is not None
                                                          else ", times 1.55 for training (no Watch activity yet)"))
        adaptive, n = _adaptive_tdee()
        if adaptive:
            trust = min(1.0, n / 28)
            tdee = formula * (1 - trust) + adaptive * trust
            why.append(f"your logs and weigh-ins say you burn about {adaptive:,.0f}; weighted {trust:.0%} after {n} days")
        else:
            tdee = formula
        to_go = weight - goal
        pace = next((p for gap, p in PACE if to_go > gap), 0.0)
        deficit = min(pace * 500, tdee * MAX_DEFICIT)
        kcal = max(MIN_KCAL, tdee - deficit)
        why.append(f"maintenance {tdee:,.0f}; {to_go:.0f} lb from {goal:.0f}, so losing {deficit / 500:.1f} lb a week "
                   f"(−{deficit:,.0f})" if pace else f"maintenance {tdee:,.0f}; at your goal, so no deficit")
    kind = _day_kind(day)
    kcal += DAY_SHIFT[kind]
    if DAY_SHIFT[kind]:
        why.append(f"{'+' if DAY_SHIFT[kind] > 0 else '−'}{abs(DAY_SHIFT[kind])} for a {kind} day")
    own = _own_kcal()
    if own:
        kcal = own + DAY_SHIFT[kind]
        why = [f"your own target {own:,}" + (f", {'+' if DAY_SHIFT[kind] > 0 else '−'}{abs(DAY_SHIFT[kind])} for a "
                                              f"{kind} day" if DAY_SHIFT[kind] else "")]
    kcal = int(round(kcal / 25) * 25)
    protein = int(round(goal or kcal * 0.3 / 4))
    fat = int(round(max(kcal * FAT_MIN_SHARE / 9, FAT_MIN_G_PER_LB * (goal or 0))))
    carbs = max(0, int(round((kcal - protein * 4 - fat * 9) / 4)))
    return {"kcal": kcal, "protein": protein, "carbs": carbs, "fat": fat, "day_kind": kind,
            "maintenance": round(tdee) if tdee else None, "weight_lb": weight or None, "goal_lb": goal or None,
            "why": "; ".join(why) + "."}


def targets() -> dict:
    """Today's numbers (plan_today): what the Food tab, his voice and the trainer all use."""
    return plan_today()


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


def weight_log(args: Dict[str, Any]):
    from tools import ToolResult
    if err := _owner_check("weight_log"):
        return err
    try:
        lb = float(args.get("pounds"))
        assert 80 <= lb <= 700
    except (TypeError, ValueError, AssertionError):
        return ToolResult(ok=False, tool_name="weight_log", error="say the weight in pounds")
    out = log_weight(lb)
    t = targets()
    return ToolResult(ok=True, tool_name="weight_log", data={
        "logged_lb": out["lb"], "seven_day_average_lb": out["now"], "today_kcal": t["kcal"], "protein_g": t["protein"],
        "note": "Confirm in a few words; mention the 7-day average if it's moving, not the single reading."})


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "weight_log", "description": "Log his weigh-in ('I weigh 296', 'scale says 294.5 this morning').",
        "parameters": {"type": "object", "properties": {"pounds": {"type": "number"}}, "required": ["pounds"]}}},
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


def coach(days: int = 14) -> dict:
    """His last two weeks against the plan: averages, weight trend, and what to work on."""
    t = targets()
    full = [h for h in history(days) if h["kcal"] >= FULL_DAY_KCAL]
    out: dict[str, Any] = {"today_kcal": t["kcal"], "today_protein_g": t["protein"], "days_logged": len(full),
                           "maintenance": t["maintenance"], "how_today_is_worked_out": t["why"]}
    if full:
        out["avg_kcal"] = round(sum(h["kcal"] for h in full) / len(full))
        out["avg_protein_g"] = round(sum(h["protein"] for h in full) / len(full))
    w = weigh_ins(28)
    if len(w) >= 2:
        span = (datetime.fromisoformat(w[-1][0]) - datetime.fromisoformat(w[0][0])).days
        if span >= 7:
            out["lb_per_week"] = round((w[-1][1] - w[0][1]) / span * 7, 1)
    out["weight_lb"] = current_weight()
    advice = []
    if not w:
        advice.append("No weigh-ins yet: weigh in a few mornings a week so the numbers can follow you.")
    if len(full) < 5:
        advice.append("Log most meals for a week so the plan can learn what you really burn.")
    elif out["avg_protein_g"] < t["protein"] * 0.85:
        advice.append(f"Protein averages {out['avg_protein_g']} g against {t['protein']} g: that's what keeps the "
                      "muscle on a cut.")
    rate = out.get("lb_per_week")
    if rate is not None and rate < -2.5:
        advice.append(f"Losing {abs(rate)} lb a week: faster than the plan; eat to the target, not under it.")
    out["advice"] = advice
    return out


def trainer_view() -> dict:
    """For training_today: today's fuel so far and the last two weeks."""
    d = day()
    return {"today": {"eaten_kcal": d["totals"]["kcal"], "kcal_left": d["left"]["kcal"],
                      "protein_g": d["totals"]["protein"], "protein_left_g": d["left"]["protein"],
                      "logged_anything": bool(d["entries"])},
            "last_two_weeks": coach()}
