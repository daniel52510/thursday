"""Reminders that find him, and promises THURSDAY keeps.

Reminders: "remind me at 5 to call the bank" (reminder_set). When one is due, run() pushes it to his phone (ntfy,
which reaches him away from home over Tailscale) and, if a satellite is free and it's daytime, says it out loud.
Kept in brain/reminders.db, so a restart doesn't lose them.

Promises: when a reply says "I'll look into it" / "I'll research that" and no tool did anything, the sentence is
saved as an open promise (capture). Open promises go into his prompt (prompt_note), so THURSDAY does them or asks
whether he still wants them; promise_done closes one, and starting a mission/reminder/watch in that conversation
closes its open ones. The SpaceX research he agreed to on 2026-11-05 and never started is the case this covers.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

logger = logging.getLogger("thursday.voice.reminders")  # under voice_ws's logger: its handler reaches docker logs
DB = os.path.join(os.getenv("THURSDAY_BRAIN_DIR", "./brain"), "reminders.db")


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB) or ".", exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS reminders (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, text TEXT NOT NULL,
            due TEXT NOT NULL, created TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', conversation TEXT);
        CREATE TABLE IF NOT EXISTS promises (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, text TEXT NOT NULL,
            created TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open', conversation TEXT);""")
    return conn


def _zone():
    from zoneinfo import ZoneInfo
    return ZoneInfo(os.getenv("THURSDAY_TZ", "America/New_York"))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def parse_when(at: str | None, in_minutes: Any = None, now: datetime | None = None) -> datetime | None:
    """His time as UTC: minutes from now, "HH:MM" (today, or tomorrow if that's past), or "YYYY-MM-DD HH:MM"."""
    now = now or _now()
    if in_minutes not in (None, ""):
        try:
            return now + timedelta(minutes=float(in_minutes))
        except (TypeError, ValueError):
            return None
    s = str(at or "").strip().replace("T", " ")
    if not s:
        return None
    local_now = now.astimezone(_zone())
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s[:19] if fmt.endswith("%S") else s[:16], fmt).replace(tzinfo=_zone()).astimezone(timezone.utc)
        except ValueError:
            pass
    m = re.fullmatch(r"(\d{1,2}):(\d{2})\s*([ap]\.?m\.?)?", s, re.I)
    if m:
        h, mi = int(m[1]), int(m[2])
        if m[3]:
            h = h % 12 + (12 if m[3][0].lower() == "p" else 0)
        when = local_now.replace(hour=h, minute=mi, second=0, microsecond=0)
        if when <= local_now:
            when += timedelta(days=1)
        return when.astimezone(timezone.utc)
    return None


def _say_when(due: datetime) -> str:
    local, today = due.astimezone(_zone()), _now().astimezone(_zone()).date()
    day = "today" if local.date() == today else "tomorrow" if local.date() == today + timedelta(days=1) \
        else local.strftime("%A %b %-d")
    return f"{day} at {local.strftime('%-I:%M %p')}"


def _owner_turn(name: str):
    from accounts import OWNER
    from calendar_tool import TURN
    from tools import ToolResult
    turn = TURN.get()
    if not turn or turn.get("user") != OWNER:
        return None, ToolResult(ok=False, tool_name=name, error="reminders_are_daniels_only")
    return turn, None


def reminder_set(args: Dict[str, Any]):
    from tools import ToolResult
    turn, err = _owner_turn("reminder_set")
    if err:
        return err
    text = " ".join(str(args.get("text") or "").split())[:200]
    due = parse_when(args.get("at"), args.get("in_minutes"))
    if not text:
        return ToolResult(ok=False, tool_name="reminder_set", error="no_text: ask what to remind him about")
    if not due or due <= _now() - timedelta(minutes=1) or due > _now() + timedelta(days=366):
        return ToolResult(ok=False, tool_name="reminder_set",
                          error="bad_time: give at as 'HH:MM' or 'YYYY-MM-DD HH:MM' (his local time), or in_minutes")
    rid = uuid.uuid4().hex[:8]
    with _connect() as conn:
        conn.execute("INSERT INTO reminders(id, user_id, text, due, created, conversation) VALUES (?,?,?,?,?,?)",
                     (rid, turn["user"], text, due.isoformat(), _now().isoformat(), turn.get("conversation")))
    resolve_conversation(turn.get("conversation"))
    logger.info("reminder %s set for %s", rid, due.isoformat(timespec="minutes"))
    return ToolResult(ok=True, tool_name="reminder_set", data={
        "status": "set", "id": rid, "text": text, "when": _say_when(due),
        "note": "Confirm in a few words with the time. It reaches his phone by push, and the speaker if he's home."})


def reminder_list(args: Dict[str, Any]):
    from tools import ToolResult
    turn, err = _owner_turn("reminder_list")
    if err:
        return err
    with _connect() as conn:
        rows = conn.execute("SELECT id, text, due FROM reminders WHERE user_id = ? AND status = 'pending' "
                            "ORDER BY due LIMIT 20", (turn["user"],)).fetchall()
    return ToolResult(ok=True, tool_name="reminder_list", data={"reminders": [
        {"id": r["id"], "text": r["text"], "when": _say_when(datetime.fromisoformat(r["due"]))} for r in rows]})


def reminder_cancel(args: Dict[str, Any]):
    from tools import ToolResult
    turn, err = _owner_turn("reminder_cancel")
    if err:
        return err
    key = str(args.get("id_or_text") or "").strip().lower()
    with _connect() as conn:
        rows = conn.execute("SELECT id, text FROM reminders WHERE user_id = ? AND status = 'pending'",
                            (turn["user"],)).fetchall()
        hit = [r for r in rows if key and (r["id"] == key or key in r["text"].lower())]
        if len(hit) != 1:
            return ToolResult(ok=False, tool_name="reminder_cancel", error=(
                "no_match: call reminder_list and use its id" if not hit else "several_match: ask which one"))
        conn.execute("UPDATE reminders SET status = 'cancelled' WHERE id = ?", (hit[0]["id"],))
    return ToolResult(ok=True, tool_name="reminder_cancel", data={"cancelled": hit[0]["text"]})


# ---------------------------------------------------------------- promises
# "I'll look into it" said with no tool doing anything: a promise to keep. Rocky's grammar too ("I research it").
_VERBS = (r"(?:look into|research|find out|check (?:on|into|back)|get back to|follow up|dig into|keep an eye on|"
          r"look up|figure out|compare|investigate|work on)")
_PROMISE = re.compile(
    rf"[^.!?]*\b(?:(?:I'?ll|I will|I'm going to|I am going to|let me|I agreed to)\s+(?:\w+\s+){{0,2}}{_VERBS}"
    rf"|I {_VERBS})\b[^.!?]*[.!?]?", re.I)  # second form: Rocky's grammar ("I research it tonight")
# Tools that do the thing a promise says: running one in that conversation keeps its promises.
KEEPERS = {"mission_start", "reminder_set", "watch_start", "routine_start", "improve_start", "code_review_start"}


def capture(user_id: str, reply: str, tools_run: set[str], conversation: str | None) -> None:
    from accounts import OWNER
    if user_id != OWNER:
        return
    if tools_run & KEEPERS:  # he did something about it this turn: that conversation's promises are kept
        resolve_conversation(conversation)
        return
    m = _PROMISE.search(reply or "")
    if not m:
        return
    text = " ".join(m.group(0).split())[:240]
    with _connect() as conn:
        if conn.execute("SELECT 1 FROM promises WHERE user_id = ? AND status = 'open' AND text = ?",
                        (user_id, text)).fetchone():
            return
        conn.execute("INSERT INTO promises(id, user_id, text, created, conversation) VALUES (?,?,?,?,?)",
                     (uuid.uuid4().hex[:8], user_id, text, _now().isoformat(), conversation))
    logger.info("promise noted (%d chars)", len(text))


def resolve_conversation(conversation: str | None) -> None:
    if conversation:
        with _connect() as conn:
            conn.execute("UPDATE promises SET status = 'done' WHERE conversation = ? AND status = 'open'", (conversation,))


def promise_done(args: Dict[str, Any]):
    from tools import ToolResult
    turn, err = _owner_turn("promise_done")
    if err:
        return err
    with _connect() as conn:
        n = conn.execute("UPDATE promises SET status = ? WHERE id = ? AND user_id = ? AND status = 'open'",
                         ("dropped" if args.get("dropped") else "done", str(args.get("id") or ""), turn["user"])).rowcount
    return ToolResult(ok=bool(n), tool_name="promise_done", data={"closed": n},
                      **({} if n else {"error": "no_open_promise_with_that_id"}))


def prompt_note(user_id: str) -> str:
    """Open promises (last 14 days) for the end of his prompt, or ""."""
    since = (_now() - timedelta(days=14)).isoformat()
    try:
        with _connect() as conn:
            rows = conn.execute("SELECT id, text, created FROM promises WHERE user_id = ? AND status = 'open' AND "
                                "created >= ? ORDER BY created DESC LIMIT 5", (user_id, since)).fetchall()
    except sqlite3.Error:
        return ""
    if not rows:
        return ""
    items = "\n".join(f"- [{r['id']}] {r['text']} ({datetime.fromisoformat(r['created']).astimezone(_zone()).strftime('%b %-d')})"
                      for r in rows)
    return ("Things you told him you'd do and haven't yet. When it fits (or right away if he asks what's pending), "
            "do it with the right tool (mission_start for research, reminder_set...) after his yes, or ask if he "
            "still wants it; then call promise_done with its id (dropped=true if he doesn't). Never claim one is done "
            f"or underway unless a tool did it:\n{items}")


# ---------------------------------------------------------------- delivery
async def _deliver(r: sqlite3.Row) -> None:
    import approvals
    await asyncio.to_thread(approvals.push, "Reminder", r["text"], tags=("alarm_clock",), priority=4)
    try:
        import mind
        import voice_ws
        from accounts import OWNER
        hour = datetime.now(_zone()).hour
        idle = [s for s in voice_ws.SATELLITES if s.state == "idle"]
        if idle and 8 <= hour < 22:
            style = mind.voice_style(OWNER)
            line = f"Reminder: {r['text']}"
            if style.get("voice") != "tara":
                import eridian
                line = eridian.translate(line)
            import httpx
            async with httpx.AsyncClient() as http:
                audio = await voice_ws.synthesize_ahead(http, line, style)
            for sat in idle:
                if audio and await sat.announce(audio=audio, conversation_id=r["conversation"], cue="announce"):
                    break
    except Exception as exc:
        logger.warning("reminder not spoken: %s", type(exc).__name__)


async def run() -> None:
    """Every 15 s: due reminders go out (push, and out loud at home). Late ones still go (the Mac was asleep)."""
    while True:
        await asyncio.sleep(15)
        try:
            with _connect() as conn:
                due = conn.execute("SELECT * FROM reminders WHERE status = 'pending' AND due <= ? ORDER BY due",
                                   (_now().isoformat(),)).fetchall()
                for r in due:
                    conn.execute("UPDATE reminders SET status = 'sent' WHERE id = ?", (r["id"],))
            for r in due:
                logger.info("reminder %s due: delivering", r["id"])
                await _deliver(r)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("reminders")


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "reminder_set",
        "description": "Remind him later: a push to his phone (works away from home) and out loud on the speaker if "
                       "he's home. 'Remind me at 5 to call the bank', 'in 20 minutes check the oven', 'tomorrow at 9 "
                       "the dentist'. Give either at (his local time) or in_minutes.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "What to remind him, short ('Call the bank')"},
            "at": {"type": "string", "description": "His local time: 'HH:MM' (24 h; today, or tomorrow if past) or "
                                                   "'YYYY-MM-DD HH:MM'"},
            "in_minutes": {"type": "number", "description": "Instead of at: minutes from now"}},
            "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "reminder_list", "description": "His upcoming reminders.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "reminder_cancel", "description": "Cancel one of his reminders.",
        "parameters": {"type": "object", "properties": {
            "id_or_text": {"type": "string", "description": "Its id from reminder_list, or words from its text"}},
            "required": ["id_or_text"]}}},
    {"type": "function", "function": {
        "name": "promise_done",
        "description": "Close one of the open promises in your notes: you did it, or he doesn't want it (dropped).",
        "parameters": {"type": "object", "properties": {
            "id": {"type": "string"}, "dropped": {"type": "boolean"}}, "required": ["id"]}}},
]
