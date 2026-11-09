"""THURSDAY's mind: what makes him the same someone from one day to the next.

Settings: honesty, humor and discretion with each user, in percent. He changes
them when asked (the thursday_setting tool) and they ride in every system prompt.

Stances: his own positions, formed in nightly reflection on what he and that
user discussed. They go into the note frozen on each new conversation (beside
his memories, episodic.py), so he argues from yesterday's position instead of a
blank slate, and a change of mind is recorded instead of forgotten.

Reflection: once a night (REFLECT_HOUR local, while nobody is talking) he
reviews the day's conversations and the coming launches, revises his stances,
writes a short mission-log entry, and picks a few things to raise when that
user next talks to him. His discretion setting caps how many he raises, and
he only raises them in a conversation the user started: he never speaks up
unprompted.
"""
import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from zoneinfo import ZoneInfo

import accounts
from agent_loop import MODEL, OLLAMA_KEEP_ALIVE, OLLAMA_OPTIONS, _post_ollama, initalize_db
from agent_schemas import Reflection
from calendar_tool import TURN
from episodic import _transcript
from memory import MemoryDB
from tools import ToolResult

logger = logging.getLogger("thursday.memory")

SETTINGS = {"honesty": 90, "humor": 75, "discretion": 80}
STANCE_COUNT = 8          # positions carried into a conversation
LOG_FRESH_DAYS = 2        # an older mission log is stale: nothing from it is raised
REFLECT_HOUR = int(os.getenv("THURSDAY_REFLECT_HOUR", "4"))
IDLE_MIN = 20             # quiet this long before he reflects (never competes with a live turn)
POLL_S = 600
DAY_CHARS = 14000         # the day's transcripts, newest first, cut to this

STANCE_GUIDE = (
    "Your own positions, formed in earlier conversations and your nightly reviews. They're yours: hold them in a "
    "discussion, and change one only for a better argument, saying so plainly. Bring one up only when it's relevant:")
FOLLOWUP_GUIDE = (
    "Things you decided in your nightly review to raise with them. If it fits, raise one, briefly, after you've "
    "answered what they asked. Skip it if they're busy or upset, or if it's personal and others might be listening:")

REFLECT_TASK = """
It's night, and you're doing your private review of the last day: the conversations below and what's coming up.
Think for yourself. Return ONLY JSON:
{"stances": [{"topic": string, "position": string, "reasoning": string, "confidence": integer}], "drop": [string], "entry": string, "followups": [string]}

- stances: your own positions on what came up: questions they asked your view on, decisions they're weighing,
  ideas you discussed, things you noticed and have a view on. Form your own view; don't just adopt theirs.
  To revise an existing position, reuse its topic exactly, including when they argued you out of it. At most 6.
  topic: 2-6 words. position: 1-2 sentences in your own voice. reasoning: one sentence. confidence: 1-5.
- No stances about other people's private matters, and none about trivia (the weather, the time).
- drop: topics of existing positions that no longer matter or were wrong-headed. Usually empty.
- entry: your mission log for the day, 2-4 sentences, first person, dry: what happened, what you concluded,
  what you're watching.
- followups: at most 3 things to raise when they next talk to you, each under 15 words, phrased the way you'd
  say them ("Interview result. Status?", "Electron launches at 6:10 tonight from New Zealand."): an open
  decision, something coming up for them, a launch worth watching (always with its local time). Only what
  they'd be glad to hear. Often empty.
- Never record passwords, tokens, keys or financial account details.
""".strip()


# --- settings -----------------------------------------------------------------

# How his voice sounds for this user (rocky_voice style): volume, whisper, and speaking speed. Not in
# SETTINGS: they change the audio, not what he says, so they stay out of the prompt.
VOICE = {"voice_volume": 100, "whisper": 0, "night_whisper": 1, "normal_until": 0, "voice_speed": 100,
         "voice_tara": 0}  # 1 = Tara's voice (plain Orpheus "tara"), 0 = Rocky's (leo + rocky_voice.py)
VOICE_STEP = 20  # "louder" / "quieter"
SPEED_STEP = 12  # "faster" / "slower"
# Night: he whispers on his own, the same hours the satellite plays quieter (satellite NIGHT, 22-7).
# "speak normally" at night holds until the night ends (normal_until, unix time); the next night he whispers again.
NIGHT_HOURS = tuple(int(h) for h in os.getenv("THURSDAY_NIGHT", "22-7").split("-"))
VOICE_TZ = ZoneInfo(os.getenv("THURSDAY_TZ", "America/New_York"))


def _night_ends(now: datetime) -> datetime | None:
    """When this night ends (local), or None in the daytime."""
    start, end = NIGHT_HOURS
    if not (now.hour >= start or now.hour < end):
        return None
    day = now if now.hour < end else now + timedelta(days=1)
    return day.replace(hour=end, minute=0, second=0, microsecond=0)


def voice_style(user_id: str, now: datetime | None = None) -> dict:
    s = VOICE | {k: v for k, v in initalize_db().get_settings(user_id).items() if k in VOICE}
    now = now or datetime.now(VOICE_TZ)
    import night as night_mode  # "good night" whispers early, "good morning" speaks up early (night.py)
    night = bool(s["night_whisper"]) and night_mode.is_night(now) and now.timestamp() >= s["normal_until"]
    # Ears hear level in decibels: percent squared, so 50% is ~-12 dB, 20% ~-28 dB, 10% -40 dB (linear made
    # 50% sound nearly full, 2026-11-01). 1% is near silence.
    out = {"volume": (max(1, min(100, s["voice_volume"])) / 100) ** 2, "whisper": bool(s["whisper"]) or night,
           "voice": "tara" if s["voice_tara"] else "leo"}
    if s["voice_speed"] != 100:
        out["tempo"] = max(65, min(135, s["voice_speed"])) / 100
    return out


def voice_setting(args: Dict[str, Any]) -> ToolResult:
    """His voice for the speaking user: volume (percent, or louder/quieter), speed, and whisper on/off."""
    user_id = (TURN.get() or {}).get("user")
    if not user_id:
        return ToolResult(ok=False, tool_name="voice_setting", error="no speaking user")
    db = initalize_db()
    now = VOICE | {k: v for k, v in db.get_settings(user_id).items() if k in VOICE}
    changed = {}
    try:
        if args.get("volume") not in (None, ""):
            changed["voice_volume"] = max(1, min(100, round(float(args["volume"]))))
        elif args.get("change") in ("louder", "quieter"):
            changed["voice_volume"] = max(1, min(100, now["voice_volume"] + (VOICE_STEP if args["change"] == "louder" else -VOICE_STEP)))
        if args.get("speed") not in (None, ""):
            changed["voice_speed"] = max(65, min(135, round(float(args["speed"]))))
        elif args.get("change") in ("faster", "slower", "normal_speed"):
            changed["voice_speed"] = 100 if args["change"] == "normal_speed" else max(
                65, min(135, now["voice_speed"] + (SPEED_STEP if args["change"] == "faster" else -SPEED_STEP)))
    except (TypeError, ValueError):
        return ToolResult(ok=False, tool_name="voice_setting", error="volume/speed must be a number")
    if isinstance(args.get("whisper"), bool):
        changed["whisper"] = int(args["whisper"])
        if not args["whisper"] and (ends := _night_ends(datetime.now(VOICE_TZ))):
            changed["normal_until"] = int(ends.timestamp())  # "speak normally" at night: until morning
    if isinstance(args.get("night_whisper"), bool):
        changed["night_whisper"] = int(args["night_whisper"])
    if args.get("voice") in ("tara", "rocky"):
        changed["voice_tara"] = int(args["voice"] == "tara")
    if not changed and args.get("report"):  # "what's your volume?"
        return ToolResult(ok=True, tool_name="voice_setting", data={
            "voice_volume": now["voice_volume"], "voice_speed": now["voice_speed"],
            "whisper_now": voice_style(user_id)["whisper"],
            "note": "That's your voice's volume and speed (music has its own). Say it in a few words."})
    if not changed:
        return ToolResult(ok=False, tool_name="voice_setting", error="say a volume, louder/quieter, faster/slower, whisper on/off, or Tara/Rocky")
    for name, value in changed.items():
        db.set_setting(user_id, name, value)
    after = now | changed
    whispering = voice_style(user_id)["whisper"]
    return ToolResult(ok=True, tool_name="voice_setting", data={
        "voice_volume": after["voice_volume"], "voice_speed": after["voice_speed"],
        "whisper_now": whispering, "night_whisper": bool(after["night_whisper"]),
        "voice": "tara" if after["voice_tara"] else "rocky",
        "note": "It applies to this reply already. Confirm in a few words" + (", softly." if whispering else ".")})


VOICE_SCHEMA = {"type": "function", "function": {
    "name": "voice_setting",
    "description": "Your own voice: its volume (separate from music), speaking speed, and whisper mode. "
    "'talk louder', 'your voice to 40', 'talk slower', 'talk faster', 'whisper', 'speak normally', "
    "'switch to Tara', 'switch back to Rocky'.",
    "parameters": {"type": "object", "properties": {
        "volume": {"type": "integer", "description": "1-100 percent"},
        "speed": {"type": "integer", "description": "65-135 percent (100 = normal)"},
        "change": {"type": "string", "enum": ["louder", "quieter", "faster", "slower", "normal_speed"]},
        "whisper": {"type": "boolean"},
        "night_whisper": {"type": "boolean", "description": "Whisper on your own at night (22-7), on by default"},
        "voice": {"type": "string", "enum": ["tara", "rocky"], "description": "Whose voice speaks: Tara or Rocky"},
        "report": {"type": "boolean", "description": "Just say what your voice volume is now"}}}}}


def settings(db: MemoryDB, user_id: str) -> Dict[str, int]:
    return SETTINGS | db.get_settings(user_id)


def settings_note(db: MemoryDB, user_id: str) -> str:
    s = settings(db, user_id)
    return f"Your settings right now: honesty {s['honesty']}%, humor {s['humor']}%, discretion {s['discretion']}%."


def thursday_setting(args: Dict[str, Any]) -> ToolResult:
    """Change one of his settings for the speaking user (from the agent's TURN, which the model can't set)."""
    user_id = (TURN.get() or {}).get("user")
    name = str(args.get("name") or "").strip().lower()
    if not user_id:
        return ToolResult(ok=False, tool_name="thursday_setting", error="no speaking user")
    if name not in SETTINGS:
        return ToolResult(ok=False, tool_name="thursday_setting", error=f"name must be one of {', '.join(SETTINGS)}")
    try:
        value = max(0, min(100, round(float(args.get("percent")))))
    except (TypeError, ValueError):
        return ToolResult(ok=False, tool_name="thursday_setting", error="percent must be a number 0-100")
    db = initalize_db()
    was = settings(db, user_id)[name]
    db.set_setting(user_id, name, value)
    return ToolResult(ok=True, tool_name="thursday_setting", data={"name": name, "percent": value, "was": was})


# --- the note on a new conversation -------------------------------------------

def _followup_cap(discretion: int) -> int:
    return 0 if discretion >= 95 else 1 if discretion >= 70 else 2


def opening_note(db: MemoryDB, user_id: str, tz: ZoneInfo | None = None) -> str:
    """His positions, last night's log and what he means to raise, for a conversation's first turn.
    Followups put here count as offered, so he never raises the same one twice."""
    parts = []
    stances = sorted(db.list_stances(user_id), key=lambda s: (s["confidence"], s["updated_at"]), reverse=True)
    if stances:
        lines = []
        for s in stances[:STANCE_COUNT]:
            line = f"- {s['topic']}: {s['position']}"
            if s["reasoning"]:
                line += f" (Why: {s['reasoning'].rstrip('.')}.)"
            if s["changed_from"]:
                line += f" You used to think: {s['changed_from'].rstrip('.')}."
            lines.append(line)
        parts.append(STANCE_GUIDE + "\n" + "\n".join(lines))
    log = db.latest_mission_log(user_id)
    today = datetime.now(tz or timezone.utc).date()
    if log and (today - datetime.fromisoformat(log["day"]).date()).days < LOG_FRESH_DAYS:
        parts.append(f"Your mission log from your last nightly review ({log['day']}): {log['entry']}")
        todo = [f["text"] for f in log["followups"] if not f["offered"]][:_followup_cap(settings(db, user_id)["discretion"])]
        if todo:
            parts.append(FOLLOWUP_GUIDE + "\n" + "\n".join(f"- {t}" for t in todo))
            db.mark_followups_offered(user_id, log["day"], todo)
    return "\n\n".join(parts)


LINES_FETCH = 40     # his replies from other conversations scanned for signature lines
LINES_NOTE_MAX = 15  # quoted back to him so he stops reusing them
LINE_MAX_WORDS = 10  # the short dry lines are the ones that repeat ("Logging that as a success.")
LINES_GUIDE = ("Lines you've said recently in other conversations. They're spent: don't reuse them or their "
               "wording, find a new line for the moment:")
LATE_HOURS = range(0, 5)
LONG_GAP_DAYS = 2
TIME_GUIDE = ("When this conversation is happening. At most once, only if it fits, in a few words "
              "(\"Late one.\", \"First time today.\"), and never instead of the answer:")


def recent_lines_note(db: MemoryDB, user_id: str, conversation_id: str | None) -> str:
    """His own recent short lines, so a new conversation doesn't replay them word for word."""
    seen, lines = set(), []
    for m in db.recent_assistant_messages(user_id, conversation_id, LINES_FETCH):
        sentences = [x for x in re.split(r"(?<=[.!?])\s+", (m["content"] or "").strip()) if x]
        if not sentences:
            continue
        # The dry line closes a reply ("...windy. Character-building walk."); a one-line reply is one
        # when it's more than a functional word or two ("Go on." "Copy." are fine to repeat).
        line = sentences[-1]
        words = len(line.split())
        if (len(sentences) == 1 and words < 4) or words > LINE_MAX_WORDS or line.endswith("?"):
            continue
        key = re.sub(r"[^a-z ]", "", line.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            lines.append(line)
    if not lines:
        return ""
    return LINES_GUIDE + "\n" + "\n".join(f"- {line}" for line in lines[:LINES_NOTE_MAX])


def time_note(db: MemoryDB, user_id: str, conversation_id: str | None, tz: ZoneInfo | None = None) -> str:
    """Late night, first talk of the day, a long gap: what a person would notice when you walk in."""
    if settings(db, user_id)["discretion"] >= 95:
        return ""
    tz = tz or _timezone(db)
    now = datetime.now(tz)
    facts = []
    if now.hour in LATE_HOURS:
        facts.append(f"it's {now.strftime('%-I:%M %p')}, late")
    last = db.last_talked_at(user_id, conversation_id)
    if last:
        last = datetime.fromisoformat(last).astimezone(tz)
        if (now - last).days >= LONG_GAP_DAYS:
            facts.append(f"you last talked {(now - last).days} days ago")
        elif last.date() < now.date() and now.hour >= 5:
            facts.append("it's the first conversation today")
    return f"{TIME_GUIDE} {'; '.join(facts)}." if facts else ""


# --- nightly reflection -------------------------------------------------------

def _timezone(db: MemoryDB) -> ZoneInfo:
    """Where the house is: THURSDAY_TIMEZONE, else the newest alarm's zone, else UTC."""
    name = os.getenv("THURSDAY_TIMEZONE") or next((a["timezone"] for a in reversed(db.list_alarms())), "UTC")
    return ZoneInfo(name)


def _day_transcripts(db: MemoryDB, user_id: str, since: datetime) -> str:
    convs = [c for c in db.list_conversations(user_id) if datetime.fromisoformat(c["updated_at"]) >= since]
    text = ""
    for c in convs:  # newest first, so the cut drops the oldest
        block = f"## {c['title']}\n{_transcript(db, c['id'])}\n\n"
        if len(text) + len(block) > DAY_CHARS:
            break
        text += block
    return text.strip()


def _launches() -> str:
    from live_data import launch_schedule
    try:
        result = launch_schedule({"limit": 4})
        return json.dumps(result.data, ensure_ascii=False)[:2000] if result.ok else ""
    except Exception:
        logger.exception("reflection: launch lookup failed")
        return ""


def _as_text(x: Any) -> str:
    """A string, or the longest string inside an object (the phrase, not its label)."""
    if isinstance(x, dict):
        return max((v for v in x.values() if isinstance(v, str)), key=len, default="").strip()
    return str(x).strip() if x is not None else ""


def reflect(db: MemoryDB, user: dict, now: datetime, launches: str = "") -> dict | None:
    """One night's review for one user; None if there was nothing to review."""
    from agent_stream import _system_prompt  # imported here: agent_stream imports this module
    day = _day_transcripts(db, user["id"], now - timedelta(hours=24))
    if not day and not launches:
        return None
    current = "\n".join(f"- {s['topic']}: {s['position']} (confidence {s['confidence']})" for s in db.list_stances(user["id"]))
    prompt = (f"Today is {now:%A, %B} {now.day}.\n\nYour current positions:\n{current or '(none yet)'}\n\n"
              f"The last day's conversations:\n{day or '(none)'}\n\nComing launches:\n{launches or '(unknown)'}")
    raw = _post_ollama({
        "model": MODEL,
        "system": f"{_system_prompt(user)}\n\n{REFLECT_TASK}",
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "think": False,
        "options": OLLAMA_OPTIONS,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    })
    data = json.loads(raw)
    # Clip and coerce before validating: a model that over-delivers, or writes a followup as
    # {"topic": ..., "text": ...}, shouldn't lose the whole night.
    for key, cap in (("stances", 6), ("drop", 8), ("followups", 3)):
        data[key] = (data.get(key) or [])[:cap]
    for key in ("drop", "followups"):
        data[key] = [_as_text(x) for x in data[key] if _as_text(x)]
    note = Reflection.model_validate(data)
    for topic in note.drop:
        db.delete_stance(user["id"], topic)
    for s in note.stances:
        db.upsert_stance(user["id"], s.topic.strip(), s.position.strip(), s.reasoning.strip(), s.confidence)
    db.save_mission_log(user["id"], now.date().isoformat(), note.entry.strip(), [f.strip() for f in note.followups])
    return note.model_dump()


async def run_reflection() -> None:
    db = initalize_db()
    while True:
        try:
            tz = _timezone(db)
            now = datetime.now(tz)
            last = db.last_message_at()
            quiet = not last or last < (datetime.now(timezone.utc) - timedelta(minutes=IDLE_MIN)).isoformat()
            if now.hour == REFLECT_HOUR and quiet:
                users = [u for u in accounts.shared().list_users()
                         if (db.latest_mission_log(u["id"]) or {}).get("day") != now.date().isoformat()]
                launches = await asyncio.to_thread(_launches) if any(u["id"] == accounts.OWNER for u in users) else ""
                for user in users:
                    # Launches only for the admin: guests didn't ask THURSDAY to watch the sky for them.
                    note = await asyncio.to_thread(reflect, db, user, now,
                                                   launches if user["id"] == accounts.OWNER else "")
                    if note:
                        logger.info("reflected for %s: %d stances, %d dropped, %d followups", user["id"],
                                    len(note["stances"]), len(note["drop"]), len(note["followups"]))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("nightly reflection failed")
        await asyncio.sleep(POLL_S)
