"""Where he is and what he's doing, so THURSDAY knows how to reach him (proactive.deliver) and what to say.

Signals, each cheap and already there:
  - phone: home / away / offline / unknown, from the Mac's helper (scripts/host_mem.sh pings his iPhone over
    Tailscale once a minute: a home-network path, or the Mac's own public address, means home)
  - asleep: last night's sleep from his Watch hasn't ended yet, or it's the middle of the night
  - in a meeting: a timed calendar event is happening now
  - room: a satellite is free to speak in
A stale reading (the helper stopped) counts as unknown, never as away.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

FILE = Path(os.getenv("PRESENCE_FILE", "/hostmem/presence.json"))
STALE_S = 300


def _zone():
    from zoneinfo import ZoneInfo
    return ZoneInfo(os.getenv("THURSDAY_TZ", "America/New_York"))


def phone() -> str:
    try:
        d = json.loads(FILE.read_text())
    except (OSError, ValueError):
        return "unknown"
    if time.time() - d.get("at", 0) >= STALE_S:
        return "unknown"
    recent = d.get("recent") or [d.get("phone", "unknown")] * 3  # a file from before the history: one reading
    last3 = recent[-3:]
    # A home-network path can't happen away, but a waking phone at home sometimes takes the internet route for a
    # minute (seen 2026-11-07): home if any of the last three said so, away only when all three agree.
    if "home" in last3:
        return "home"
    if len(last3) == 3 and all(r == "away" for r in last3):
        return "away"
    return "offline" if recent[-1] == "offline" else "unknown"


def asleep(now: datetime | None = None) -> bool:
    """Night hours, unless his Watch says he's already up (a sleep stage ended within the last hour, then stopped)."""
    now = now or datetime.now(_zone())
    if not (now.hour >= 23 or now.hour < 6):
        return False
    try:
        import health
        with health._connect() as conn:
            last = conn.execute("SELECT MAX(end) FROM samples WHERE metric LIKE 'sleep_%'").fetchone()[0]
        if last:
            ended = datetime.fromisoformat(last)
            if timedelta(0) < now.astimezone(timezone.utc) - ended < timedelta(hours=1) and now.hour >= 5:
                return False  # woke up early: the night's sleep just ended
    except Exception:
        pass
    return True


def in_meeting(events: list[dict] | None = None, now: datetime | None = None) -> str | None:
    """The title of a timed event happening now, or None."""
    now = now or datetime.now(_zone())
    if events is None:
        try:
            import calendar_tool
            r = calendar_tool.calendar_events({"start_date": now.date().isoformat(), "days": 1, "timezone": str(_zone())})
            events = (r.data or {}).get("events", []) if r.ok else []
        except Exception:
            events = []
    for ev in events:
        if ev.get("all_day"):
            continue
        try:
            start, end = datetime.fromisoformat(str(ev["start"])), datetime.fromisoformat(str(ev.get("end") or ev["start"]))
        except (KeyError, ValueError):
            continue
        start, end = (d if d.tzinfo else d.replace(tzinfo=_zone()) for d in (start, end))
        if start <= now < end:
            return str(ev.get("title") or "an event")
    return None


def free_satellite() -> bool:
    try:
        import voice_ws
        return any(s.state == "idle" for s in voice_ws.SATELLITES)
    except Exception:
        return False


def now_state() -> dict:
    t = datetime.now(_zone())
    return {"phone": phone(), "asleep": asleep(t), "meeting": in_meeting(now=t), "satellite": free_satellite(),
            "hour": t.hour}


def channel(state: dict, urgent: bool = False) -> str:
    """How to reach him: speak (home, awake, a satellite free), push (away, in a meeting, or no satellite),
    or hold (asleep or night, unless urgent)."""
    night = state["hour"] >= 22 or state["hour"] < 8
    if (state["asleep"] or night) and not urgent:
        return "hold"
    if state["meeting"]:
        return "push"
    if state["phone"] == "home" and state["satellite"]:
        return "speak"
    return "push"


def prompt_line(state: dict) -> str:
    """One line of context for his prompt."""
    where = {"home": "He's at home.", "away": "He's out (not home).", "offline": "His phone is offline.",
             "unknown": ""}.get(state["phone"], "")
    meeting = f" He's in '{state['meeting']}' right now." if state.get("meeting") else ""
    return (where + meeting).strip()
