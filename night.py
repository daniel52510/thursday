"""Night mode: one answer to "is it night for him right now?", and the good-night / good-morning routines.

Night, first match wins:
  1. an override he set: "good night" (or his iPhone's Sleep Focus turning on: a Shortcuts automation POSTs
     /api/night) makes it night until morning; "good morning" (or the Focus turning off) makes it day until evening
  2. his Watch: inside the night hours, after 5 am, a sleep stage that ended in the last 90 minutes means he's up.
     (Health Auto Export syncs sleep after he wakes, so the Watch can say he woke, never that he fell asleep:
     that's what the Sleep Focus is for.)
  3. the clock: NIGHT_HOURS (THURSDAY_NIGHT, 22-7)

Everything asks is_night(): proactive delivery holds what can wait (presence.channel, via quiet()), his voice
whispers and keeps it short (mind.voice_style, agent_stream), the satellites play quietly with a soft ping instead of
a greeting and no thinking loop (broadcast on every change), and the web app turns to its night look (/api/night).

good_night(): night now, the lights go warm and dim and fade out over LIGHTS_FADE_MIN, music (if playing) fades out
over SLEEP_TIMER_MIN, and he hears his first alarm and first event tomorrow. good_morning(): day now, lights up warm
before 11, and what he needs for the day (calendar, weather, readiness, what was held overnight).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import struct
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

logger = logging.getLogger("thursday.voice.night")  # under voice_ws's logger: its handler reaches docker logs
FILE = os.path.join(os.getenv("THURSDAY_BRAIN_DIR", "./brain"), "night.json")
NIGHT_HOURS = tuple(int(h) for h in os.getenv("THURSDAY_NIGHT", "22-7").split("-"))
QUIET_HOURS = (22, 8)       # proactive holds what can wait this long (a bit past the night: he's just up)
LIGHTS_FADE_MIN = 10
SLEEP_TIMER_MIN = 20
SLEEP_TARGET_H = float(os.getenv("SLEEP_TARGET_H", "8"))
DEFAULT_WAKE = os.getenv("THURSDAY_WAKE", "07:00")
_lock = threading.Lock()
_last = {"night": None}      # what the satellites were last told
GOOD_NIGHT = re.compile(r"^\W*(?:(?:ok(?:ay)?|alright|well|hey thursday)\W+)?(?:good\s?night|night night|nighty night|"
                        r"i'?m (?:going|heading|off) to (?:bed|sleep)|(?:going|heading) to (?:bed|sleep)|bed ?time)\b",
                        re.I)
GOOD_MORNING = re.compile(r"^\W*(?:(?:ok(?:ay)?|hey thursday)\W+)?(?:good\s?morning|morning(?:,)? thursday|"
                          r"i'?m (?:up|awake))\b", re.I)


def _zone():
    from zoneinfo import ZoneInfo
    return ZoneInfo(os.getenv("THURSDAY_TZ", "America/New_York"))


def _now() -> datetime:
    return datetime.now(_zone())


def _in(hours: tuple[int, int], now: datetime) -> bool:
    start, end = hours
    return now.hour >= start or now.hour < end if start > end else start <= now.hour < end


def _load() -> dict:
    try:
        with open(FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _override(now: datetime) -> dict | None:
    o = _load()
    return o if o.get("mode") in ("night", "day") and o.get("until", 0) > now.timestamp() else None


def _woke(now: datetime) -> bool:
    """His Watch: a sleep stage ended in the last 90 minutes, and it's past 5 am."""
    if not 5 <= now.hour < 12:
        return False
    try:
        import health
        with health._connect() as conn:
            last = conn.execute("SELECT MAX(end) FROM samples WHERE metric LIKE 'sleep_%'").fetchone()[0]
        return bool(last) and timedelta(0) <= now.astimezone(timezone.utc) - datetime.fromisoformat(last) < timedelta(minutes=90)
    except Exception:
        return False


def state(now: datetime | None = None) -> dict:
    now = now or _now()
    if o := _override(now):
        # His Watch ends a "good night" too: a sleep stage that ended after he said it means he's up.
        if o["mode"] == "night" and _woke(now) and now.timestamp() - o.get("at", 0) > 3 * 3600:
            return {"night": False, "why": "watch: you're up", "until": None}
        return {"night": o["mode"] == "night", "why": o.get("by", "you"), "until": o["until"]}
    if _in(NIGHT_HOURS, now):
        if _woke(now):
            return {"night": False, "why": "watch: you're up", "until": None}
        return {"night": True, "why": "hours", "until": None}
    return {"night": False, "why": "hours", "until": None}


def is_night(now: datetime | None = None) -> bool:
    return state(now)["night"]


def quiet(now: datetime | None = None) -> bool:
    """When proactive delivery holds what can wait: an override decides; else the quiet hours."""
    now = now or _now()
    if o := _override(now):
        return o["mode"] == "night" and state(now)["night"]
    return _in(QUIET_HOURS, now) and not _woke(now)


def _until(mode: str, now: datetime) -> datetime:
    """Night lasts until 11 am (he says good morning before that); day lasts until tonight's night hours, or three
    hours if he's up in the middle of the night."""
    if mode == "night":
        end = now.replace(hour=11, minute=0, second=0, microsecond=0)
        return end if end > now + timedelta(hours=1) else end + timedelta(days=1)
    start = now.replace(hour=NIGHT_HOURS[0], minute=0, second=0, microsecond=0)
    if _in(NIGHT_HOURS, now) and now.hour >= NIGHT_HOURS[0]:
        return now + timedelta(hours=3)
    return start if start > now else start + timedelta(days=1)


def set_mode(mode: str, by: str) -> dict:
    """night / day (until the next change of the clock's), or auto (the clock and the Watch decide)."""
    now = _now()
    with _lock:
        if mode == "auto":
            data = {}
        else:
            data = {"mode": mode, "by": by, "at": now.timestamp(), "until": _until(mode, now).timestamp()}
        tmp = FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, FILE)
    logger.info("night mode: %s (%s)", mode, by)
    return state(now)


async def broadcast(force: bool = False) -> None:
    """Tell the satellites when night starts or ends: quieter, no greeting, no thinking loop."""
    import voice_ws
    on = is_night()
    if on == _last["night"] and not force:
        return
    _last["night"] = on
    for sat in list(voice_ws.SATELLITES):
        try:
            await sat.ws.send_json({"type": "night", "on": on})
        except Exception:
            pass


def message() -> dict:
    return {"type": "night", "on": is_night()}


# ---------- what the routines do ----------

def _lights_down() -> str:
    """Warm and dim at once, then a slow fade to off (LIFX fades on the bulb itself)."""
    import lights
    picked, err = lights._pick(None)
    if err or not picked:
        return "no lights reached"
    for st in picked:
        if st["power"] != "on":
            continue
        warm = struct.pack("<BHHHHI", 0, 0, 0, round(0.2 * 65535), 2200, 3000)
        lights._ask(st["ip"], lights._SET_COLOR, warm, want=lights._ACK)
        off = struct.pack("<HI", 0, LIGHTS_FADE_MIN * 60_000)
        lights._ask(st["ip"], lights._SET_POWER, off, want=lights._ACK)
    return f"dimmed warm, fading out over {LIGHTS_FADE_MIN} minutes"


def _lights_up() -> str:
    import lights
    r = lights.light_set({"power": "on", "brightness": 60, "color": "warm white"})
    return "on, warm" if r.ok else "couldn't reach them"


async def _music_sleep_timer() -> None:
    """Fades the music out over SLEEP_TIMER_MIN, pauses it, and puts the volume back for tomorrow."""
    import music
    st = await asyncio.to_thread(music.state, True)
    start = ((st or {}).get("player") or {}).get("volume")
    if start is None:
        return
    steps = 20
    for i in range(1, steps + 1):
        await asyncio.sleep(SLEEP_TIMER_MIN * 60 / steps)
        if not is_night():
            return  # he's up again: leave it
        await asyncio.to_thread(music._cmd, "setVolume", max(0, round(start * (1 - i / steps))))
    await asyncio.to_thread(music._cmd, "pause")
    await asyncio.sleep(2)
    await asyncio.to_thread(music._cmd, "setVolume", start)
    logger.info("sleep timer: music faded out and paused")


def _first_alarm(after: datetime) -> datetime | None:
    import alarms
    from accounts import OWNER
    from agent_loop import initalize_db
    best = None
    for a in initalize_db().list_alarms(OWNER):
        if not a["enabled"]:
            continue
        for d in (0, 1):
            at = alarms.scheduled_today(a, (after + timedelta(days=d)).astimezone(timezone.utc))
            if at and at > after.astimezone(timezone.utc) and (best is None or at < best):
                best = at
    return best.astimezone(_zone()) if best else None


def _first_event_at(day: datetime) -> tuple[str, datetime | None]:
    """The day's first timed event: (title, start), or ("", None)."""
    import calendar_tool
    try:
        r = calendar_tool.calendar_events({"start_date": day.date().isoformat(), "days": 1, "timezone": str(_zone())})
        ev = next((e for e in ((r.data or {}).get("events", []) if r.ok else []) if not e.get("all_day")), None)
        start = datetime.fromisoformat(str(ev["start"])) if ev else None
    except Exception:
        return "", None
    if not ev:
        return "", None
    return str(ev.get("title") or "an event"), start if start.tzinfo else start.replace(tzinfo=_zone())


def _first_event(day: datetime) -> str:
    title, start = _first_event_at(day)
    return f"{title} at {start.strftime('%-I:%M %p')}" if start else "nothing on the calendar"


ALARM_LEAD = timedelta(minutes=90)   # the bedtime check: up this long before the first event


def alarm_check(now: datetime, alarm: datetime | None, title: str, start: datetime | None) -> dict | None:
    """When tomorrow's first event is early and the alarm is missing or too late: the alarm he'd want."""
    if not start or start - now > timedelta(hours=20) or start <= now:
        return None
    if alarm and alarm <= start - timedelta(minutes=60):
        return None  # an hour or more of room already
    if not alarm and start.hour >= 12:
        return None  # nothing in the morning: sleep in
    want = start - ALARM_LEAD
    want = want.replace(minute=want.minute - want.minute % 5, second=0, microsecond=0)
    want = max(want, want.replace(hour=5, minute=0))
    if want <= now + timedelta(hours=3):
        return None  # too soon to be worth waking for
    have = f"your alarm is at {alarm.strftime('%-I:%M')}" if alarm else "you don't have an alarm set"
    return {"time": want.strftime("%H:%M"), "say": want.strftime("%-I:%M %p"), "tz": str(_zone()),
            "why": f"{title} at {start.strftime('%-I:%M %p')}",
            "question": f"Your first thing tomorrow is {title} at {start.strftime('%-I:%M')}, and {have}. "
                        f"Want an alarm at {want.strftime('%-I:%M')}?"}


def good_night() -> dict:
    now = _now()
    set_mode("night", "you said good night")
    done = {"night_mode": "on: notifications held till morning, quiet voice", "lights": _lights_down()}
    import music
    st = music.state() or {}
    if ((st.get("player") or {}).get("trackState")) == music.PLAYING:
        done["music"] = f"fading out over {SLEEP_TIMER_MIN} minutes"
        done["_sleep_timer"] = True
    tomorrow = now + timedelta(days=1) if now.hour >= 12 else now
    alarm = _first_alarm(now)
    done["first_alarm"] = alarm.strftime("%-I:%M %p") if alarm else "no alarm set"
    title, start = _first_event_at(tomorrow)
    done["tomorrow"] = f"{title} at {start.strftime('%-I:%M %p')}" if start else "nothing on the calendar"
    import alarms
    from accounts import OWNER
    if alarms.prefs(OWNER)["alarm_check"] and (check := alarm_check(now, alarm, title, start)):
        import asks
        if asks.record("alarm_check", check["question"], [{"key": "set", "label": "Set it"}, {"key": "no", "label": "No"}],
                       check, key=f"alarm_check:{tomorrow.date().isoformat()}"):
            done["ask_him"] = check["question"]
    if alarm:
        hours = (alarm - now).total_seconds() / 3600
        done["sleep_if_now"] = f"about {hours:.1f} hours before the alarm"
    return done


def good_morning() -> dict:
    now = _now()
    set_mode("day", "you said good morning")
    out: dict[str, Any] = {"night_mode": "off"}
    if now.hour < 11:
        out["lights"] = _lights_up()
    out["today"] = _first_event(now)
    try:
        import alarms
        import trainer
        from accounts import OWNER
        if alarms.prefs(OWNER)["alarm_training"] and (t := trainer.training_today({})).ok:
            out["training"] = {"session": t.data["session"], "intensity": t.data["intensity"],
                               "first_lift": (t.data.get("loads") or [{}])[0]}
    except Exception:
        pass
    try:  # where he is: what a satellite said on connect (as the alarm briefing does), else THURSDAY_LOCATION
        import tools
        import voice_ws
        where = next((s._context.get("location") for s in voice_ws.SATELLITES if s._context and s._context.get("location")),
                     "") or os.getenv("THURSDAY_LOCATION", "")
        if where and (w := tools.get_weather({"location": where, "days": 1})).ok:
            out["weather"] = w.data
    except Exception:
        pass
    try:
        import health
        if r := health.readiness():
            out["readiness"] = {"score": r.get("score"), "label": r.get("label")}
    except Exception:
        pass
    try:
        import proactive
        with proactive._db() as conn:
            out["held_overnight"] = [row[0] for row in conn.execute("SELECT title FROM proactive_held").fetchall()][:5]
    except Exception:
        pass
    return out


def night_mode(args: Dict[str, Any]):
    """The tool: good night, good morning, or back to automatic."""
    from accounts import OWNER
    from calendar_tool import TURN
    from tools import ToolResult
    turn = TURN.get()
    if not turn or turn.get("user") != OWNER:
        return ToolResult(ok=False, tool_name="night_mode", error="night_mode_is_daniels")
    action = str(args.get("action") or "").lower()
    if action == "good_night":
        data = good_night()
        note = ("Say good night in one or two quiet sentences: when his alarm is and the first thing tomorrow, and "
                "that the lights are fading. Nothing else." + (" Then ask ask_him, word for word; his yes or no is "
                                                               "handled for you." if data.get("ask_him") else ""))
    elif action == "good_morning":
        data = good_morning()
        note = ("A short, warm good morning: the first thing on today's calendar, the weather in a few words, and "
                "readiness only if it's notably low or high. Three sentences at most.")
    elif action == "auto":
        data, note = set_mode("auto", "you"), "Confirm in a few words: night mode follows the clock and his Watch again."
    else:
        return ToolResult(ok=False, tool_name="night_mode", error="action must be good_night, good_morning or auto")
    timer = data.pop("_sleep_timer", False)
    if timer:
        _spawn(_music_sleep_timer())
    _spawn(broadcast())
    return ToolResult(ok=True, tool_name="night_mode", data={**data, "note": note})


_LOOP: dict = {"loop": None}


def _spawn(coro) -> None:
    loop = _LOOP["loop"]
    if loop and loop.is_running():
        asyncio.run_coroutine_threadsafe(coro, loop)
    else:
        coro.close()


# ---------- the bedtime nudge, and the clock's own changes ----------

def bedtime(now: datetime) -> datetime:
    """When to be in bed for SLEEP_TARGET_H before tomorrow's first alarm (or THURSDAY_WAKE)."""
    alarm = _first_alarm(now.replace(hour=12, minute=0) if now.hour >= 12 else now)
    if not alarm or alarm - now > timedelta(hours=20):
        h, m = map(int, DEFAULT_WAKE.split(":"))
        alarm = (now + timedelta(days=1 if now.hour >= 12 else 0)).replace(hour=h, minute=m, second=0, microsecond=0)
    return alarm - timedelta(hours=SLEEP_TARGET_H)


async def run() -> None:
    """Every 30 s: tell the satellites when night starts or ends; and once an evening, nudge him toward bed."""
    import proactive
    _LOOP["loop"] = asyncio.get_running_loop()
    nudged = ""
    while True:
        try:
            await broadcast()
            now = _now()
            if 18 <= now.hour or now.hour < 2:
                bed = await asyncio.to_thread(bedtime, now)
                lead = (bed - now).total_seconds() / 60
                day = (now - timedelta(hours=6)).date().isoformat()
                if 0 < lead <= 30 and nudged != day and not (_override(now) or {}).get("mode") == "night":
                    nudged = day
                    line = (f"Bed by {bed.strftime('%-I:%M')} gets you your {SLEEP_TARGET_H:g} hours. "
                            "Say good night when you're heading in and I'll handle the lights.")
                    await proactive.deliver("Wind down", line, tag="crescent_moon", urgent=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("night")
        await asyncio.sleep(30)


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "night_mode",
        "description": "Night routines: good_night ('good night', 'I'm going to bed': lights fade out, music fades, "
                       "notifications held, tomorrow's alarm), good_morning ('good morning', 'I'm up': day mode, "
                       "lights on, today's rundown), auto (back to the clock).",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["good_night", "good_morning", "auto"]}}, "required": ["action"]}}},
]
