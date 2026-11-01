"""Alarms she sets herself in conversation ("wake me at 7, I've got the flight").

They're "wake" alarms: at ring time she wakes you in her own voice after a soft
ping, remembering why you wanted to be up (alarms.py). Alarms ring on the
satellite, which is the admin's, so only the admin can set them. The speaking
user and their time zone come from the agent's TURN, which the model can't set.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict
from zoneinfo import ZoneInfo

import accounts
from agent_loop import initalize_db
from calendar_tool import TURN
from tools import ToolResult

SNOOZE_LOOKBACK = timedelta(minutes=30)  # how long after ringing "snooze" still means that alarm
DEFAULT_SNOOZE_MIN = 10
_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _clock(hour: int, minute: int) -> str:
    return datetime(2000, 1, 1, hour, minute).strftime("%I:%M %p").lstrip("0")


def _repeat(days: list[int]) -> str:
    if sorted(days) == [0, 1, 2, 3, 4]:
        return "weekdays"
    if sorted(days) == [5, 6]:
        return "weekends"
    if len(days) == 7:
        return "every day"
    return ", ".join(_DAY_NAMES[d] + "s" for d in sorted(days))


def _rings(alarm: dict, now: datetime) -> str:
    """When it rings next, the way she'd say it: "tomorrow at 7:00 AM", "weekdays at 6:30 AM"."""
    hour, minute = map(int, alarm["time"].split(":"))
    if alarm["days"]:
        return f"{_repeat(alarm['days'])} at {_clock(hour, minute)}"
    today = now.replace(hour=hour, minute=minute, second=0, microsecond=0) > now
    return f"{'today' if today else 'tomorrow'} at {_clock(hour, minute)}"


def _in_minutes(now: datetime, minutes: int) -> datetime:
    # Round up to the minute: "in 5 minutes" never rings in 4.
    at = now + timedelta(minutes=max(1, minutes))
    return at.replace(second=0, microsecond=0) + (timedelta(minutes=1) if at.second or at.microsecond else timedelta())


def _just_rang(user_id: str, now: datetime) -> dict | None:
    """The alarm that just woke them: the latest to ring within SNOOZE_LOOKBACK."""
    rang = [a for a in initalize_db().list_alarms(user_id) if a.get("last_fired_at")
            and now - datetime.fromisoformat(a["last_fired_at"]) < SNOOZE_LOOKBACK]
    return max(rang, key=lambda a: a["last_fired_at"], default=None)


def _clear_snoozes(user_id: str) -> int:
    """Delete their snooze alarms, pending or spent: there's only ever one wake-up in progress."""
    db = initalize_db()
    snoozes = [a for a in db.list_alarms(user_id) if a.get("style") == "snooze"]
    for a in snoozes:
        db.delete_alarm(a["id"], user_id)
    return sum(a["enabled"] for a in snoozes)


def _who(tool: str) -> tuple[dict | None, ZoneInfo | None, ToolResult | None]:
    turn = TURN.get() or {}
    user = accounts.shared().get_user(turn.get("user") or "")
    if not user or user["role"] != "admin":
        return None, None, ToolResult(ok=False, tool_name=tool, error="alarms_ring_on_daniels_satellite_only")
    if not turn.get("timezone"):
        return None, None, ToolResult(ok=False, tool_name=tool, error="unknown_timezone")
    return user, ZoneInfo(turn["timezone"]), None


def alarm_set(args: Dict[str, Any]) -> ToolResult:
    user, tz, err = _who("alarm_set")
    if err:
        return err
    now = datetime.now(tz)
    days = sorted({int(d) for d in args.get("days") or [] if 0 <= int(d) <= 6})
    if args.get("in_minutes") not in (None, ""):
        hhmm, days = f"{_in_minutes(now, int(args['in_minutes'])):%H:%M}", []
    else:
        try:
            hour, minute = map(int, str(args.get("time", "")).split(":"))
            assert 0 <= hour < 24 and 0 <= minute < 60
        except (ValueError, AssertionError):
            return ToolResult(ok=False, tool_name="alarm_set", error="time must be HH:MM (24h) or use in_minutes")
        hhmm = f"{hour:02d}:{minute:02d}"
    note = " ".join(str(args.get("note") or "").split())[:300]
    alarm = initalize_db().create_alarm(hhmm, days, tz.key, note, user_id=user["id"], style="wake")
    return ToolResult(ok=True, tool_name="alarm_set",
                      data={"alarm_id": alarm["id"], "rings": _rings(alarm, now), "note": note})


def alarm_list(args: Dict[str, Any]) -> ToolResult:
    user, tz, err = _who("alarm_list")
    if err:
        return err
    now = datetime.now(tz)
    alarms = [a for a in initalize_db().list_alarms(user["id"]) if a["enabled"]]
    return ToolResult(ok=True, tool_name="alarm_list", data={"alarms": [
        {"alarm_id": a["id"], "rings": _rings(a, now), "note": a["brief"], "kind": a.get("style", "briefing")}
        for a in alarms]})


def alarm_cancel(args: Dict[str, Any]) -> ToolResult:
    user, tz, err = _who("alarm_cancel")
    if err:
        return err
    alarm_id = str(args.get("alarm_id") or "")
    if not initalize_db().delete_alarm(alarm_id, user["id"]):
        return ToolResult(ok=False, tool_name="alarm_cancel", error="no such alarm; call alarm_list for ids")
    return ToolResult(ok=True, tool_name="alarm_cancel", data={"cancelled": alarm_id})


def alarm_snooze(args: Dict[str, Any]) -> ToolResult:
    """Ring the alarm that just woke them again in a few minutes, keeping why they wanted to be up."""
    user, tz, err = _who("alarm_snooze")
    if err:
        return err
    now = datetime.now(tz)
    rang = _just_rang(user["id"], now)
    if not rang:
        return ToolResult(ok=False, tool_name="alarm_snooze", error="no alarm rang recently; use alarm_set with in_minutes")
    minutes = min(60, max(1, int(args.get("minutes") or DEFAULT_SNOOZE_MIN)))
    import alarms  # here, not at the top: alarms imports the agent, which imports the tools
    alarms.end_wakeup(user["id"])  # no more nudges; the snooze wakes them next
    # A briefing alarm's brief is its topics, not a reason to be up.
    note = rang["brief"] if rang.get("style") in ("wake", "snooze") else ""
    _clear_snoozes(user["id"])
    alarm = initalize_db().create_alarm(f"{_in_minutes(now, minutes):%H:%M}", [], tz.key, note,
                                        user_id=user["id"], style="snooze")
    return ToolResult(ok=True, tool_name="alarm_snooze", data={"rings": _rings(alarm, now), "minutes": minutes})


def alarm_dismiss(args: Dict[str, Any]) -> ToolResult:
    """They're up: the wake-up is over, and any snooze still pending is cancelled."""
    user, tz, err = _who("alarm_dismiss")
    if err:
        return err
    import alarms
    return ToolResult(ok=True, tool_name="alarm_dismiss", data={"stopped_waking": alarms.end_wakeup(user["id"]),
                                                                "cancelled_snoozes": _clear_snoozes(user["id"])})
