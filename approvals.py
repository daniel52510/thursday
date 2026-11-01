"""Approvals: actions that leave the house happen only with his OK.

A mission can't change anything outside: it proposes (propose_event), and the proposal waits
here, in the DB, for him. He answers on the Agentic page (Approve / Not now) or out loud. Out
loud, approval_decide only goes ahead when his message this turn is a plain yes, checked here
in code (calendar_tool.plain_yes), never left to the model. Proposals expire after EXPIRE.

Kinds: calendar_create, saved through the same code as the voice calendar flow
(calendar_tool.prepare_event / save_event), and reminder, a phone push ntfy holds until its time
(up to DELAY_MAX ahead). Messages and calls plug in later.

His phone (ntfy, NTFY_TOPIC) lives here too: each new proposal is pushed, and tapping it opens
the Agentic page, where he decides logged in; the push carries no way to approve on its own.
Finished missions and send_to_phone use the same push().
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

import calendar_tool
from accounts import OWNER
from agent_loop import initalize_db
from calendar_tool import TURN
from memory import now_iso
from tools import ToolResult

EXPIRE = timedelta(hours=48)
MAX_PER_MISSION = 5

logger = logging.getLogger("thursday.voice")

NTFY_URL = os.getenv("NTFY_URL", "").rstrip("/")
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "")
PUBLIC_URL = os.getenv("THURSDAY_PUBLIC_URL", "http://thursday.local").rstrip("/")
DELAY_MAX = timedelta(days=3)  # ntfy's default message-delay-limit


def link(conversation_id: str | None = None) -> str:
    """Where a tap on his phone lands: a conversation, or the Agentic page."""
    return f"{PUBLIC_URL}/#c={conversation_id}" if conversation_id else f"{PUBLIC_URL}/#agentic"


def push(title: str, message: str, *, click: str | None = None, tags: tuple[str, ...] = (), priority: int = 3,
         at: datetime | None = None) -> bool:
    """One notification to his phone; at = deliver later. Best effort: a down ntfy never stops what sent it."""
    if not (NTFY_URL and NTFY_TOPIC):
        return False
    # The topic goes in the JSON body, not the URL: httpx logs URLs, and the topic is the only lock.
    body: Dict[str, Any] = {"topic": NTFY_TOPIC, "title": title, "message": message[:4000], "tags": list(tags),
                            "priority": priority, "click": click or link()}
    if at:
        body["delay"] = str(int(at.timestamp()))
    try:
        httpx.post(NTFY_URL, timeout=3, json=body).raise_for_status()
        return True
    except httpx.HTTPError as exc:
        logger.info("phone push failed (%s): %s", title, type(exc).__name__)
        return False


def _push(approval: dict) -> None:
    """Tell his phone something waits for his OK."""
    body = approval["summary"] + (f"\n{approval['why']}" if approval.get("why") else "")
    tag = "alarm_clock" if approval["kind"] == "reminder" else "calendar"
    if push("THURSDAY needs your OK", body, tags=(tag,), priority=4):
        logger.info("approval %s pushed to phone", approval["id"])


def pending(user_id: str, mission_id: str | None = None) -> list[dict]:
    """What's waiting for his OK; proposals past EXPIRE are marked expired on the way."""
    db, now, out = initalize_db(), datetime.now(timezone.utc), []
    for a in db.list_approvals(user_id, "pending", mission_id):
        if datetime.fromisoformat(a["created_at"]) + EXPIRE < now:
            db.update_approval(a["id"], status="expired", decided_at=now_iso())
        else:
            out.append(a)
    return out


def decide(approval_id: str, user_id: str, approve: bool) -> dict | None:
    """Carry out (or decline) one pending proposal. None if there's no such pending one."""
    db = initalize_db()
    if not any(a["id"] == approval_id for a in pending(user_id)):
        return None
    if not approve:
        db.update_approval(approval_id, status="declined", decided_at=now_iso())
    else:
        approval = db.get_approval(approval_id, user_id)
        if approval["kind"] == "reminder":
            at = datetime.fromisoformat(approval["args"]["at_utc"])
            if at <= datetime.now(timezone.utc):
                ok, result = False, "its time already passed"
            else:
                ok = push("Reminder", approval["args"]["text"], tags=("alarm_clock",), priority=4, at=at)
                result = "scheduled" if ok else "phone unreachable"
            db.update_approval(approval_id, status="approved" if ok else "failed", result=result, decided_at=now_iso())
            logger.info("approval %s reminder: %s", approval_id, result)
            return db.get_approval(approval_id, user_id)
        token = TURN.set({"user": user_id})  # whose calendar: the owner's, whatever thread this runs on
        try:
            cal, event, _ = calendar_tool.prepare_event(approval["args"])
            calendar_tool.save_event(cal, event)
            db.update_approval(approval_id, status="approved", result="saved", decided_at=now_iso())
            logger.info("approval %s carried out: %s", approval_id, approval["summary"])
        except Exception as exc:  # iCloud down, calendar gone: say so, don't pretend
            db.update_approval(approval_id, status="failed", result=f"{type(exc).__name__}: {exc}"[:200],
                               decided_at=now_iso())
            logger.info("approval %s failed: %s", approval_id, type(exc).__name__)
        finally:
            TURN.reset(token)
    return db.get_approval(approval_id, user_id)


# --- the mission's tool -------------------------------------------------------------

def propose_event(args: Dict[str, Any]) -> ToolResult:
    turn = TURN.get() or {}
    if not str(turn.get("turn") or "").startswith("mission-"):
        return ToolResult(ok=False, tool_name="propose_event", error="only_missions_propose")
    user, mission_id = turn.get("user") or OWNER, turn["turn"].removeprefix("mission-")
    if len(pending(user, mission_id)) >= MAX_PER_MISSION:
        return ToolResult(ok=False, tool_name="propose_event", error=f"already_{MAX_PER_MISSION}_proposals: report now")
    event = {k: args.get(k) for k in ("title", "start", "duration_minutes", "location", "calendar")}
    event["timezone"] = turn.get("timezone")
    try:
        _, _, read_back = calendar_tool.prepare_event(event)  # checks it the way the save will
    except (ValueError, ZoneInfoNotFoundError, LookupError) as exc:
        return ToolResult(ok=False, tool_name="propose_event", error=f"bad_input: {exc}")
    approval = initalize_db().create_approval(user, mission_id, "calendar_create", event, read_back,
                                              str(args.get("why") or "").strip()[:300])
    _push(approval)
    return ToolResult(ok=True, tool_name="propose_event", data={
        "approval_id": approval["id"], "waiting_for_his_ok": read_back,
        "note": "Nothing is on his calendar yet; it's saved only if he approves. Say so in your report."})


def propose_reminder(args: Dict[str, Any]) -> ToolResult:
    turn = TURN.get() or {}
    if not str(turn.get("turn") or "").startswith("mission-"):
        return ToolResult(ok=False, tool_name="propose_reminder", error="only_missions_propose")
    user, mission_id = turn.get("user") or OWNER, turn["turn"].removeprefix("mission-")
    if len(pending(user, mission_id)) >= MAX_PER_MISSION:
        return ToolResult(ok=False, tool_name="propose_reminder", error=f"already_{MAX_PER_MISSION}_proposals: report now")
    text = " ".join(str(args.get("text") or "").split())[:200]
    try:
        tz = ZoneInfo(turn.get("timezone") or "America/New_York")
        at = datetime.fromisoformat(str(args.get("at") or "")).replace(tzinfo=tz)
    except (ValueError, ZoneInfoNotFoundError):
        return ToolResult(ok=False, tool_name="propose_reminder", error="bad_input: at must be YYYY-MM-DDTHH:MM local")
    now = datetime.now(timezone.utc)
    if not text or at <= now:
        return ToolResult(ok=False, tool_name="propose_reminder", error="bad_input: needs text and a time in the future")
    if at - now > DELAY_MAX:
        return ToolResult(ok=False, tool_name="propose_reminder",
                          error="too_far: reminders reach 3 days ahead; propose_event for anything later")
    read_back = f"remind you on your phone: '{text}', {at:%A} at {f'{at:%I:%M %p}'.lstrip('0')}"
    approval = initalize_db().create_approval(user, mission_id, "reminder",
                                              {"text": text, "at_utc": at.astimezone(timezone.utc).isoformat()},
                                              read_back, str(args.get("why") or "").strip()[:300])
    _push(approval)
    return ToolResult(ok=True, tool_name="propose_reminder", data={
        "approval_id": approval["id"], "waiting_for_his_ok": read_back,
        "note": "Nothing is set yet; it reaches his phone only if he approves. Say so in your report."})


REMINDER_SCHEMA = {"type": "function", "function": {
    "name": "propose_reminder",
    "description": "Propose a reminder on his phone at a set time, within the next 3 days (buy tickets when sales open, "
    "leave for the airport). Nothing is set: it waits for his OK.",
    "parameters": {"type": "object", "properties": {
        "text": {"type": "string", "description": "What to remind him, short"},
        "at": {"type": "string", "description": "YYYY-MM-DDTHH:MM local time"},
        "why": {"type": "string", "description": "One line: why then"}},
        "required": ["text", "at"]}}}


PROPOSE_SCHEMA = {"type": "function", "function": {
    "name": "propose_event",
    "description": "Propose an event for his calendar. Nothing is saved: it waits for his OK. Use it when the mission "
    "asks you to schedule, book or block time.",
    "parameters": {"type": "object", "properties": {
        "title": {"type": "string"},
        "start": {"type": "string", "description": "YYYY-MM-DDTHH:MM local time, or YYYY-MM-DD for all day"},
        "duration_minutes": {"type": "integer", "description": "Default 60"},
        "location": {"type": "string"},
        "why": {"type": "string", "description": "One line: why this slot"}},
        "required": ["title", "start"]}}}


# --- the voice model's tools ---------------------------------------------------------------

def _brief(a: dict) -> dict:
    return {"approval_id": a["id"], "what": a["summary"], "why": a["why"], "proposed": a["created_at"][:16]}


def approval_list(args: Dict[str, Any]) -> ToolResult:
    if (TURN.get() or {}).get("user") != OWNER:
        return ToolResult(ok=False, tool_name="approval_list", error="approvals_are_daniels_only")
    return ToolResult(ok=True, tool_name="approval_list", data={"waiting_for_his_ok": [_brief(a) for a in pending(OWNER)]})


def approval_decide(args: Dict[str, Any]) -> ToolResult:
    turn = TURN.get() or {}
    if turn.get("user") != OWNER:
        return ToolResult(ok=False, tool_name="approval_decide", error="approvals_are_daniels_only")
    approve = args.get("approve") is True
    waiting = pending(OWNER)
    if approve and not calendar_tool.plain_yes(turn.get("user_said") or ""):
        # The model can't approve on its own: only his plain yes, this turn, does.
        return ToolResult(ok=False, tool_name="approval_decide",
                          error="not_confirmed: read it back and ask him; go ahead only on a plain yes")
    if args.get("approval_id"):
        ids = [a["id"] for a in waiting if a["id"] == str(args["approval_id"])]
    elif args.get("all") or len(waiting) == 1:
        ids = [a["id"] for a in waiting]
    else:
        ids = []
    if not ids:
        return ToolResult(ok=False, tool_name="approval_decide", error="which_one" if waiting else "nothing_waiting",
                          data={"waiting_for_his_ok": [_brief(a) for a in waiting]})
    done = [decide(i, OWNER, approve) for i in ids]
    return ToolResult(ok=True, tool_name="approval_decide", data={"results": [
        {"what": a["summary"], "status": a["status"], **({"error": a["result"]} if a["status"] == "failed" else {})}
        for a in done if a]})


def send_to_phone(args: Dict[str, Any]) -> ToolResult:
    """A mission's write-up (the latest finished one, or mission_id), or a short text he asked for."""
    if (TURN.get() or {}).get("user") != OWNER:
        return ToolResult(ok=False, tool_name="send_to_phone", error="phone_is_daniels_only")
    if text := str(args.get("text") or "").strip():
        title, body, click = "THURSDAY", text, link()
    else:
        db = initalize_db()
        mission = db.get_mission(str(args["mission_id"]), OWNER) if args.get("mission_id") else None
        mission = mission or next((m for m in db.list_missions(OWNER, "done", limit=20)
                                   if not m.get("watch_id") and m["announced_at"] != "quiet"), None)
        if not mission:
            return ToolResult(ok=False, tool_name="send_to_phone", error="nothing_to_send: no finished mission")
        title, click = f"Mission · {mission['goal'][:60]}", link(mission["conversation_id"])
        body = f"{mission['summary'] or ''}\n\n{mission['result'] or ''}".strip()
    if not push(title, body, click=click, tags=("memo",)):
        return ToolResult(ok=False, tool_name="send_to_phone", error="phone_unreachable")
    return ToolResult(ok=True, tool_name="send_to_phone", data={"sent": title if not text else text[:80]})


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "send_to_phone",
        "description": "Send something to his phone: a mission's write-up (the latest finished one by default) or a "
        "short text he asks for ('send me that', 'text me the address'). Only ever to him.",
        "parameters": {"type": "object", "properties": {
            "mission_id": {"type": "string", "description": "A finished mission, if not the latest"},
            "text": {"type": "string", "description": "Plain text to send instead of a mission write-up"}}}}},
    {"type": "function", "function": {
        "name": "approval_list", "description": "What your missions proposed that's waiting for his OK.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "approval_decide",
        "description": "His answer to something waiting for his OK. approve=true only when he just said yes to it; "
        "false when he says no or not now.",
        "parameters": {"type": "object", "properties": {
            "approve": {"type": "boolean"},
            "approval_id": {"type": "string"},
            "all": {"type": "boolean", "description": "His answer covers everything waiting ('yes, book them all')"}},
            "required": ["approve"]}}},
]
