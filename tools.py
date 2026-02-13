from __future__ import annotations
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Any, Callable, Dict, Literal, Optional
from pydantic import BaseModel, Field

#tool contract
class ToolCall(BaseModel):
    name: Literal["get_time", "echo"]
    args: Dict[str, Any] = Field(default_factory=dict)

#tool contract results
class ToolResult(BaseModel):
    ok: bool
    tool_name: str
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

#Tool Result Function to Get Time
def get_time(args: Dict[str, Any]) -> ToolResult:
    """
    Args:
        - tz: Optional IANA timezone string (e.g. "America/New_York", "Europe/Berlin", Asia/Tokyo).
        - If tz is omitted, use the machine's local timezone (ONLY CORRECT if THURSDAY runs on a user's machine).
    Returns:
        - uts-iso: ISO Timestamp in UTC
        - local-iso: ISO Timestamp in requested tz (or machine local)
        - tz: timezone used
        - offset_seconds: UTC offset in seconds for that tz at that moment (DST-safe).
    """
    tz_name = str(args.get("tz", "")).strip() or None
    now_utc = datetime.now(timezone.utc)

    try:
        if tz_name:
            tz = ZoneInfo(tz_name)
            now_local = now_utc.astimezone(tz)
            used_tz = tz_name
        else:
            now_local = now_utc.astimezone()
            used_tz = str(now_local.tzinfo) if now_local.tzinfo else "local"

        offset = now_local.utcoffset()
        offset_seconds = int(offset.total_seconds()) if offset else 0

        return ToolResult(
            ok=True,
            tool_name="get_time",
            data={
                "utc_iso": now_utc.isoformat().replace("+00:00", "Z"),
                "local_iso": now_local.isoformat(),
                "tz": used_tz,
                "offset_seconds": offset_seconds,
            },
        )
    except ZoneInfoNotFoundError:
        return ToolResult(
           ok=False,
            tool_name="get_time",
            error="unknown_timezone",
            data={"tz": tz_name or ""},  
        )

    #now = datetime.now(timezone.utc)
    #dt_local = now.astimezone()
    #return ToolResult(ok=True, tool_name="get_time", data={"time": dt_local.strftime("%I:%M %p").lstrip("0")})

#Tool Result to echo text
def echo(args: Dict[str, Any]) -> ToolResult:
    text = str(args.get("text", ""))
    return ToolResult(ok=True, tool_name="echo", data={"text": text})

#Registry (Central Allowlist for Thursday)
TOOLS: Dict[str, Callable[[Dict[str, Any]], ToolResult]] = {
    "get_time": get_time,
    "echo": echo,
}

def execute_tool(call: ToolCall) -> ToolResult:
    fn = TOOLS.get(call.name)
    if fn is None:
        return ToolResult(ok=False, tool_name=call.name, error="unknown_tool")
    try:
        return fn(call.args)
    except Exception as e:
        return ToolResult(ok=False, tool_name=call.name, error=f"tool_exception: {e}")