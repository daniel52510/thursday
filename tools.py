from __future__ import annotations
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Any, Callable, Dict, Literal, Optional
from pydantic import BaseModel, Field
import requests

#tool contract
class ToolCall(BaseModel):
    name: Literal["get_time", "echo", "get_weather"]
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
    Args
        - tz: Optional IANA timezone string (e.g. "America/New_York", "Europe/Berlin", Asia/Tokyo).
        - If tz is omitted, use the machine's local timezone (ONLY CORRECT if THURSDAY runs on a user's machine).
        - If user prompts their location, then return MyLocation as the value for tz.
    Returns
        - uts-iso: ISO Timestamp in UTC
        - local-iso: ISO Timestamp in requested tz (or machine local)
        - tz: timezone used
        - offset_seconds: UTC offset in seconds for that tz at that moment (DST-safe).
    """
    tz_name = (args.get("timezone") or args.get("tz") or "")
    tz_name = str(tz_name).strip() or None
    print("TZ_NAME: ", tz_name)
    now_utc = datetime.now(timezone.utc)
    try:
        if tz_name is not None:
            print("reaching this conditional!")
            tz = ZoneInfo(tz_name)
            now_local = now_utc.astimezone(tz)
            print("NOW_LOCAL: ", now_local)
            used_tz = tz_name
        #creating elif and else statement (NEEDED) to allow graceful erroring if location does not exist or if user ask about own location
        else:
            #now_local = now_utc.astimezone()
            return ToolResult(ok=False, tool_name="get_time", error="missing_location")
            #used_tz = str(now_local.tzinfo) if now_local.tzinfo else "local"

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

def get_weather(args: Dict[str, Any]) -> ToolResult:
    """
    Args
        - location: str (required)
        - units: Literal["imperial","metric"] = "imperial"
        - days: int
    """ 
    location = str(args.get("location")).strip()
    units = str(args.get("units", "imperial")).lower().strip()
    if not location:
        return ToolResult(ok=False, tool_name="get_weather", error="missing_location")
    if units not in ("imperial", "metric"):
        units = "imperial"
    parts = [p.strip() for p in location.split(",")]
    city = parts[0]
    maybe_state = parts[1] if len(parts) > 1 else ""
    print("MAYBE-STATE: ", maybe_state)
    if len(maybe_state) == 2 and maybe_state.isalpha():
        state = maybe_state.upper()
    days_raw = args.get("days", 1)
    try:
        days = int(days_raw)
    except (TypeError, ValueError):
        days = 1
    # Clamp forecast length to a safe range
    days = max(1, min(days, 7))
    
    GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
    #LOCATION is being resolved by City, State and not just City when searching the USA
    resp = requests.get(
        GEOCODE_URL,
        params={
            "name": city,
            "count": 5,
            "language": "en",
            "format": "json",
        },
        timeout=10,
    )
    resp.raise_for_status()
    geo = resp.json()

    results = geo.get("results") or []
    if state:
        candidates = [r for r in results if (r.get("country_code") == "US") or (r.get("admin1") == maybe_state)]
        if candidates:
            results = candidates
    if not results:
        return ToolResult(
        ok=False,
        tool_name="get_weather",
        error="geocode_no_results",
        data={"input_location": location}
        )
    #Gotta figure out results so that for US states, it is linked to state abbreviations!
    best = results[0]  # for now; later you can choose best more carefully
    lat = best["latitude"]
    lon = best["longitude"]

    print("BEST: ", best)

    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    resp = requests.get(
        FORECAST_URL,
        params={
        "latitude": lat,
        "longitude": lon,
        "current_weather": "true",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto",
        "forecast_days": days,
        # Units
        "temperature_unit": "fahrenheit" if units == "imperial" else "celsius",
        "wind_speed_unit": "mph" if units == "imperial" else "kmh",
        "precipitation_unit": "inch" if units == "imperial" else "mm",
        },
        timeout=10
    )
    resp.raise_for_status()
    wx = resp.json()
    current = wx.get("current_weather") or {}
    daily = wx.get("daily") or {}

    return ToolResult(ok=True, tool_name="get_weather", data={
    "input_location": location,
    "resolved_location": ", ".join(
        [x for x in [best.get("name"), best.get("admin1"), best.get("country")] if x]
    ),
    "latitude": lat,
    "longitude": lon,
    "timezone": wx.get("timezone"),
    "units": units,
    "current": current,
    "today": {
        "date": (daily.get("time") or [None])[0],
        "temp_max": (daily.get("temperature_2m_max") or [None])[0],
        "temp_min": (daily.get("temperature_2m_min") or [None])[0],
        "precip_sum": (daily.get("precipitation_sum") or [None])[0],
    }
})

#Tool Result to echo text
def echo(args: Dict[str, Any]) -> ToolResult:
    text = str(args.get("text", ""))
    return ToolResult(ok=True, tool_name="echo", data={"text": text})

#Registry (Central Allowlist for Thursday)
TOOLS: Dict[str, Callable[[Dict[str, Any]], ToolResult]] = {
    "get_time": get_time,
    "echo": echo,
    "get_weather": get_weather,
}

def execute_tool(call: ToolCall) -> ToolResult:
    fn = TOOLS.get(call.name)
    if fn is None:
        return ToolResult(ok=False, tool_name=call.name, error="unknown_tool")
    try:
        return fn(call.args)
    except Exception as e:
        return ToolResult(ok=False, tool_name=call.name, error=f"tool_exception: {e}")