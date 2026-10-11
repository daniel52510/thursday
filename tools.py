from __future__ import annotations
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Any, Callable, Dict, Literal, Optional
from pydantic import BaseModel, Field
import os
import requests

#tool contract
class ToolCall(BaseModel):
    name: Literal["get_time", "echo", "get_weather", "web_search"]
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
    #print("TZ_NAME: ", tz_name)
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

def weather_score_candidate(r: dict, expected_admin1: str, maybe_state_or_country: str, maybe_state: str, city: str) -> int:
    score = 0
    if maybe_state:
        if r.get("country_code") == "US":
            score += 50
        if expected_admin1 and r.get("admin1") == expected_admin1:
            score += 80 
    if maybe_state_or_country and not maybe_state:
        if str(r.get("country", "")).lower() == maybe_state_or_country.lower():
                score += 80
    if str(r.get("name","")).lower() == city.lower():
        score += 5
    return score
    
def get_weather(args: Dict[str, Any]) -> ToolResult:
    US_STATES = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    }
    """
    Args
        - location: str (required)
        - units: Literal["imperial","metric"] = "imperial"
        - days: int
    """ 
    location = str(args.get("location", "") or "").strip()
    units = str(args.get("units", "imperial")).lower().strip()
    if not location:
        return ToolResult(ok=False, tool_name="get_weather", error="missing_location")
    if units not in ("imperial", "metric"):
        units = "imperial"
    parts = [p.strip() for p in location.split(",")]
    city = parts[0]
    maybe_state_or_country = parts[1] if len(parts) > 1 else ""
    maybe_state_or_country = maybe_state_or_country.strip()
    days_raw = args.get("days", 2)
    try:
        days = int(days_raw)
    except (TypeError, ValueError):
        days = 2
    # Clamp forecast length to a safe range
    days = max(1, min(days, 7))

    maybe_state = ""
    # Accept either 2-letter state codes ("OH") or full state names ("Ohio") after the comma.
    if maybe_state_or_country:
        token = maybe_state_or_country.strip()
        if len(token) == 2 and token.isalpha():
            maybe_state = token.upper()
        else:
            # Try to map full state name -> abbreviation
            token_norm = token.lower()
            for abbr, full in US_STATES.items():
                if full.lower() == token_norm:
                    maybe_state = abbr
                    break

    expected_admin1 = US_STATES.get(maybe_state, "")
    GEOCODE_URL = os.getenv("OPEN_METEO_GEOCODE_URL", "https://geocoding-api.open-meteo.com/v1/search")
    #LOCATION is being resolved by City, State and not just City when searching the USA
    resp = requests.get(
        GEOCODE_URL,
        params={
            # Prefer the full user-provided location for geocoding so state/country hints influence ranking.
            "name": city,
            "count": 10,
            "language": "en",
            "format": "json",
        },
        timeout=10,
    )
    resp.raise_for_status() 
    geo = resp.json()

    results = geo.get("results") or []
    if maybe_state:
        # Open-Meteo `admin1` is typically the full state name (e.g., "Florida"), not the 2-letter code.
        candidates = [
            r for r in results
            if (r.get("country_code") == "US") or (r.get("admin1") == expected_admin1)
        ]
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
    best = max(
        results,
        key=lambda r: weather_score_candidate(r, expected_admin1, maybe_state_or_country, maybe_state, city),
    )
    lat = best["latitude"]
    lon = best["longitude"]

    print("BEST: ", best)

    FORECAST_URL = os.getenv("OPEN_METEO_FORECAST_URL", "https://api.open-meteo.com/v1/forecast")
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

# --- web_search (SearXNG) ---------------------------------------------------
_NEWS_HINTS = ("news", "latest", "breaking", "headline", "today", "this week", "update")
_SEARXNG_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "THURSDAY/1.0",
    # SearXNG's bot detection logs an error (and the limiter may block) when no
    # client IP header is present. THURSDAY calls it directly over the Docker
    # network, so identify ourselves as a local client.
    "X-Forwarded-For": "127.0.0.1",
    "X-Real-IP": "127.0.0.1",
}


def _normalize_results(raw_results: list, limit: int) -> list[dict]:
    results: list[dict] = []
    seen: set[str] = set()
    for r in raw_results or []:
        url = str(r.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        snippet = " ".join(str(r.get("content") or "").split())
        results.append(
            {
                "title": str(r.get("title") or "").strip(),
                "url": url,
                "snippet": snippet[:400],
                "engine": r.get("engine", ""),
                "published_date": r.get("publishedDate") or r.get("published_date"),
            }
        )
        if len(results) >= limit:
            break
    return results


def web_search(args: Dict[str, Any]) -> ToolResult:
    """
    Args
        - query: the query to search the web for (required).
        - max_results: optional int, 1-10 (default 5).
    Returns
        - results: list of {title, url, snippet, engine, published_date}
        - ok=False with error/unresponsive_engines when SearXNG returns nothing,
          so the model reports the failure instead of inventing an answer.
    """
    query = str(args.get("query") or args.get("q") or "").strip()
    if not query:
        return ToolResult(ok=False, tool_name="web_search", error="missing_query")

    try:
        limit = max(1, min(int(args.get("max_results", 5)), 10))
    except (TypeError, ValueError):
        limit = 5

    base_url = os.getenv("SEARXNG_BASE_URL", "http://localhost:55000").rstrip("/")
    search_url = f"{base_url}/search"
    timeout = float(os.getenv("SEARXNG_TIMEOUT", "15"))

    # News-style questions try the news category first (recent results),
    # then everything falls back to the general web category.
    attempts: list[dict] = []
    if any(h in query.lower() for h in _NEWS_HINTS):
        attempts.append({"categories": "news", "time_range": "week"})
    attempts.append({"categories": "general"})

    unresponsive: dict[str, str] = {}
    for extra in attempts:
        params = {
            "q": query,
            "format": "json",
            "language": "en",
            "safesearch": "0",
            "pageno": 1,
            **extra,
        }
        try:
            resp = requests.get(search_url, params=params, headers=_SEARXNG_HEADERS, timeout=timeout)
        except requests.RequestException as exc:
            return ToolResult(
                ok=False,
                tool_name="web_search",
                error=f"searxng_unreachable: {exc}",
                data={"query": query, "search_url": search_url},
            )

        if resp.status_code == 403:
            return ToolResult(
                ok=False,
                tool_name="web_search",
                error="searxng_json_disabled",
                data={"query": query, "hint": "Add 'json' to search.formats in searxng/settings.yml"},
            )
        if resp.status_code == 429:
            return ToolResult(
                ok=False,
                tool_name="web_search",
                error="searxng_rate_limited",
                data={"query": query, "hint": "Set server.limiter: false in searxng/settings.yml"},
            )
        if not resp.ok:
            return ToolResult(
                ok=False,
                tool_name="web_search",
                error=f"searxng_http_{resp.status_code}",
                data={"query": query, "response_preview": resp.text[:300]},
            )

        try:
            data = resp.json()
        except ValueError:
            return ToolResult(
                ok=False,
                tool_name="web_search",
                error="invalid_json_response",
                data={"query": query, "response_preview": resp.text[:300]},
            )

        for item in data.get("unresponsive_engines") or []:
            if isinstance(item, (list, tuple)) and item:
                unresponsive[str(item[0])] = str(item[1]) if len(item) > 1 else ""

        results = _normalize_results(data.get("results"), limit)
        if results:
            return ToolResult(
                ok=True,
                tool_name="web_search",
                data={
                    "query": query,
                    "category": extra["categories"],
                    "result_count": len(results),
                    "results": results,
                },
            )

    if unresponsive:
        print(f"[web_search] no results for {query!r}; unresponsive engines: {unresponsive}")
    return ToolResult(
        ok=False,
        tool_name="web_search",
        error="no_results",
        data={
            "query": query,
            "unresponsive_engines": unresponsive,
            "hint": "Search engines returned nothing (often blocked or rate-limited). Tell the user the search failed.",
        },
    )

#Tool Result to echo text
def echo(args: Dict[str, Any]) -> ToolResult:
    text = str(args.get("text", ""))
    return ToolResult(ok=True, tool_name="echo", data={"text": text})

#Registry (Central Allowlist for Thursday)
TOOLS: Dict[str, Callable[[Dict[str, Any]], ToolResult]] = {
    "get_time": get_time,
    "echo": echo,
    "get_weather": get_weather,
    "web_search": web_search,
}

def execute_tool(call: ToolCall) -> ToolResult:
    fn = TOOLS.get(call.name)
    if fn is None:
        return ToolResult(ok=False, tool_name=call.name, error="unknown_tool")
    try:
        return fn(call.args)
    except Exception as e:
        return ToolResult(ok=False, tool_name=call.name, error=f"tool_exception: {e}")
    
