"""Live data from the source instead of search snippets: sports scores (ESPN's
public scoreboard feed) and rocket launches (The Space Devs' Launch Library 2).
Both are free and keyless. ESPN's feed is unofficial and can change shape;
Launch Library allows ~15 requests an hour without a key, so its answers are cached.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

from tools import ToolResult

ESPN = "https://site.api.espn.com/apis/site/v2/sports"
LEAGUES = {
    "nfl": "football/nfl", "college-football": "football/college-football",
    "nba": "basketball/nba", "wnba": "basketball/wnba",
    "mens-college-basketball": "basketball/mens-college-basketball",
    "womens-college-basketball": "basketball/womens-college-basketball",
    "mlb": "baseball/mlb", "nhl": "hockey/nhl",
    "mls": "soccer/usa.1", "premier-league": "soccer/eng.1", "la-liga": "soccer/esp.1",
    "serie-a": "soccer/ita.1", "bundesliga": "soccer/ger.1", "ligue-1": "soccer/fra.1",
    "champions-league": "soccer/uefa.champions",
}
TIMEOUT = 8
_teams_cache: dict[str, tuple[float, list[dict]]] = {}
TEAMS_TTL_S = 24 * 3600


def _tz(args: Dict[str, Any]) -> timezone | ZoneInfo:
    try:
        return ZoneInfo(str(args.get("timezone") or args.get("tz") or "UTC"))
    except (ZoneInfoNotFoundError, ValueError):
        return timezone.utc


def _local(iso: str, tz) -> str:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(tz).strftime("%a %b %-d, %-I:%M %p")
    except ValueError:
        return iso


def _game(event: dict, tz) -> dict:
    comp = event["competitions"][0]
    status = comp.get("status") or event.get("status") or {}
    sides = {}
    for c in comp.get("competitors", []):
        score = c.get("score")
        sides[c.get("homeAway", "home")] = {
            "team": c["team"].get("displayName"),
            "score": score.get("displayValue") if isinstance(score, dict) else score,
            "record": next((r.get("summary") for r in c.get("records", []) if r.get("summary")), None),
            "winner": c.get("winner"),
        }
    state = (status.get("type") or {}).get("state")  # pre / in / post
    return {
        "matchup": event.get("name"),
        "when": _local(event.get("date", ""), tz),
        "state": {"pre": "scheduled", "in": "live", "post": "final"}.get(state, state),
        "status": (status.get("type") or {}).get("shortDetail"),
        "away": sides.get("away"),
        "home": sides.get("home"),
        "broadcast": ", ".join(n for b in comp.get("broadcasts", []) for n in b.get("names", [])) or None,
    }


def _teams(path: str) -> list[dict]:
    cached = _teams_cache.get(path)
    if cached and time.monotonic() - cached[0] < TEAMS_TTL_S:
        return cached[1]
    resp = requests.get(f"{ESPN}/{path}/teams", params={"limit": 1000}, timeout=TIMEOUT)
    resp.raise_for_status()
    teams = [t["team"] for s in resp.json().get("sports", []) for lg in s.get("leagues", []) for t in lg.get("teams", [])]
    _teams_cache[path] = (time.monotonic(), teams)
    return teams


def _find_team(path: str, name: str) -> dict | None:
    q = name.lower().strip()
    best, best_score = None, 0
    for t in _teams(path):
        names = [t.get(k, "") for k in ("displayName", "shortDisplayName", "name", "nickname", "location", "abbreviation")]
        names = [n.lower() for n in names if n]
        score = 3 if q in names else 2 if any(q in n for n in names) else 1 if any(n in q for n in names if len(n) > 3) else 0
        if score > best_score:
            best, best_score = t, score
    return best


def sports_scores(args: Dict[str, Any]) -> ToolResult:
    """
    Args
        - league: one of LEAGUES (required).
        - team: optional team name; returns its live game, last result and next game.
        - date: optional YYYY-MM-DD; without a team, that day's scoreboard (default today).
    """
    league = str(args.get("league") or "").lower().strip()
    path = LEAGUES.get(league)
    if not path:
        return ToolResult(ok=False, tool_name="sports_scores", error="unknown_league", data={"leagues": sorted(LEAGUES)})
    tz = _tz(args)
    try:
        if args.get("team"):
            team = _find_team(path, str(args["team"]))
            if not team:
                return ToolResult(ok=False, tool_name="sports_scores", error="team_not_found", data={"team": args["team"]})
            resp = requests.get(f"{ESPN}/{path}/teams/{team['id']}/schedule", timeout=TIMEOUT)
            resp.raise_for_status()
            games = [_game(e, tz) for e in resp.json().get("events", [])]
            live = next((g for g in games if g["state"] == "live"), None)
            if live:
                # The schedule feed lags during a game; the scoreboard has the live score.
                board = requests.get(f"{ESPN}/{path}/scoreboard", timeout=TIMEOUT).json().get("events", [])
                live = next((_game(e, tz) for e in board if team["displayName"] in e.get("name", "")), live)
            return ToolResult(ok=True, tool_name="sports_scores", data={
                "team": team["displayName"],
                "live_game": live,
                "last_game": next((g for g in reversed(games) if g["state"] == "final"), None),
                "next_game": next((g for g in games if g["state"] == "scheduled"), None),
            })
        day = str(args.get("date") or datetime.now(tz).strftime("%Y-%m-%d"))
        resp = requests.get(f"{ESPN}/{path}/scoreboard", params={"dates": day.replace("-", "")}, timeout=TIMEOUT)
        resp.raise_for_status()
        games = [_game(e, tz) for e in resp.json().get("events", [])][:16]
        return ToolResult(ok=True, tool_name="sports_scores", data={"league": league, "date": day, "games": games})
    except (requests.RequestException, ValueError, KeyError) as exc:
        return ToolResult(ok=False, tool_name="sports_scores", error=f"espn_failed: {exc}")


LL2 = "https://ll.thespacedevs.com/2.3.0/launches"
LAUNCH_TTL_S = 10 * 60
_launch_cache: dict[tuple, tuple[float, list]] = {}


def _launch(l: dict, tz) -> dict:
    mission = l.get("mission") or {}
    pad = l.get("pad") or {}
    return {
        "name": l.get("name"),
        "when": _local(l.get("net", ""), tz),
        "time_precision": (l.get("net_precision") or {}).get("name"),
        "status": (l.get("status") or {}).get("name"),
        "provider": (l.get("launch_service_provider") or {}).get("name"),
        "rocket": ((l.get("rocket") or {}).get("configuration") or {}).get("full_name"),
        "pad": f"{pad.get('name', '')}, {(pad.get('location') or {}).get('name', '')}".strip(", "),
        "mission": (mission.get("description") or "")[:240] or None,
        "orbit": (mission.get("orbit") or {}).get("name"),
        "launch_probability": l.get("probability"),
        "weather_concerns": l.get("weather_concerns"),
        "webcast_live": l.get("webcast_live"),
        "fail_reason": l.get("failreason") or None,
    }


def launch_schedule(args: Dict[str, Any]) -> ToolResult:
    """
    Args
        - when: "upcoming" (default) or "recent".
        - search: optional rocket, provider or mission (e.g. "Starship", "Falcon 9", "Blue Origin").
        - florida_only: optional bool, launches from Cape Canaveral / Kennedy only.
        - limit: 1-6 (default 4).
    """
    when = "previous" if str(args.get("when", "upcoming")).lower() in ("recent", "previous", "past") else "upcoming"
    limit = max(1, min(int(args.get("limit") or 4), 6))
    params = {"limit": limit, "mode": "normal"}
    if args.get("search"):
        params["search"] = str(args["search"])
    if args.get("florida_only"):
        params["location__ids"] = "12,27"  # Cape Canaveral SFS, Kennedy Space Center
    key = (when, tuple(sorted(params.items())))
    cached = _launch_cache.get(key)
    try:
        if cached and time.monotonic() - cached[0] < LAUNCH_TTL_S:
            results = cached[1]
        else:
            resp = requests.get(f"{LL2}/{when}/", params=params, timeout=TIMEOUT)
            if resp.status_code == 429:
                return ToolResult(ok=False, tool_name="launch_schedule", error="rate_limited_try_web_search")
            resp.raise_for_status()
            results = resp.json().get("results", [])
            _launch_cache[key] = (time.monotonic(), results)
    except (requests.RequestException, ValueError) as exc:
        return ToolResult(ok=False, tool_name="launch_schedule", error=f"launch_library_failed: {exc}")
    tz = _tz(args)
    return ToolResult(ok=True, tool_name="launch_schedule",
                      data={"when": "recent" if when == "previous" else "upcoming", "launches": [_launch(l, tz) for l in results]})
