"""Shared Ollama config, search routing, and fact extraction for agent_stream."""
import json
import os
import re
from typing import Optional

import requests

from agent_schemas import FactExtraction
from memory import MemoryDB

FACT_EXTRACTOR_SYSTEM = """
You are a fact extractor for a local assistant.
Return ONLY valid JSON matching this shape:
{"facts":[{"key":string,"value":any,"confidence":number,"source":"explicit_user"|"assistant_inference"|"tool_result"}]}

Rules:
- Extract ONLY stable, long-lived facts (name, preferences, timezone, ongoing constraints/goals).
- Max 5 facts.
- DO NOT store secrets (passwords, tokens, API keys), financial numbers, or anything sensitive.
- If nothing worth saving: {"facts":[]}
""".strip()

URL = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "gemma4:26b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))
# Ollama defaults to a 4096-token context and silently drops the START of the
# prompt when it overflows -- i.e. the system prompt with the JSON rules.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_OPTIONS = {"num_ctx": OLLAMA_NUM_CTX}
# Ollama unloads an idle model after 5 minutes; reloading a 26B model costs
# seconds on the next turn. Keep it resident (Ollama duration string, or -1 forever).
_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
# Ollama reads a bare number as seconds (negative = never unload) but rejects "-1" as a string.
OLLAMA_KEEP_ALIVE = int(_KEEP_ALIVE) if _KEEP_ALIVE.lstrip("-").isdigit() else _KEEP_ALIVE

# --- Search routing backstop ----------------------------------------------
# Small local models often answer "current" questions from stale training data
# instead of calling web_search. If the question clearly needs fresh info and
# the model picked NO tool, we force a web_search ourselves.
_EXPLICIT_SEARCH = re.compile(
    r"\b(search|look\s+(it\s+|that\s+|this\s+)?up|google\s+it|"
    r"(check|verify|confirm)\b[^.?!]{0,30}\bonline|on\s+the\s+(web|internet))\b",
    re.I,
)
_FRESHNESS = re.compile(
    r"\b(current(ly)?|latest|newest|today'?s?|tonight|right\s+now|this\s+(week|weekend|month|year|season)|"
    r"breaking|news|headlines?|recent(ly)?|still|anymore|nowadays|these\s+days|next|upcoming|"
    r"price|prices|cost|worth|stock|market\s+cap|exchange\s+rate|bitcoin|crypto|"
    r"version|released?|release\s+date|launch(ed|es)?|out\s+yet|announced?|"
    r"score|won|win|winning|standings|playoffs?|election|polls?|"
    r"open\s+(now|today)|hours|near\s+me|nearby|around\s+me|(19|20)\d\d)\b",
    re.I,
)
# A role anywhere in the question: who holds it may have changed since training.
_OFFICE_HOLDER = re.compile(
    r"\b(ceo|cfo|cto|coo|president|vice\s+president|prime\s+minister|governor|mayor|senator|"
    r"chancellor|pope|king|queen|chair(man|woman|person)?|head\s+coach|coach|owner|founder|leader|"
    r"spokesperson|secretary|minister|director)s?\b|"
    r"\bwho\s+(runs|leads|owns|heads|founded|won|is\s+winning)\b",
    re.I,
)
# Questions the weather/time tools answer; forcing a search first would only add latency.
_OWN_TOOL = re.compile(r"\b(weather|forecast|rain|snow|temperature|umbrella|what\s+time|time\s+is\s+it|what('s|\s+is)\s+the\s+date|what\s+day)\b", re.I)
# Words that say "go search" but not WHAT to search for.
_SEARCH_FILLER = {
    "search", "look", "up", "google", "check", "verify", "confirm", "double", "online", "web",
    "internet", "please", "again", "it", "that", "this", "your", "my", "info", "information",
    "answer", "the", "on", "can", "could", "you", "for", "me", "do", "a", "now", "and", "go", "ahead",
}


def _needs_search(text: str) -> bool:
    if _EXPLICIT_SEARCH.search(text):
        return True
    if _OWN_TOOL.search(text):
        return False
    return bool(_FRESHNESS.search(text) or _OFFICE_HOLDER.search(text))


def _has_search_content(text: str) -> bool:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return any(w not in _SEARCH_FILLER for w in words)


def _search_query_for(user_prompt: str, history: list[dict]) -> str:
    """Use the prompt itself, unless it's just 'check that online' -- then
    search for the user's previous question instead."""
    if _has_search_content(user_prompt):
        return user_prompt.strip()
    for m in reversed(history):
        content = str(m.get("content") or "").strip()
        if m.get("role") == "user" and content and content != user_prompt.strip() and _has_search_content(content):
            return content
    return user_prompt.strip()


def should_extract_facts(text: str) -> bool:
    t = text.lower()
    triggers = [
        "remember", 
        "from now on",
        "my name",
        "call me",
        "i live",
        "i am",
        "i prefer",
        "timezone",
        "my email",
        "my phone",
        "my address",
    ]
    return any(x in t for x in triggers)


def initalize_db() -> MemoryDB:
    # Creating the DB object creates tables + enables WAL.
    # Your methods open/close connections per call, so this isn't a global connection.
    return MemoryDB()


def preload_model() -> None:
    """Load the model into Ollama now (a request with no prompt only loads it)."""
    try:
        requests.post(URL, json={"model": MODEL, "keep_alive": OLLAMA_KEEP_ALIVE}, timeout=OLLAMA_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        print(f"[ollama] preload of {MODEL} failed: {exc}")


def _post_ollama(payload: dict) -> str:
    """Call Ollama and return the raw JSON string in resp['response']."""
    try:
        resp = requests.post(URL, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Could not reach Ollama at {URL}: {exc}") from exc
    return resp.json()["response"]


_EPHEMERAL_FACT = re.compile(r"time|date|day|weather|temperature|forecast|current", re.I)


def run_fact_extractor(
    db: MemoryDB,
    user_prompt: str,
    final_reply: str,
    tool_results_json: Optional[list],
) -> None:
    if not should_extract_facts(user_prompt):
        return

    extractor_prompt = f"""
USER_TEXT:
{user_prompt}

ASSISTANT_TEXT:
{final_reply}

TOOL_RESULTS_JSON:
{json.dumps(tool_results_json, ensure_ascii=False) if tool_results_json else "null"}
""".strip()

    payload = {
        "model": MODEL,
        "system": FACT_EXTRACTOR_SYSTEM,
        "prompt": extractor_prompt,
        "format": "json",
        "stream": False,
        "think": False,
        "options": OLLAMA_OPTIONS,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }

    raw = _post_ollama(payload)
    parsed = json.loads(raw)
    extraction = FactExtraction.model_validate(parsed)

    # convenience method you added in memory.py
    # Time, date and weather are never long-lived facts, and tool output isn't
    # something the user told us; both went stale or wrong in practice.
    keep = [
        f.model_dump() for f in extraction.facts
        if f.source != "tool_result" and not _EPHEMERAL_FACT.search(f.key)
    ]
    if keep:
        db.upsert_facts(keep)
