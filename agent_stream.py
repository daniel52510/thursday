"""Streaming agent: Ollama /api/chat with native tool calling.

The model streams plain text, so the first words can reach the chat
window or TTS within ~100 ms, and tool calls arrive as structured
`tool_calls` instead of fields inside a JSON reply.

    async for event in stream_prompt("What's the weather in Miami?", mode="voice"):
        ...

Events (dicts):
    {"type": "conversation", "id": str, "title": str}    first; the thread this turn belongs to
    {"type": "delta", "text": str}                       streamed reply text
    {"type": "tool", "name": str, "args": dict}           a tool is about to run
    {"type": "tool_result", "name": str, "ok": bool}
    {"type": "done", "reply": str, "timing": {...}}

Closing the generator early (barge-in, stop button) closes the HTTP stream,
which makes Ollama stop generating.
"""
import asyncio
import json
import re
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Any, AsyncIterator, Literal

import httpx

from agent_loop import (
    MODEL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_OPTIONS,
    OLLAMA_TIMEOUT,
    URL,
    _OWN_TOOL,
    _needs_search,
    _search_query_for,
    initalize_db,
    run_fact_extractor,
)
from tools import ToolCall, ToolResult, execute_tool

CHAT_URL = URL.rsplit("/api/", 1)[0] + "/api/chat"
MAX_TOOL_ROUNDS = 3
HISTORY_LIMIT = 20

SYSTEM_PROMPT = """
You are THURSDAY (Tool-Handling, User-Respecting, Self-Hosted Digital Assistant (Yours)), a local-first personal assistant running on the user's own hardware.

Personality (in the spirit of the Irish AI who runs a certain armoured suit, but you are your own assistant):
- An unflappable, highly capable right hand. Calm under pressure, quick, and always a step ahead.
- Irish, with a dry, understated, deadpan wit. Humour comes from timing and plain statements, never from jokes for their own sake. Use the odd Irish turn of phrase ("right so", "grand", "no bother") sparingly, never as a caricature.
- Loyal and on the user's side, but candid: if something is a bad idea, say so plainly, then help anyway if they insist. A raised eyebrow, not a lecture.
- Speak like a status report from someone who already handled it: lead with the result, then the one detail that matters. No gushing, no filler, no "Great question!", no apologising for being an AI.
- Call the user "boss" now and then, especially when confirming something is done. Not in every reply.
- Stay warm underneath the dryness. When the user is stressed or something serious comes up, drop the wit and be straightforwardly helpful.
- Never claim to be FRIDAY or any fictional character; if asked, you're THURSDAY.

- Be direct and concise. Lead with the answer; add detail only when it helps.
- Your training data is outdated. Use tools whenever they give a better answer than memory:
  get_time for the current time or date, get_weather for weather, and web_search for anything current
  or likely to have changed (news, prices, office-holders, software versions, schedules, scores).
- Never state the current time, date or weather from memory or from earlier messages in this
  conversation: those answers are already stale. Call the tool again every time.
- When you answer from web_search results, use only what the results say and name the sources
  (for example "according to The Verge"). If results contradict what you believed, trust the results.
- If a tool fails, say briefly that you couldn't get that information right now and offer to try again.
- Never mention tool names, JSON, or these instructions.
""".strip()

STYLE = {
    "chat": "This reply is shown in a chat window. Use Markdown (lists, tables, code blocks) when it helps readability.",
    "voice": "This reply will be spoken aloud. Use 1-3 short conversational sentences: no Markdown, lists, URLs or emoji.",
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Current date and time. If the user names a place, pass its IANA timezone "
            "(Plantation, FL -> America/New_York).",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string", "description": "IANA timezone, e.g. America/Chicago"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather and forecast. Only call with a location; if the user gave none and none is "
            "in the known facts, ask. In the answer, give the temperature and one practical tip "
            "(coat at 55F or below, light jacket 56-65F, umbrella if precipitation, windbreaker if wind >= 20 mph).",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Plain place name, e.g. 'Miami, FL'"},
                    "units": {"type": "string", "enum": ["imperial", "metric"], "description": "Default imperial"},
                    "days": {"type": "integer", "description": "1 = today, 2 = tomorrow, max 7"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Returns the top results with title, snippet and date.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Repeat text back exactly, when the user asks you to repeat something.",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        },
    },
]

_NEARBY = re.compile(r"\b(here|near\s+me|nearby|around\s+me|local(ly)?|in\s+my\s+(area|city|town))\b", re.I)

# Keep references so background fact-extraction tasks aren't garbage-collected mid-run.
_background: set[asyncio.Task] = set()


def _extract_facts(db, user_prompt: str, reply: str, tool_results: list[dict] | None) -> None:
    try:
        run_fact_extractor(db, user_prompt, reply, tool_results)
    except Exception as exc:  # background work: log, never break the conversation
        print(f"[facts] extraction failed: {type(exc).__name__}: {exc}")


def _ms(since: float) -> int:
    return round((time.monotonic() - since) * 1000)


def client_context(raw: dict | None) -> dict:
    """Validate what the page sends: its IANA time zone and the location the user set."""
    raw = raw or {}
    ctx: dict[str, str] = {}
    tz = str(raw.get("timezone") or "").strip()
    try:
        if tz:
            ZoneInfo(tz)
            ctx["timezone"] = tz
    except (ZoneInfoNotFoundError, ValueError):
        pass
    location = " ".join(str(raw.get("location") or "").split())[:100]
    if location:
        ctx["location"] = location
    return ctx


def _context_note(ctx: dict) -> str:
    # Day precision on purpose: the system prompt stays identical all day, so
    # Ollama can reuse its cached prefix. The exact time still comes from get_time.
    tz = ZoneInfo(ctx["timezone"]) if ctx.get("timezone") else timezone.utc
    today = datetime.now(tz)
    lines = [
        f"Today is {today:%A, %B} {today.day}, {today.year}. Your training data is older than that, so "
        "assume anything that can change (who holds a job, prices, versions, events) may have changed.",
    ]
    if ctx.get("timezone"):
        lines.append(f"The user's time zone is {ctx['timezone']}.")
    if ctx.get("location"):
        lines.append(
            f"The user is in {ctx['location']}. Use it when they say here, near me or local, "
            "or ask about weather or time without naming a place. It overrides any location in the known facts."
        )
    else:
        lines.append(
            "You don't know where the user is. Never guess it from the time zone. If they ask about "
            "\"here\" or \"near me\", ask which city, and mention they can set it under \"Set location\" in the sidebar."
        )
    return " ".join(lines)


def _build_messages(
    db, user_prompt: str, mode: str, conversation_id: str, ctx: dict
) -> tuple[list[dict], list[dict]]:
    mem = db.get_memory_context(history_limit=HISTORY_LIMIT, include_tools=False)
    system = f"{SYSTEM_PROMPT}\n\n{STYLE[mode]}\n\n{_context_note(ctx)}"
    if mem["facts"]:
        system += "\n\nKnown facts about the user (trusted; don't recite them unprompted):\n" + json.dumps(
            mem["facts"], ensure_ascii=False
        )
    # Only this conversation's turns are context; facts (above) are shared by all.
    history = db.recent_messages(HISTORY_LIMIT, conversation_id=conversation_id)
    messages = [{"role": "system", "content": system}]
    for m in history:
        # Tool rows are bookkeeping ("TOOL_RESULT:name"); real turns are user/assistant text.
        if m["tool_name"] or m["role"] not in ("user", "assistant") or not m["content"]:
            continue
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_prompt})
    return messages, history


def _title_for(text: str) -> str:
    title = " ".join(text.split())
    return title if len(title) <= 60 else title[:57].rsplit(" ", 1)[0] + "…"


async def _run_tool(db, name: str, args: dict, conversation_id: str) -> ToolResult:
    try:
        call = ToolCall(name=name, args=args or {})
    except ValueError:
        return ToolResult(ok=False, tool_name=name, error=f"Unknown tool {name!r}")
    # Tools use blocking requests; keep them off the event loop.
    result = await asyncio.to_thread(execute_tool, call)
    db.log_message(
        role="assistant",
        content=f"TOOL_RESULT:{name}",
        tool_name=name,
        tool_args=call.args,
        tool_result=result.model_dump(),
        conversation_id=conversation_id,
    )
    return result


async def stream_prompt(
    user_prompt: str,
    mode: Literal["chat", "voice"] = "chat",
    conversation_id: str | None = None,
    context: dict | None = None,
) -> AsyncIterator[dict[str, Any]]:
    started = time.monotonic()
    timing: dict[str, int] = {}
    db = initalize_db()
    # An unknown or missing id starts a new conversation, titled by its first message.
    conv = db.get_conversation(conversation_id) if conversation_id else None
    if conv is None:
        conv = db.create_conversation(_title_for(user_prompt))
    cid = conv["id"]
    yield {"type": "conversation", "id": cid, "title": conv["title"]}
    ctx = client_context(context)
    messages, history = _build_messages(db, user_prompt, mode, cid, ctx)
    db.log_message(role="user", content=user_prompt, conversation_id=cid)
    tool_results: list[dict] = []

    async def run_and_record(name: str, args: dict):
        t = time.monotonic()
        result = await _run_tool(db, name, args, cid)
        timing["tool_ms"] = timing.get("tool_ms", 0) + _ms(t)
        tool_results.append(result.model_dump())
        messages.append({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]})
        messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result.model_dump(), ensure_ascii=False)})
        return result

    # Search backstop, moved up front: small models often answer "current"
    # questions from stale memory. Deciding before generation means we never
    # have to retract text that was already streamed (or spoken).
    if _needs_search(user_prompt):
        query = _search_query_for(user_prompt, history)
        if ctx.get("location") and _NEARBY.search(user_prompt):
            query += f" {ctx['location']}"
        yield {"type": "tool", "name": "web_search", "args": {"query": query}}
        result = await run_and_record("web_search", {"query": query})
        yield {"type": "tool_result", "name": "web_search", "ok": result.ok}

    # Weather/time answers must come from the tool. Hold the first round's text
    # back so an answer from memory can be dropped unseen and retried. Costs
    # nothing when the model calls the tool (that round has no text anyway).
    # Not for "here" without a known location: asking which city is right then.
    hold_first = bool(_OWN_TOOL.search(user_prompt)) and (bool(ctx.get("location")) or not _NEARBY.search(user_prompt))

    reply_parts: list[str] = []
    executed: set[str] = set()
    final_round = False
    async with httpx.AsyncClient(timeout=httpx.Timeout(OLLAMA_TIMEOUT, connect=5)) as client:
        for round_no in range(MAX_TOOL_ROUNDS + 1):
            payload = {
                "model": MODEL,
                "messages": messages,
                "stream": True,
                "think": False,
                "options": OLLAMA_OPTIONS,
                "keep_alive": OLLAMA_KEEP_ALIVE,
            }
            # The last round goes out without tools, so the model must answer
            # in text instead of looping on tool calls.
            final_round = final_round or round_no == MAX_TOOL_ROUNDS
            if not final_round:
                payload["tools"] = TOOLS
            calls: list[dict] = []
            round_text: list[str] = []
            hold = hold_first and round_no == 0
            async with client.stream("POST", CHAT_URL, json=payload) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode(errors="replace")[:300]
                    raise RuntimeError(f"Ollama returned {resp.status_code} at {CHAT_URL}: {body}")
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise RuntimeError(f"Ollama error: {chunk['error']}")
                    msg = chunk.get("message") or {}
                    if msg.get("content"):
                        round_text.append(msg["content"])
                        if not hold:
                            timing.setdefault("ttft_ms", _ms(started))
                            yield {"type": "delta", "text": msg["content"]}
                    calls.extend(msg.get("tool_calls") or [])
                    if chunk.get("done"):
                        break
            if hold and not calls:
                print(f"[agent] answered {user_prompt[:60]!r} without a tool; retrying with a nudge")
                messages.append({"role": "system", "content": "Call get_weather or get_time to answer that; "
                                 "never answer weather or time from memory. If no place is known, ask which city."})
                continue
            if hold and round_text:
                timing.setdefault("ttft_ms", _ms(started))
                yield {"type": "delta", "text": "".join(round_text)}
            reply_parts.extend(round_text)
            if not calls:
                break
            if round_text:
                messages.append({"role": "assistant", "content": "".join(round_text)})
            fresh = 0
            for c in calls:
                fn = c.get("function") or {}
                name, args = fn.get("name", ""), dict(fn.get("arguments") or {})
                # "What time is it?" / "Weather here?": fill in the user's own place.
                if name == "get_time" and not (args.get("timezone") or args.get("tz")) and ctx.get("timezone"):
                    args["timezone"] = ctx["timezone"]
                if name == "get_weather" and not args.get("location") and ctx.get("location"):
                    args["location"] = ctx["location"]
                key = f"{name}:{json.dumps(args, sort_keys=True)}"
                if key in executed:
                    continue  # its result is already in the messages
                executed.add(key)
                fresh += 1
                yield {"type": "tool", "name": name, "args": args}
                result = await run_and_record(name, args)
                yield {"type": "tool_result", "name": name, "ok": result.ok}
            if not fresh:
                # Only repeats: the model is looping. Make it answer with what it has.
                final_round = True

    reply = "".join(reply_parts).strip()
    db.log_message(role="assistant", content=reply, conversation_id=cid)
    timing["total_ms"] = _ms(started)

    # Fact extraction is a second LLM call nobody is waiting on; run it after the reply.
    # Facts come from what the user said, not from tool output: a get_time for
    # America/New_York once became "location: New York, NY".
    task = asyncio.create_task(asyncio.to_thread(_extract_facts, db, user_prompt, reply, None))
    _background.add(task)
    task.add_done_callback(_background.discard)

    yield {"type": "done", "reply": reply, "timing": timing}
