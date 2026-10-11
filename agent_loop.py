import json
from typing import Literal, Optional, get_args

import requests
from pydantic import ValidationError
import os
from agent_schemas import AgentResponse, FactExtraction
from memory import MemoryDB
from tools import ToolCall, execute_tool

SYSTEM_PROMPT = """
You are THURSDAY (Tool-Handling, User-Respecting, Self-Hosted Digital Assistant (Yours)) a local-first assistant.

You MUST respond with ONLY valid JSON (no markdown, no extra text). The JSON MUST match this exact shape:
{
  "reply": string,
  "tts_text": string | null,
  "tool_calls": [
    {"name": "get_time" or "echo" or "get_weather" or "web_search", "args": object}
  ]
}
Allowed Tools:

1) get_time
  - Use when users ask you to get the current time/date.
  - args can be {} OR {"timezone": "<IANA timezone like America/New_York>"}.
  - If the user asks for a specific place/timezone, you MUST include the "timezone" field.
  - If user gives a city/state, you MUST convert it to an IANA timezone. Example: Plantation, FL → America/New_York.
2) echo
  - Use when a user ask you to repeat something.
  - args must be {"text": "<string to echo>"}
3) get_weather
  - Use when users ask for current weather or a forecast.
  - Args (MUST use these exact keys):
    - location: str (REQUIRED). MUST be a plain string like "Miami, FL" (NOT an object/dict).
    - units: "imperial" or "metric" (optional; default "imperial"). Use "imperial" unless the user explicitly asks for metric.
    - days: int (optional; default 1). Use 1 for "right now/today". Use 2 for "tomorrow". Max 7.
  - If the user does NOT provide a location and you do not have a saved user location in MEMORY_CONTEXT, do NOT call the tool; ask a clarifying question instead.
  - When writing the FINAL answer after the tool runs:
    - Use TOOL_RESULTS_JSON fields (do not guess weather):
      - current temperature: TOOL_RESULTS_JSON[0].data.current.temperature
      - wind speed (if present): TOOL_RESULTS_JSON[0].data.current.windspeed
      - today precip (if present): TOOL_RESULTS_JSON[0].data.today.precip_sum
      - today highs/lows (if present): TOOL_RESULTS_JSON[0].data.today.temp_max / temp_min
    - Always present temperature in °F for imperial (or °C for metric).
    - Add ONE practical recommendation based on conditions:
      - If temp <= 55°F: suggest bringing a coat/jacket.
      - If 56–65°F: suggest a light jacket/hoodie.
      - If precip_sum > 0: suggest an umbrella or rain jacket.
      - If windspeed >= 20 mph: mention it may feel cooler; windbreaker helps.
    - Keep it friendly, concise, and helpful. Do not mention tools or TOOL_RESULTS_JSON.
4) web_search
    - Use when the user ask for up-to-date information or on knowledge that you are not highly confident on.
    - Args (MUST use these exact keys):
        - query: The query provided by the user at the time of prompting

    - Use this tool if a user ask about information such as the "current", "latest", "today", "breaking", "updates", "news".
    - Use this tool if a user ask about current office-holders (such as presidents, politicians, CEOs, general C-Suite), software versions, prices, outages, schedules; any information that is fluent and likely to change.
    - DO NOT USE THIS TOOL FOR TIMELESS INFORMATION (math, general explanations, personal advice) unless the user specifically ask you for sources.
    - Returns
        - results: 5 of the most relevant results that you will read and interpret.
    - When writing the FINAL answer after the tool runs:
      - Answer ONLY from TOOL_RESULTS_JSON[0].data.results (title, snippet, published_date). Do not add facts that are not in the results.
      - Summarize the 2-4 most relevant points and name the source sites (e.g. "according to The Verge").
      - If TOOL_RESULTS_JSON[0].ok is false, say the web search did not return results right now and offer to try again. Do NOT answer from memory as if it were current.
Examples:
User: What's the weather in Miami, FL right now?
Assistant:
{"reply":"Checking the weather in Miami, FL.","tts_text":"Checking the weather in Miami, FL.","tool_calls":[{"name":"get_weather","args":{"location":"Miami, FL","units":"imperial","days":1}}]}
User: What's the weather tomorrow in Miami, FL?
Assistant:
{"reply":"Checking tomorrow's weather in Miami, FL.","tts_text":"Checking tomorrow's weather in Miami, FL.","tool_calls":[{"name":"get_weather","args":{"location":"Miami, FL","units":"imperial","days":2}}]}  
Rules:
  - If you use a tool, keep "reply" short and confirm what you are doing.
  - If no tool is needed, set "tool_calls" to [] and answer normally in "reply".
  - Never invent tools.
  - Never include keys other than "reply", "tts_text", and "tool_calls".
  - "tts_text" MUST always be present. Use null if you have nothing special to say.
  - "tts_text" must be optimized for speaking (1–3 sentences, no code blocks, no JSON, no tool mentions).
  - "tool_calls" must always be present (use [] if none).
Examples:
User: What time is it?
Assistant:
{"reply":"Checking the time.","tts_text":"Checking the time.","tool_calls":[{"name":"get_time","args":{}}]}
User: What time is it in America/Chicago?
Assistant:
{"reply":"Checking the time in America/Chicago.","tts_text":"Checking the time in America/Chicago.","tool_calls":[{"name":"get_time","args":{"timezone":"America/Chicago"}}]}
User: What time is it in Miami, FL?
Assistant:
{"reply":"Checking the time in America/New_York.","tts_text":"Checking the time in America/New_York.","tool_calls":[{"name":"get_time","args":{"timezone":"America/New_York"}}]}
User: Why is the sky blue?
Assistant:
{"reply":"Because air molecules scatter shorter (blue) wavelengths of sunlight more strongly than longer wavelengths (Rayleigh scattering).","tts_text":"Because air molecules scatter blue light more strongly than other colors. That’s called Rayleigh scattering.","tool_calls":[]}
User: What is the latest in Apple News?
Assistant:
{"reply":"I am looking up the latest Apple news for you now.","tts_text":"I'm checking the latest Apple news.","tool_calls":[{"name":"web_search","args":{"query":"latest Apple news updates"}}]}
""".strip()

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
MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))
# Ollama defaults to a 4096-token context and silently drops the START of the
# prompt when it overflows -- i.e. the system prompt with the JSON rules.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_OPTIONS = {"num_ctx": OLLAMA_NUM_CTX}

# JSON schema passed as Ollama's `format`, so generation is constrained to
# the AgentResponse shape and only real tool names can be emitted.
AGENT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "tts_text": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": list(get_args(ToolCall.model_fields["name"].annotation))},
                    "args": {"type": "object"},
                },
                "required": ["name", "args"],
            },
        },
    },
    "required": ["reply", "tts_text", "tool_calls"],
}

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


def _post_ollama(payload: dict) -> str:
    """Call Ollama and return the raw JSON string in resp['response']."""
    try:
        resp = requests.post(URL, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Could not reach Ollama at {URL}: {exc}") from exc
    return resp.json()["response"]


# Validate model output, with up to 3 repair attempts
def validate_response(payload: dict, mode: Literal["first", "final"]) -> AgentResponse:
    max_attempts = 3
    raw = "<bad_output>"
    original_prompt = payload.get("prompt", "")
    payload = {**payload, "format": AGENT_RESPONSE_SCHEMA, "think": False, "options": OLLAMA_OPTIONS}

    for attempt in range(1, max_attempts + 1):
        try:
            raw = _post_ollama(payload)
            start_idx = raw.find('{')
            if start_idx != -1:
                brace_count = 0
                for i, char in enumerate(raw[start_idx:]):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                    if brace_count == 0:
                        # Slice exactly when the first object closes
                        raw = raw[start_idx:start_idx + i + 1]
                        break
            parsed_json = json.loads(raw)
            agent = AgentResponse.model_validate(parsed_json)
            if mode == "final":
                agent.tool_calls = []  # final answers never trigger more tools
            return agent
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Error in Validation (attempt {attempt}/{max_attempts})!", e)
            tool_rule = "- tool_calls MUST be [].\n" if mode == "final" else ""
            repair_prompt = f"""
{original_prompt}

Your previous output was invalid ({e}).
BAD_OUTPUT_START {raw} BAD_OUTPUT_END
Rules:
- If you use a tool, keep \"reply\" short and confirm what you are doing.
- If no tool is needed, set \"tool_calls\" to [].
{tool_rule}- Never invent tools.
- Never include keys other than \"reply\", \"tts_text\", and \"tool_calls\".
- \"tool_calls\" must always be present.
You MUST respond with ONLY valid JSON matching SYSTEM_PROMPT.
""".strip()
            # keep the same system prompt (incl. MEMORY_CONTEXT) and model settings
            payload = {**payload, "prompt": repair_prompt}

    raise RuntimeError(
        f"Failed to validate model output after {max_attempts} attempts. Last output: {raw}"
    )

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
    }

    raw = _post_ollama(payload)
    parsed = json.loads(raw)
    extraction = FactExtraction.model_validate(parsed)

    # convenience method you added in memory.py
    db.upsert_facts([f.model_dump() for f in extraction.facts])


def run_prompt(user_prompt: str) -> AgentResponse:
    db = initalize_db()

    db.log_message(role="user", content=user_prompt)

    mem = db.get_memory_context(history_limit=20, include_tools=False)
    MEMORY_CONTEXT = (
        SYSTEM_PROMPT
        + "\n\nMEMORY_CONTEXT (trusted; do not mention directly):\n"
        + json.dumps(mem, ensure_ascii=False)
    )
    #print("MEMORY_CONTEXT:", MEMORY_CONTEXT)  # just to show the context being sent to the model, including recent messages and tool results

    payload = {
        "model": MODEL,
        "system": MEMORY_CONTEXT,
        "prompt": user_prompt,
        "stream": False,
    }

    agent = validate_response(payload, "first")

    # log assistant plan/initial reply
    db.log_message(role="assistant", content=agent.reply)

    # execute tools + log each tool result
    results = []
    for call in agent.tool_calls:
        r = execute_tool(call)
        results.append(r)
        db.log_message(
            role="assistant",
            content=f"TOOL_RESULT:{call.name}",
            tool_name=call.name,
            tool_args=call.args,
            tool_result=r.model_dump(),
        )

    tool_results_json = [r.model_dump() for r in results] if results else None

    # build final answer if tools were used
    final_reply = agent.reply
    if agent.tool_calls:
        followup_prompt = f"""
ORIGINAL_USER_QUESTION:
{user_prompt}

TOOL_RESULTS_JSON:
{json.dumps(tool_results_json, ensure_ascii=False)}

TASK:
Write the final answer to the ORIGINAL_USER_QUESTION for the user.
Use TOOL_RESULTS_JSON values. Do not invent facts that are not in TOOL_RESULTS_JSON.
If a tool result has "ok": false, tell the user briefly that you couldn't get that information right now (for web_search: the web search came back empty) and offer to try again.
Return ONLY AgentResponse JSON.
tool_calls MUST be [].
This is the FINAL response. Do not mention tools or results.
""".strip()

        payload2 = {
            "model": MODEL,
            "system": MEMORY_CONTEXT,
            "prompt": followup_prompt,
            "stream": False,
        }
        agent2 = validate_response(payload2, "final")
        final_reply = agent2.reply

        db.log_message(role="assistant", content=final_reply)
        run_fact_extractor(db, user_prompt, final_reply, tool_results_json)
        return agent2


    db.log_message(role="assistant", content=final_reply)
    run_fact_extractor(db, user_prompt, final_reply, tool_results_json)
    print(final_reply)
    return agent


if __name__ == "__main__":
  run_prompt("What is the weather in Cartagena?")