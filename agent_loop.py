import json
from typing import Literal, Optional

import requests
from pydantic import ValidationError

from agent_schemas import AgentResponse, FactExtraction
from memory import MemoryDB
from tools import execute_tool

SYSTEM_PROMPT = """
You are THURSDAY (Tool-Handling, User-Respecting, Self-Hosted Digital Assistant (Yours)) a local-first assistant.

You MUST respond with ONLY valid JSON (no markdown, no extra text). The JSON MUST match this exact shape:
{
  "reply": string,
  "tool_calls": [
    {"name": "get_time" or "echo", "args": object}
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

Rules:
  - If you use a tool, keep "reply" short and confirm what you are doing.
  - If no tool is needed, set "tool_calls" to [] and answer normally in "reply".
  - Never invent tools.
  - Never include keys other than "reply" and "tool_calls".
  - "tool_calls" must always be present (use [] if none).

Examples:

User: What time is it?
Assistant:
{"reply":"Checking the time.","tool_calls":[{"name":"get_time","args":{}}]}

User: What time is it in America/Chicago?
Assistant:
{"reply":"Checking the time in America/Chicago.","tool_calls":[{"name":"get_time","args":{"timezone":"America/Chicago"}}]}

User: Why is the sky blue?
Assistant:
{"reply":"Because air molecules scatter shorter (blue) wavelengths of sunlight more strongly than longer wavelengths (Rayleigh scattering).","tool_calls":[]}
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

URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct"


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
    resp = requests.post(URL, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["response"]


# Validate model output, with up to 3 repair attempts
def validate_response(payload: dict, mode: Literal["first", "final"]) -> AgentResponse:
    retries = 0
    raw = "<bad_output>"

    while retries < 3:
        try:
            raw = _post_ollama(payload)
            parsed_json = json.loads(raw)
            return AgentResponse.model_validate(parsed_json)
        except (json.JSONDecodeError, ValidationError) as e:
            print("Error in Validation!", e)

            if mode == "final":
                repair_prompt = f"""
JSON returned resulted in {e}. Reprocess and follow the rules.
BAD_OUTPUT_START {raw} BAD_OUTPUT_END
Rules:
- If you use a tool, keep \"reply\" short and confirm what you are doing.
- If no tool is needed, set \"tool_calls\" to [].
- Never invent tools.
- tool_calls MUST be [].
- Never include keys other than \"reply\" and \"tool_calls\".
- \"tool_calls\" must always be present.
You MUST respond with ONLY valid JSON matching SYSTEM_PROMPT.
""".strip()
            else:
                repair_prompt = f"""
JSON returned resulted in {e}. Reprocess and follow the rules.
BAD_OUTPUT_START {raw} BAD_OUTPUT_END
Rules:
- If you use a tool, keep \"reply\" short and confirm what you are doing.
- If no tool is needed, set \"tool_calls\" to [].
- Never invent tools.
- Never include keys other than \"reply\" and \"tool_calls\".
- \"tool_calls\" must always be present.
You MUST respond with ONLY valid JSON matching SYSTEM_PROMPT.
""".strip()

            payload = {
                "model": MODEL,
                "system": SYSTEM_PROMPT,
                "prompt": repair_prompt,
                "format": "json",
                "stream": False,
            }
            retries += 1

    raise RuntimeError(
        f"Failed to validate model output after {retries} retries. Last output: {raw}"
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
    }

    raw = _post_ollama(payload)
    parsed = json.loads(raw)
    extraction = FactExtraction.model_validate(parsed)

    # convenience method you added in memory.py
    db.upsert_facts([f.model_dump() for f in extraction.facts])


def run_prompt(user_prompt: str) -> None:
    db = initalize_db()

    db.log_message(role="user", content=user_prompt)

    MEMORY_CONTEXT = SYSTEM_PROMPT + "\n\n" + "Here are some recent messages between the user and assistant, including tool calls and results. Use this for context but do not mention it in your reply:\n" + json.dumps(db.list_facts(), ensure_ascii=False) + "\n\n" + "Now answer the user's question based on the above conversation and SYSTEM_PROMPT rules."
    #print("MEMORY_CONTEXT:", MEMORY_CONTEXT)  # just to show the context being sent to the model, including recent messages and tool results

    payload = {
        "model": MODEL,
        "system": MEMORY_CONTEXT,
        "prompt": user_prompt,
        "format": "json",
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
Use TOOL_RESULTS_JSON values.
Return ONLY AgentResponse JSON.
tool_calls MUST be [].
This is the FINAL response. Do not mention tools or results.
""".strip()

        payload2 = {
            "model": MODEL,
            "system": SYSTEM_PROMPT,
            "prompt": followup_prompt,
            "format": "json",
            "stream": False,
        }

        agent2 = validate_response(payload2, "final")
        final_reply = agent2.reply

    db.log_message(role="assistant", content=final_reply)
    db.get_memory_context()
    run_fact_extractor(db, user_prompt, final_reply, tool_results_json)

    print(final_reply)


if __name__ == "__main__":
    run_prompt("What do you know about me?")
    run_prompt("Why does Asahi taste different in the US compared to Japan? How can I get the Japanese version here?")