import requests
import json
from pydantic import ValidationError
from agent_schemas import AgentResponse
from tools import execute_tool

SYSTEM_PROMPT= """
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

URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct"

# we are taking in parameters parsed_json and return an agent with a validated schema. The model will try to validate 2-3 times 
def validate_response(payload, mode):
    retries=0
    raw = "<bad_output>"
    resp = requests.post(URL, json=payload, timeout=60)
    resp.raise_for_status()
    while(retries < 3):
        try:
            raw = resp.json()["response"]
            parsed_json = json.loads(raw)
            agent = AgentResponse.model_validate(parsed_json)
            return agent
        except (json.JSONDecodeError, ValidationError) as e:
            print("Error in Validation!")
            if(mode == "final"):
                repair_prompt = f"""
                JSON returned resulted in {e}. Since this is the case, reprocess the prompt and remember the following rules.
                BAD_OUTPUT_START {raw} BAD_OUTPUT_END
                Rules:
                - If you use a tool, keep "reply" short and confirm what you are doing.
                - If no tool is needed, set "tool_calls" to [] and answer normally in "reply".
                - Never invent tools.
                - tool_calls MUST BE []
                - Never include keys other than "reply" and "tool_calls".
                - "tool_calls" must always be present (use [] if none).
                        You MUST respond with ONLY  valid JSON (no markdown, no extra text). The JSON MUST match the same shape as in the SYSTEM_PROMPT.
                """.strip()
            else:
                repair_prompt = f"""
                JSON returned resulted in {e}. Since this is the case, reprocess the prompt and remember the following rules.
                BAD_OUTPUT_START {raw} BAD_OUTPUT_END
                Rules:
                - If you use a tool, keep "reply" short and confirm what you are doing.
                - If no tool is needed, set "tool_calls" to [] and answer normally in "reply".
                - Never invent tools.
                - Never include keys other than "reply" and "tool_calls".
                - "tool_calls" must always be present (use [] if none).
                - You MUST respond with ONLY  valid JSON (no markdown, no extra text). The JSON MUST match the same shape as in the SYSTEM_PROMPT.
                """.strip()
            #new payload created with repair_prompt
            payload = {
            "model": MODEL,
            "system": SYSTEM_PROMPT,
            "prompt": repair_prompt,
            "format": "json",
            "stream": False,
            }
            #resp is sending a request to the API for
            resp = requests.post(URL, json=payload, timeout=60)
            resp.raise_for_status()
            retries += 1

def run_prompt(user_prompt: str):
    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": user_prompt,
        "format": "json",
        "stream": False,
    }
    agent = validate_response(payload,"first")
    results = [execute_tool(call) for call in agent.tool_calls]
    print("Results: ", results)
    if agent.tool_calls:
      tool_results_json = [r.model_dump() for r in results]
      followup_prompt = f"""
      ORIGINAL_USER_QUESTION:
      {user_prompt}
        TOOL_RESULTS_JSON:
        {json.dumps(tool_results_json)}
        TASK:
        Write the final answer to the ORIGINAL_USER_QUESTION for the user.
        Use TOOL_RESULTS_JSON values.
        Return ONLY AgentResponse JSON.
        tool_calls MUST be [].
        reply MUST be a direct answer (no meta phrases like "based on the results").
        This is the FINAL response. Do not mention tools or results. Just answer.
      """.strip()
      payload2 = {
          "model": MODEL,
          "system": SYSTEM_PROMPT,
          "prompt": followup_prompt,
          "format": "json",
          "stream": False,
          }
      agent2 = validate_response(payload2,"final")
      print(agent2.reply)
    #print(json.dumps(payload2, indent=2, sort_keys=True))
    else:
        print(agent.reply)
        #print(json.dumps(payload, indent=2, sort_keys=True))


run_prompt("How big is Venus?")
 