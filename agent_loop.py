import requests
import json
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
  - Use when users ask you to get the current time/date
  - arg must be {}
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

User: Echo: THURSDAY online
Assistant:
{"reply":"Echoing that.","tool_calls":[{"name":"echo","args":{"text":"THURSDAY online"}}]}

User: Why is the sky blue?
Assistant:
{"reply":"Because air molecules scatter shorter (blue) wavelengths of sunlight more strongly than longer wavelengths (Rayleigh scattering).","tool_calls":[]}
""".strip()

URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct"

def run_prompt(user_prompt: str):
    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": user_prompt,
        "format": "json",
        "stream": False,
    }

    resp = requests.post(URL, json=payload, timeout=60)
    resp.raise_for_status()

    raw = resp.json()["response"]
    #print("RAW:", raw)

    parsed = json.loads(raw)
    agent = AgentResponse.model_validate(parsed)

    results = [execute_tool(call) for call in agent.tool_calls]

    #if results:
        #first = results[0]
        #if first.ok:
            #print("It is:", first.data.get("time"))
        #else:
            #print("Tool error:", first.error)
    #else:
        #print("No tools were called.")

    #print("Reply:", agent.reply)
    #print("Tool calls:", [c.model_dump() for c in agent.tool_calls])
    #print("Tool results:", [r.model_dump() if hasattr(r, "model_dump") else r for r in results])
    #print()
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
      response2 = requests.post(URL, json={
          "model": MODEL,
          "system": SYSTEM_PROMPT,
          "prompt": followup_prompt,
          "format": "json",
          "stream": False,
          }, timeout=60).json()
        
      raw2 = response2["response"]
      agent2 = AgentResponse.model_validate(json.loads(raw2))
      print("FINAL Reply:", agent2.reply)
    else:
        print(agent.reply)


run_prompt("Where am I?")
