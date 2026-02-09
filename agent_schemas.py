from pydantic import BaseModel, Field
from tools import ToolCall

class AgentResponse(BaseModel):
    reply: str
    tool_calls: list[ToolCall] = Field(default_factory=list)

AgentResponse.model_validate({
  "reply": "…",
  "tool_calls": [
    {"name": "get_time", "args": {}}
  ]
})