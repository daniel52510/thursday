from pydantic import BaseModel
from tools import ToolCall

class AgentResponse(BaseModel):
    reply: str
    tool_calls: list[ToolCall] = []

AgentResponse.model_validate({...})