from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Literal, Optional
from pydantic import BaseModel, Field

#tool contract
class ToolCall(BaseModel):
    name: Literal["get_time", "echo"]
    args: Dict[str, Any] = Field(default_factory=dict)

#tool contract results
class ToolResult(BaseModel):
    ok: bool
    tool_name: str
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

#Tool Result Function to Get Time
def get_time(_:Dict[str, Any]) -> ToolResult:
    now = datetime.now(timezone.utc).isoformat()
    return ToolResult(ok=True, tool_name="get_time", data={"utc": now})

#Tool Result to echo text
def echo(args: Dict[str, Any]) -> ToolResult:
    text = str(args.get("text", ""))
    return ToolResult(ok=True, tool_name="echo", data={"text": text})

#Registry (Central Allowlist for Thursday)
TOOLS: Dict[str, Callable[[Dict[str, Any]], ToolResult]] = {
    "get_time": get_time,
    "echo": echo,
}

def execute_tool(call: ToolCall) -> ToolResult:
    fn = TOOLS.get(call.name)
    if fn is None:
        return ToolResult(ok=False, tool_name=call.name, error="unknown_tool")
    try:
        return fn(call.args)
    except Exception as e:
        return ToolResult(ok=False, tool_name=call.name, error=f"tool_exception: {e}")