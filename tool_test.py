from tools import ToolCall, execute_tool

print(execute_tool(ToolCall(name="get_time")).model_dump())
print(execute_tool(ToolCall(name="echo", args={"text": "THURSDAY online"})).model_dump())