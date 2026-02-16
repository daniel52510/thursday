from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from tools import ToolCall


class AgentResponse(BaseModel):
    reply: str
    tool_calls: list[ToolCall] = Field(default_factory=list)

class ExtractedFact(BaseModel):
    """A single stable fact worth persisting."""

    key: str
    value: Any
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: Literal["explicit_user", "assistant_inference", "tool_result"] = "explicit_user"


class FactExtraction(BaseModel):
    """Output shape from the fact-extractor LLM pass."""

    facts: list[ExtractedFact] = Field(default_factory=list)