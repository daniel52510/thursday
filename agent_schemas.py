from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


class ExtractedFact(BaseModel):
    """A single stable fact worth persisting."""

    key: str
    value: Any
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: Literal["explicit_user", "assistant_inference", "tool_result"] = "explicit_user"


class FactExtraction(BaseModel):
    """Output shape from the fact-extractor LLM pass."""

    facts: list[ExtractedFact] = Field(default_factory=list)

class MemoryNote(BaseModel):
    """Output shape from the memory-writer LLM pass: THURSDAY's own memory of one conversation."""

    summary: str = Field(min_length=1, max_length=600)
    their_mood: str = Field(default="", max_length=120)
    her_feeling: str = Field(default="", max_length=120)
    open_threads: list[str] = Field(default_factory=list, max_length=3)
    importance: int = Field(default=2, ge=1, le=5)
