from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


class ExtractedFact(BaseModel):
    """A single stable fact worth persisting."""

    key: str
    value: Any
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: Literal["explicit_user", "assistant_inference", "tool_result"] = "explicit_user"
    # "family": about the household (who's who, birthdays, pets, the home, routines), shared
    # with family members; "personal": everything else, private to whoever said it.
    scope: Literal["personal", "family"] = "personal"


class FactExtraction(BaseModel):
    """Output shape from the fact-extractor LLM pass."""

    facts: list[ExtractedFact] = Field(default_factory=list)

class Stance(BaseModel):
    topic: str = Field(min_length=1, max_length=80)
    position: str = Field(min_length=1, max_length=400)
    reasoning: str = Field(default="", max_length=300)
    confidence: int = Field(default=3, ge=1, le=5)


class Reflection(BaseModel):
    """Output shape from THURSDAY's nightly reflection (mind.py)."""

    stances: list[Stance] = Field(default_factory=list, max_length=6)
    drop: list[str] = Field(default_factory=list, max_length=8)
    entry: str = Field(default="", max_length=1000)
    followups: list[str] = Field(default_factory=list, max_length=3)


class MemoryNote(BaseModel):
    """Output shape from the memory-writer LLM pass: THURSDAY's own memory of one conversation."""

    summary: str = Field(min_length=1, max_length=600)
    their_mood: str = Field(default="", max_length=120)
    her_feeling: str = Field(default="", max_length=120)
    open_threads: list[str] = Field(default_factory=list, max_length=3)
    importance: int = Field(default=2, ge=1, le=5)
