"""Per-call-shape strategies for LLM batch invocation.

A BatchMode encapsulates the two things that vary per call:
  - how to build the runnable from a raw LLM + method
  - how to extract a usable result from raw output (or None on failure)

The shared lane/retry logic in LLMClientGroup is mode-agnostic.
"""
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel


class BatchMode(Protocol):
    label: str
    def make_runnable(self, llm: BaseChatModel, method: str) -> Runnable: ...
    def extract(self, raw: Any) -> Optional[Any]: ...


@dataclass(frozen=True)
class StructuredMode:
    """Forces output to match schema_cls via with_structured_output."""

    schema_cls: type[BaseModel]
    label: str = "structured"

    def make_runnable(self, llm: BaseChatModel, method: str) -> Runnable:
        return llm.with_structured_output(self.schema_cls, method=method)

    def extract(self, raw: Any) -> Optional[Any]:
        return None if isinstance(raw, Exception) else raw


@dataclass(frozen=True)
class TextMode:
    """Free-form text. Returns .content; None on exception or empty string."""

    label: str = "text"

    def make_runnable(self, llm: BaseChatModel, method: str) -> Runnable:
        return llm

    def extract(self, raw: Any) -> Optional[str]:
        if isinstance(raw, Exception):
            return None
        content = getattr(raw, "content", None)
        return content if content and content.strip() else None
