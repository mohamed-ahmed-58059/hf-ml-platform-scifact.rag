"""Schemas for LLM-generated Q&A: what the agent must emit for the EvalGenerator."""
from pydantic import BaseModel


class QAItem(BaseModel):
    question: str
    answer: str
    chunk_ids: list[str]


class QAList(BaseModel):
    items: list[QAItem]
