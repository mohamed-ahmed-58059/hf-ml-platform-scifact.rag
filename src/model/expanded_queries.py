"""Schema for LLM-driven query expansion (N diverse variants)."""
from pydantic import BaseModel


class ExpandedQueries(BaseModel):
    queries: list[str]
