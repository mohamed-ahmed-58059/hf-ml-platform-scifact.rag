"""Schema for LLM-driven query rewriting."""
from pydantic import BaseModel


class RewrittenQuery(BaseModel):
    rewritten: str
