from pydantic import BaseModel

from src.model.eval_item import EvalItem
from src.model.retrieved_chunk import RetrievedChunk


class RagContext(BaseModel):
    """State container that flows through the rag chain.

    Same shape for prod (item.query_id='0', expected/answer empty) and eval
    (item populated from JSONL). Stages read and fill in fields as they run.
    """
    item: EvalItem
    rewritten_queries: list[str] = []
    query_vectors: dict[str, list[float]] = {}
    retrieved: list[RetrievedChunk] = []
    reranked: list[RetrievedChunk] | None = None
    answer: str | None = None
    retrieval_scores: dict[str, float] = {}
    judge_scores: dict[str, float] = {}
