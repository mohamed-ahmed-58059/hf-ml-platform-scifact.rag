"""Cross-encoder reranker. Re-orders ctx.retrieved using (query, chunk) pair scoring.

Standard production pattern: dense retrieval surfaces candidates fast; a small
cross-encoder model scores each (query, candidate) pair properly and reorders.

query_source controls which form of the query the cross-encoder sees:
  "original"  — ctx.item.query_text (default; matches user intent best)
  "rewritten" — ctx.item.rewritten_query if present, else query_text
  "fused"     — "{query_text} {rewritten_query}" if rewrite present, else query_text
Empirical — try all three and let the metrics decide.

Lives on GPU. Designed to share the card with sleeping vLLM (load while vLLM
asleep; ~1 GB residual fits easily).
"""
from typing import Literal, Optional

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from src.model.eval_item import EvalItem
from src.model.rag_context import RagContext


QuerySource = Literal["original", "rewritten", "fused"]


class CrossEncoderReranker(Runnable[list[RagContext], list[RagContext]]):
    """Re-orders ctx.retrieved, keeping the top_n highest cross-encoder scores."""

    def __init__(
        self,
        model: CrossEncoder,
        top_n: int = 10,
        batch_size: int = 32,
        query_source: QuerySource = "original",
    ):
        self.model = model
        self.top_n = top_n
        self.batch_size = batch_size
        self.query_source = query_source

    def invoke(
        self,
        input: list[RagContext],
        config: Optional[RunnableConfig] = None,
    ) -> list[RagContext]:
        for ctx in tqdm(input, desc="Reranking"):
            if not ctx.retrieved:
                continue
            query = self._query_for(ctx.item)
            pairs = [(query, c.text) for c in ctx.retrieved]
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            scored = sorted(zip(scores, ctx.retrieved), key=lambda sc: -float(sc[0]))
            top = scored[:self.top_n]
            for s, chunk in top:
                chunk.score = float(s)
            ctx.retrieved = [c for _, c in top]
        return input

    def _query_for(self, item: EvalItem) -> str:
        if self.query_source == "rewritten":
            return item.rewritten_query or item.query_text
        if self.query_source == "fused" and item.rewritten_query:
            return f"{item.query_text} {item.rewritten_query}"
        return item.query_text

