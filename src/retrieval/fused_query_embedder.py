"""Fused-query embedding: concatenates original + rewritten into one text per item.

Falls back to query_text alone when no rewrite is present.
"""
from typing import Optional

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from src.embedding.embedder import Embedder
from src.model.eval_item import EvalItem
from src.model.rag_context import RagContext


class FusedQueryEmbedder(Runnable[list[EvalItem], list[RagContext]]):
    """For each embedder, encodes f'{original} {rewritten}' (or original alone) per item."""

    def __init__(self, embedders: list[Embedder]):
        if not embedders:
            raise ValueError("FusedQueryEmbedder requires at least one embedder")
        self.embedders = embedders

    def invoke(
        self,
        input: list[EvalItem],
        config: Optional[RunnableConfig] = None,
    ) -> list[RagContext]:
        contexts = [RagContext(item=item) for item in input]
        texts = [self._fused_text(c.item) for c in contexts]
        for embedder in self.embedders:
            vectors = embedder.embed_queries(texts)
            for ctx, vec in zip(contexts, vectors):
                ctx.query_vectors[embedder.model_id] = vec.tolist()
        return contexts

    def _fused_text(self, item: EvalItem) -> str:
        if item.rewritten_query:
            return f"{item.query_text} {item.rewritten_query}"
        return item.query_text
