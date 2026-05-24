"""Single-query embedding: one vector per (item, embedder).

Uses item.rewritten_query if present, otherwise item.query_text. No fan-out, no fusion.
"""
from typing import Optional

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from src.embedding.embedder import Embedder
from src.model.eval_item import EvalItem
from src.model.rag_context import RagContext


class QueryEmbedder(Runnable[list[EvalItem], list[RagContext]]):
    """For each embedder, encodes one query text per item."""

    def __init__(self, embedders: list[Embedder]):
        if not embedders:
            raise ValueError("QueryEmbedder requires at least one embedder")
        self.embedders = embedders

    def invoke(
        self,
        input: list[EvalItem],
        config: Optional[RunnableConfig] = None,
    ) -> list[RagContext]:
        contexts = [RagContext(item=item) for item in input]
        texts = [c.item.rewritten_query or c.item.query_text for c in contexts]
        for embedder in self.embedders:
            vectors = embedder.embed_queries(texts)
            for ctx, vec in zip(contexts, vectors):
                ctx.query_vectors[embedder.model_id] = vec.tolist()
        return contexts
