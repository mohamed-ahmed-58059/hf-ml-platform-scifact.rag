"""Fan-out embedding: embeds original AND rewritten as separate vectors per embedder.

Adds two vectors per (item, embedder) when rewrite is present: one keyed
'{model_id}__orig', one keyed '{model_id}__rewritten'. VectorStore.search strips
the '__suffix' to locate the right Chroma collection, and RRF fuses across all
rankings (more rankings → better recall in principle).

When an item has no rewrite, only the '__orig' key is added.
"""
from typing import Optional

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from src.embedding.embedder import Embedder
from src.model.eval_item import EvalItem
from src.model.rag_context import RagContext


class FanOutQueryEmbedder(Runnable[list[EvalItem], list[RagContext]]):
    """Per (item, embedder): one vector for the original, one for the rewrite (if present)."""

    def __init__(self, embedders: list[Embedder]):
        if not embedders:
            raise ValueError("FanOutQueryEmbedder requires at least one embedder")
        self.embedders = embedders

    def invoke(
        self,
        input: list[EvalItem],
        config: Optional[RunnableConfig] = None,
    ) -> list[RagContext]:
        contexts = [RagContext(item=item) for item in input]

        orig_texts = [c.item.query_text for c in contexts]
        rewrite_indices = [i for i, c in enumerate(contexts) if c.item.rewritten_query]
        rewrite_texts = [contexts[i].item.rewritten_query for i in rewrite_indices]

        for embedder in self.embedders:
            orig_vectors = embedder.embed_queries(orig_texts)
            for ctx, vec in zip(contexts, orig_vectors):
                ctx.query_vectors[f"{embedder.model_id}__orig"] = vec.tolist()

            if rewrite_texts:
                rewrite_vectors = embedder.embed_queries(rewrite_texts)
                for idx, vec in zip(rewrite_indices, rewrite_vectors):
                    contexts[idx].query_vectors[f"{embedder.model_id}__rewritten"] = vec.tolist()

        return contexts
