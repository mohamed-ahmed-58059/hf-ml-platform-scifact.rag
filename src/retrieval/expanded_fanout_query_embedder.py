"""Fan-out embedding over the original query + all expanded variants.

Each (item, embedder) produces 1 + N vectors when expansions are present:
  '{model_id}__orig'    — the original query
  '{model_id}__exp_0'   — first expanded variant
  '{model_id}__exp_1'   — second variant
  ...

VectorStore.search strips '__suffix' to locate the Chroma collection. RRF fuses
across all (model × variant) rankings.
"""
from typing import Optional

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from src.embedding.embedder import Embedder
from src.model.eval_item import EvalItem
from src.model.rag_context import RagContext


class ExpandedFanOutQueryEmbedder(Runnable[list[EvalItem], list[RagContext]]):
    """Per (item, embedder): one vector for the original, one per expanded variant."""

    def __init__(self, embedders: list[Embedder]):
        if not embedders:
            raise ValueError("ExpandedFanOutQueryEmbedder requires at least one embedder")
        self.embedders = embedders

    def invoke(
        self,
        input: list[EvalItem],
        config: Optional[RunnableConfig] = None,
    ) -> list[RagContext]:
        contexts = [RagContext(item=item) for item in input]

        orig_texts = [c.item.query_text for c in contexts]
        # For each variant index, collect (ctx_idx, text) where that variant exists
        max_variants = max((len(c.item.expanded_queries) for c in contexts), default=0)

        for embedder in self.embedders:
            orig_vectors = embedder.embed_queries(orig_texts)
            for ctx, vec in zip(contexts, orig_vectors):
                ctx.query_vectors[f"{embedder.model_id}__orig"] = vec.tolist()

            for variant_idx in range(max_variants):
                targets = []  # (ctx_idx, text)
                for ctx_idx, ctx in enumerate(contexts):
                    if variant_idx < len(ctx.item.expanded_queries):
                        targets.append((ctx_idx, ctx.item.expanded_queries[variant_idx]))
                if not targets:
                    continue
                texts = [t for _, t in targets]
                vectors = embedder.embed_queries(texts)
                for (ctx_idx, _), vec in zip(targets, vectors):
                    contexts[ctx_idx].query_vectors[f"{embedder.model_id}__exp_{variant_idx}"] = vec.tolist()

        return contexts
