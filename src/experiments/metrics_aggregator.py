"""Reduces a list of RagContexts to a flat {metric_name: mean_value} dict."""
from collections import defaultdict

from src.model.rag_context import RagContext


class MetricsAggregator:
    """Averages retrieval_scores + judge_scores across all RagContexts."""

    def aggregate(self, contexts: list[RagContext]) -> dict[str, float]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for ctx in contexts:
            for k, v in ctx.retrieval_scores.items():
                buckets[k].append(v)
            for k, v in ctx.judge_scores.items():
                buckets[k].append(v)
        return {k: sum(vs) / len(vs) for k, vs in buckets.items() if vs}
