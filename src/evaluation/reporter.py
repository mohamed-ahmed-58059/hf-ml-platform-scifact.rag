from collections import defaultdict
from typing import Optional

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from tabulate import tabulate

from src.model.rag_context import RagContext


JUDGE_METRICS = ("faithfulness", "relevance", "accuracy", "completeness")


class Reporter(Runnable[list[RagContext], list[RagContext]]):

    def invoke(
        self,
        input: list[RagContext],
        config: Optional[RunnableConfig] = None,
    ) -> list[RagContext]:
        retrieval_buckets: dict[str, list[float]] = defaultdict(list)
        judge_buckets: dict[str, list[float]] = defaultdict(list)

        for ctx in input:
            for k, v in ctx.retrieval_scores.items():
                retrieval_buckets[k].append(v)
            for k, v in ctx.judge_scores.items():
                judge_buckets[k].append(v)

        rows = []
        for metric in sorted(retrieval_buckets):
            vals = retrieval_buckets[metric]
            rows.append(["Retrieval", metric, f"{sum(vals) / len(vals):.4f}", len(vals)])

        for metric in JUDGE_METRICS:
            vals = judge_buckets.get(metric, [])
            score = f"{sum(vals) / len(vals):.4f}" if vals else "-"
            n = len(vals) if vals else "-"
            rows.append(["Generation", metric, score, n])

        print("\n" + tabulate(
            rows,
            headers=["Category", "Metric", "Score", "n"],
            tablefmt="grid",
        ))
        return input
