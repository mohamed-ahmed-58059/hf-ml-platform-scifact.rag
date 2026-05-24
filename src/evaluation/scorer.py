import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"ranx.*")

from typing import Optional  # noqa: E402

from langchain_core.runnables import Runnable  # noqa: E402
from langchain_core.runnables.config import RunnableConfig  # noqa: E402
from ranx import Qrels, Run, evaluate  # noqa: E402

from src.model.rag_context import RagContext  # noqa: E402

DEFAULT_METRICS = ["ndcg@10", "recall@5", "recall@10", "mrr@10", "map@10", "hit_rate@10"]


class Scorer(Runnable[list[RagContext], list[RagContext]]):

    def __init__(self, metrics: list[str] = None):
        self.metrics = metrics if metrics is not None else DEFAULT_METRICS

    def invoke(
        self,
        input: list[RagContext],
        config: Optional[RunnableConfig] = None,
    ) -> list[RagContext]:
        qrels_dict: dict[str, dict[str, int]] = {}
        run_dict: dict[str, dict[str, float]] = {}

        for ctx in input:
            if not ctx.item.expected_doc_ids:
                continue
            qrels_dict[ctx.item.query_id] = {d: 1 for d in ctx.item.expected_doc_ids}

            doc_scores: dict[str, float] = {}
            for chunk in ctx.retrieved:
                if chunk.doc_id not in doc_scores or chunk.score > doc_scores[chunk.doc_id]:
                    doc_scores[chunk.doc_id] = float(chunk.score)
            if doc_scores:
                run_dict[ctx.item.query_id] = doc_scores

        shared = sorted(set(qrels_dict) & set(run_dict))
        if not shared:
            return input

        qrels_filt = {q: qrels_dict[q] for q in shared}
        run_filt = {q: run_dict[q] for q in shared}

        per_query = evaluate(
            Qrels(qrels_filt),
            Run(run_filt),
            self.metrics,
            return_mean=False,
        )
        # Normalize to dict[metric, array] when a single metric was passed
        if not isinstance(per_query, dict):
            per_query = {self.metrics[0]: per_query}

        # Map per-query arrays back to each context
        ctx_by_qid = {ctx.item.query_id: ctx for ctx in input}
        for metric, arr in per_query.items():
            for qid, score in zip(shared, arr):
                if qid in ctx_by_qid:
                    ctx_by_qid[qid].retrieval_scores[metric] = float(score)

        return input
