"""LLM-as-judge over RagContexts. One metric per prompt for better discrimination.

For each item, fires 4 separate LLM calls (one per metric: faithfulness, relevance,
accuracy, completeness). Cache key is (query_id, metric) so resume happens at the
per-metric level.
"""
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from src.generation.llm_client_group import LLMClientGroup
from src.model.judgment import MetricScore
from src.model.rag_context import RagContext
from src.prompt.judgment import JUDGMENT_SYSTEM, JUDGMENT_USER, METRIC_RUBRICS
from src.storage.jsonl_checkpoint import JsonlCheckpoint


_METRICS = ("faithfulness", "relevance", "accuracy", "completeness")


class Judge(Runnable[list[RagContext], list[RagContext]]):
    """Scores items that have BOTH answer + reference_answer, one metric per LLM call."""

    def __init__(
        self,
        llm_group: LLMClientGroup,
        top_n: int = 10,
        max_chunk_chars: int = 600,
        cache_path: Optional[str] = None,
    ):
        self.llm_group = llm_group
        self.top_n = top_n
        self.max_chunk_chars = max_chunk_chars
        self.checkpoint = JsonlCheckpoint(cache_path)

    def invoke(
        self,
        input: list[RagContext],
        config: Optional[RunnableConfig] = None,
    ) -> list[RagContext]:
        scorable = [c for c in input if c.answer and c.item.reference_answer]
        if not scorable:
            print("[Judge] No items with both answer and reference_answer; skipping")
            return input

        cached = self._load_cached_scores()
        for ctx in scorable:
            for m in _METRICS:
                key = (ctx.item.query_id, m)
                if key in cached:
                    ctx.judge_scores[m] = cached[key]

        # Each (item × missing metric) becomes one LLM call
        pending: list[tuple[RagContext, str]] = []
        for ctx in scorable:
            for m in _METRICS:
                if m not in ctx.judge_scores:
                    pending.append((ctx, m))
        print(f"[Judge] {len(cached)} (query,metric) cached, {len(pending)} to judge")

        if pending:
            prompts = [self._build_prompt(ctx, m) for ctx, m in pending]
            on_result = self._make_on_result(pending) if self.checkpoint.enabled() else None
            results = self.llm_group.batch_structured(
                prompts,
                schema_cls=MetricScore,
                validate_fn=lambda s, _: 0.0 <= s.score <= 1.0,
                on_result=on_result,
            )
            for (ctx, metric), res in zip(pending, results):
                if res is not None:
                    ctx.judge_scores[metric] = res.score

        # Coverage stats
        total_pairs = len(scorable) * len(_METRICS)
        scored_pairs = sum(1 for c in scorable for m in _METRICS if m in c.judge_scores)
        print(f"[Judge] Scored {scored_pairs}/{total_pairs} (query,metric) pairs")
        return input

    def _load_cached_scores(self) -> dict[tuple[str, str], float]:
        return {
            (row["query_id"], row["metric"]): row["score"]
            for row in self.checkpoint.load()
        }

    def _make_on_result(self, pending: list[tuple[RagContext, str]]):
        def on_result(idx: int, parsed: MetricScore) -> None:
            ctx, metric = pending[idx]
            self.checkpoint.append({
                "query_id": ctx.item.query_id,
                "metric": metric,
                "score": parsed.score,
            })
        return on_result

    def _build_prompt(self, ctx: RagContext, metric: str) -> list:
        context = "\n".join(
            f"[{c.id}] {c.text[:self.max_chunk_chars]}"
            for c in ctx.retrieved[:self.top_n]
        )
        return [
            SystemMessage(content=JUDGMENT_SYSTEM),
            HumanMessage(content=JUDGMENT_USER.format(
                question=ctx.item.query_text,
                context=context,
                generated_answer=ctx.answer,
                reference_answer=ctx.item.reference_answer,
                metric_name=metric,
                rubric=METRIC_RUBRICS[metric],
            )),
        ]
