"""Rewrites each EvalItem's query into a search-friendly form via an LLMClientGroup.

Mid-batch resumable: pass cache_path to write {query_id, rewritten} as each result
lands. On subsequent runs, cached query_ids are skipped.

Sets item.rewritten_query in place. Downstream embedders (Single/Fused/FanOut)
decide how to use it.
"""
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from src.generation.llm_client_group import LLMClientGroup
from src.model.eval_item import EvalItem
from src.model.rewritten_query import RewrittenQuery
from src.prompt.query_rewrite import QUERY_REWRITE_SYSTEM, QUERY_REWRITE_USER
from src.storage.jsonl_checkpoint import JsonlCheckpoint


class QueryRewriter(Runnable[list[EvalItem], list[EvalItem]]):
    """Fills item.rewritten_query for each input item."""

    def __init__(
        self,
        llm_group: LLMClientGroup,
        cache_path: Optional[str] = None,
    ):
        self.llm_group = llm_group
        self.checkpoint = JsonlCheckpoint(cache_path)

    def invoke(
        self,
        input: list[EvalItem],
        config: Optional[RunnableConfig] = None,
    ) -> list[EvalItem]:
        cached = self._load_cached_rewrites()
        for item in input:
            if item.query_id in cached:
                item.rewritten_query = cached[item.query_id]
        pending = [i for i in input if not i.rewritten_query]
        print(f"[QueryRewriter] {len(cached)} cached, {len(pending)} to rewrite")

        if pending:
            prompts = [self._build_prompt(item) for item in pending]
            on_result = self._make_on_result(pending) if self.checkpoint.enabled() else None
            results = self.llm_group.batch_structured(
                prompts,
                schema_cls=RewrittenQuery,
                validate_fn=lambda r, _: bool(r.rewritten and r.rewritten.strip()),
                on_result=on_result,
            )
            for item, res in zip(pending, results):
                if res is not None:
                    item.rewritten_query = res.rewritten.strip()

        n_rewritten = sum(1 for i in input if i.rewritten_query)
        print(f"[QueryRewriter] Rewrote {n_rewritten}/{len(input)} queries")
        return input

    def _load_cached_rewrites(self) -> dict[str, str]:
        return {row["query_id"]: row["rewritten"] for row in self.checkpoint.load()}

    def _make_on_result(self, pending: list[EvalItem]):
        def on_result(idx: int, parsed: RewrittenQuery) -> None:
            self.checkpoint.append({
                "query_id": pending[idx].query_id,
                "rewritten": parsed.rewritten.strip(),
            })
        return on_result

    def _build_prompt(self, item: EvalItem) -> list:
        return [
            SystemMessage(content=QUERY_REWRITE_SYSTEM),
            HumanMessage(content=QUERY_REWRITE_USER.format(question=item.query_text)),
        ]
