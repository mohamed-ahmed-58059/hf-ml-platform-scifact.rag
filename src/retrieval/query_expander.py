"""Expands each EvalItem's query into N variants via an LLMClientGroup.

Mid-batch resumable: pass cache_path to write {query_id, queries} as each result
lands. On subsequent runs, cached query_ids are skipped.

Sets item.expanded_queries in place. ExpandedFanOutQueryEmbedder consumes them.
"""
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from src.generation.llm_client_group import LLMClientGroup
from src.model.eval_item import EvalItem
from src.model.expanded_queries import ExpandedQueries
from src.prompt.query_expansion import QUERY_EXPANSION_SYSTEM, QUERY_EXPANSION_USER
from src.storage.jsonl_checkpoint import JsonlCheckpoint


class QueryExpander(Runnable[list[EvalItem], list[EvalItem]]):
    """Fills item.expanded_queries (length n_variants) for each input item."""

    def __init__(
        self,
        llm_group: LLMClientGroup,
        n_variants: int = 3,
        cache_path: Optional[str] = None,
    ):
        self.llm_group = llm_group
        self.n_variants = n_variants
        self.checkpoint = JsonlCheckpoint(cache_path)

    def invoke(
        self,
        input: list[EvalItem],
        config: Optional[RunnableConfig] = None,
    ) -> list[EvalItem]:
        cached = self._load_cached_expansions()
        for item in input:
            if item.query_id in cached:
                item.expanded_queries = cached[item.query_id]
        pending = [i for i in input if not i.expanded_queries]
        print(f"[QueryExpander] {len(cached)} cached, {len(pending)} to expand")

        if pending:
            prompts = [self._build_prompt(item) for item in pending]
            on_result = self._make_on_result(pending) if self.checkpoint.enabled() else None
            results = self.llm_group.batch_structured(
                prompts,
                schema_cls=ExpandedQueries,
                validate_fn=lambda r, _: len(r.queries) == self.n_variants and all(q.strip() for q in r.queries),
                on_result=on_result,
            )
            for item, res in zip(pending, results):
                if res is not None:
                    item.expanded_queries = [q.strip() for q in res.queries]

        n_done = sum(1 for i in input if i.expanded_queries)
        print(f"[QueryExpander] Expanded {n_done}/{len(input)} queries")
        return input

    def _load_cached_expansions(self) -> dict[str, list[str]]:
        return {row["query_id"]: row["queries"] for row in self.checkpoint.load()}

    def _make_on_result(self, pending: list[EvalItem]):
        def on_result(idx: int, parsed: ExpandedQueries) -> None:
            self.checkpoint.append({
                "query_id": pending[idx].query_id,
                "queries": [q.strip() for q in parsed.queries],
            })
        return on_result

    def _build_prompt(self, item: EvalItem) -> list:
        return [
            SystemMessage(content=QUERY_EXPANSION_SYSTEM),
            HumanMessage(content=QUERY_EXPANSION_USER.format(
                question=item.query_text,
                n_variants=self.n_variants,
            )),
        ]
