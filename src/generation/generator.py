"""Generates an answer for each RagContext via an LLMClientGroup (text mode).

Mid-batch resumable: pass cache_path to write {query_id, answer} as each result lands.
On subsequent runs, cached query_ids are skipped.
"""
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from src.generation.llm_client_group import LLMClientGroup
from src.model.rag_context import RagContext
from src.prompt.generation import GENERATION_SYSTEM, GENERATION_USER
from src.storage.jsonl_checkpoint import JsonlCheckpoint


class Generator(Runnable[list[RagContext], list[RagContext]]):
    """Fills ctx.answer for each context. Caches per query_id when cache_path is set."""

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
        cached = self._load_cached_answers()
        for ctx in input:
            if ctx.item.query_id in cached:
                ctx.answer = cached[ctx.item.query_id]
        pending = [c for c in input if not c.answer]
        print(f"[Generator] {len(cached)} cached, {len(pending)} to generate")

        if pending:
            prompts = [self._build_prompt(c) for c in pending]
            on_result = self._make_on_result(pending) if self.checkpoint.enabled() else None
            answers = self.llm_group.batch_text(prompts, on_result=on_result)
            for ctx, ans in zip(pending, answers):
                if ans is not None:
                    ctx.answer = ans.strip()

        n_ok = sum(1 for c in input if c.answer)
        print(f"[Generator] Produced {n_ok}/{len(input)} answers")
        return input

    def _load_cached_answers(self) -> dict[str, str]:
        return {row["query_id"]: row["answer"] for row in self.checkpoint.load()}

    def _make_on_result(self, pending: list[RagContext]):
        def on_result(idx: int, answer: str) -> None:
            self.checkpoint.append({
                "query_id": pending[idx].item.query_id,
                "answer": answer.strip(),
            })
        return on_result

    def _build_prompt(self, ctx: RagContext) -> list:
        context = "\n".join(
            f"[{c.id}] {c.text[:self.max_chunk_chars]}"
            for c in ctx.retrieved[:self.top_n]
        )
        return [
            SystemMessage(content=GENERATION_SYSTEM),
            HumanMessage(content=GENERATION_USER.format(
                context=context,
                question=ctx.item.query_text,
            )),
        ]
