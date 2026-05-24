"""Agent-style Q&A generator. Delegates LLM calls to an LLMClientGroup.

Mid-batch resumable: pass cache_path to write {group_key, items} per group as each
result lands. Group key is the "|"-joined sorted chunk_ids, so the same group on
re-run skips the LLM call.
"""
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from src.generation.llm_client_group import LLMClientGroup
from src.model.embedded_chunk import EmbeddedChunk
from src.model.embedding_set import EmbeddingSet
from src.model.eval_item import EvalItem
from src.model.qa import QAItem, QAList
from src.prompt.eval_generation import EVAL_GENERATION_SYSTEM, EVAL_GENERATION_USER
from src.storage.jsonl_checkpoint import JsonlCheckpoint


class EvalGenerator(Runnable[list[EmbeddingSet], list[EvalItem]]):
    """Generates Q&A pairs from chunk windows via an LLMClientGroup."""

    def __init__(
        self,
        llm_group: LLMClientGroup,
        group_size: int = 10,
        qa_per_group: int = 2,
        max_chunk_chars: int = 600,
        cache_path: Optional[str] = None,
    ):
        self.llm_group = llm_group
        self.group_size = group_size
        self.qa_per_group = qa_per_group
        self.max_chunk_chars = max_chunk_chars
        self.checkpoint = JsonlCheckpoint(cache_path)

    def invoke(
        self,
        input: list[EmbeddingSet],
        config: Optional[RunnableConfig] = None,
    ) -> list[EvalItem]:
        chunks = self._chunks_from(input)
        groups = self._make_groups(chunks)
        group_keys = [self._group_key(g) for g in groups]
        print(f"[EvalGenerator] {len(chunks)} chunks → {len(groups)} groups of ≤{self.group_size}")

        cached = self._load_cached_groups()
        cached_results: dict[int, list[QAItem]] = {}
        pending_indices: list[int] = []
        for i, key in enumerate(group_keys):
            if key in cached:
                cached_results[i] = [QAItem(**it) for it in cached[key]]
            else:
                pending_indices.append(i)
        print(f"[EvalGenerator] {len(cached_results)} groups cached, {len(pending_indices)} to generate")

        per_group: dict[int, list[QAItem]] = dict(cached_results)
        if pending_indices:
            pending_groups = [groups[i] for i in pending_indices]
            pending_keys = [group_keys[i] for i in pending_indices]
            prompts = [self._build_prompt(g) for g in pending_groups]
            on_result = self._make_on_result(pending_groups, pending_keys) if self.checkpoint.enabled() else None

            def _validate(parsed: QAList, idx: int) -> bool:
                return len(self._extract_valid(parsed, pending_groups[idx])) == self.qa_per_group

            results = self.llm_group.batch_structured(
                prompts, schema_cls=QAList, validate_fn=_validate, on_result=on_result,
            )
            for j, parsed in enumerate(results):
                valid = self._extract_valid(parsed, pending_groups[j])
                if len(valid) == self.qa_per_group:
                    per_group[pending_indices[j]] = valid

        eval_items: list[EvalItem] = []
        for i in range(len(groups)):
            if i in per_group:
                eval_items.extend(self._to_eval_items(per_group[i], start_idx=len(eval_items)))

        print(f"[EvalGenerator] Produced {len(eval_items)} eval items")
        return eval_items

    def _chunks_from(self, sets: list[EmbeddingSet]) -> list[EmbeddedChunk]:
        if not sets:
            raise ValueError("EvalGenerator requires at least one EmbeddingSet")
        sizes = {len(s.embedded_chunks) for s in sets}
        if len(sizes) != 1:
            raise ValueError(f"EmbeddingSets have different chunk counts: {sizes}")
        return sets[0].embedded_chunks

    def _make_groups(self, chunks: list[EmbeddedChunk]) -> list[list[EmbeddedChunk]]:
        return [chunks[i : i + self.group_size] for i in range(0, len(chunks), self.group_size)]

    def _group_key(self, group: list[EmbeddedChunk]) -> str:
        return "|".join(sorted(c.id for c in group))

    def _build_prompt(self, group: list[EmbeddedChunk]) -> list:
        chunks_text = "\n".join(f"[{c.id}] {c.text[:self.max_chunk_chars]}" for c in group)
        return [
            SystemMessage(content=EVAL_GENERATION_SYSTEM),
            HumanMessage(content=EVAL_GENERATION_USER.format(
                n_chunks=len(group),
                qa_per_group=self.qa_per_group,
                chunks_text=chunks_text,
            )),
        ]

    def _extract_valid(self, result, group: list[EmbeddedChunk]) -> list[QAItem]:
        if result is None:
            return []
        allowed = {c.id for c in group}
        return [
            it for it in result.items
            if it.chunk_ids and all(cid in allowed for cid in it.chunk_ids)
        ]

    def _to_eval_items(self, items: list[QAItem], start_idx: int) -> list[EvalItem]:
        out: list[EvalItem] = []
        for i, item in enumerate(items, start=1):
            doc_ids = sorted({cid.rsplit("_", 1)[0] for cid in item.chunk_ids})
            out.append(EvalItem(
                query_id=f"gen_{start_idx + i:05d}",
                query_text=item.question,
                expected_doc_ids=doc_ids,
                reference_answer=item.answer,
            ))
        return out

    def _load_cached_groups(self) -> dict[str, list[dict]]:
        return {row["group_key"]: row["items"] for row in self.checkpoint.load()}

    def _make_on_result(self, pending_groups: list, pending_keys: list[str]):
        def on_result(idx: int, parsed: QAList) -> None:
            valid = self._extract_valid(parsed, pending_groups[idx])
            if len(valid) != self.qa_per_group:
                return
            self.checkpoint.append({
                "group_key": pending_keys[idx],
                "items": [it.model_dump() for it in valid],
            })
        return on_result
