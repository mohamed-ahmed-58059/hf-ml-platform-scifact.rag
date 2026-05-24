from collections import defaultdict
from typing import Optional

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from tqdm import tqdm

from src.chunking.chunking_strategy import ChunkingStrategy
from src.model.chunk import Chunk
from src.model.doc import Doc


class ChunkingOrchestrator(Runnable[list[Doc], list[Chunk]]):
    """Routes each doc to the strategy registered for its doc_type."""

    def __init__(self, strategies: dict[str, ChunkingStrategy]):
        self.strategies = strategies

    def invoke(
        self,
        input: list[Doc],
        config: Optional[RunnableConfig] = None,
    ) -> list[Chunk]:
        grouped: dict[str, list[Doc]] = defaultdict(list)
        for doc in input:
            grouped[doc.doc_type].append(doc)

        unknown = set(grouped) - set(self.strategies)
        if unknown:
            raise ValueError(f"No chunking strategy registered for doc_type(s): {sorted(unknown)}")

        chunks: list[Chunk] = []
        for doc_type, docs in tqdm(grouped.items(), desc="Chunking by doc_type"):
            chunks.extend(self.strategies[doc_type].chunk(docs))
        return chunks
