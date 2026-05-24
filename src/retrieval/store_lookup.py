from typing import Optional

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from tqdm import tqdm

from src.model.rag_context import RagContext
from src.storage.vector_store import VectorStore


class StoreLookup(Runnable[list[RagContext], list[RagContext]]):

    def __init__(self, store: VectorStore, top_k: int = 30, oversample: int = 3):
        self.store = store
        self.top_k = top_k
        self.oversample = oversample

    def invoke(
        self,
        input: list[RagContext],
        config: Optional[RunnableConfig] = None,
    ) -> list[RagContext]:
        for ctx in tqdm(input, desc="Searching store"):
            ctx.retrieved = self.store.search(
                query_vectors=ctx.query_vectors,
                top_k=self.top_k,
                oversample=self.oversample,
            )
        return input
