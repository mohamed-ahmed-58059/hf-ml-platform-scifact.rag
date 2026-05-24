from typing import Optional

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from tqdm import tqdm

from src.embedding.embedder import Embedder
from src.model.chunk import Chunk
from src.model.embedded_chunk import EmbeddedChunk
from src.model.embedding_set import EmbeddingSet


class EmbeddingOrchestrator(Runnable[list[Chunk], list[EmbeddingSet]]):
    """Runs each embedder over all chunks. One EmbeddingSet out per embedder.

    Embedders are NOT freed after invoke — pass the same instances to QueryEmbedder
    later so they can be reused without re-loading on GPU.
    """

    def __init__(self, embedders: list[Embedder]):
        if not embedders:
            raise ValueError("EmbeddingOrchestrator requires at least one embedder")
        self.embedders = embedders

    def invoke(
        self,
        input: list[Chunk],
        config: Optional[RunnableConfig] = None,
    ) -> list[EmbeddingSet]:
        texts = [c.text for c in input]
        sets: list[EmbeddingSet] = []
        for embedder in tqdm(self.embedders, desc="Embedders"):
            vectors = embedder.embed_passages(texts)
            embedded = [
                EmbeddedChunk(id=c.id, doc_id=c.doc_id, text=c.text, vector=v)
                for c, v in tqdm(
                    zip(input, vectors),
                    total=len(input),
                    desc=f"Building EmbeddedChunks ({embedder.model_id})",
                )
            ]
            sets.append(EmbeddingSet(model_id=embedder.model_id, embedded_chunks=embedded))
        return sets
