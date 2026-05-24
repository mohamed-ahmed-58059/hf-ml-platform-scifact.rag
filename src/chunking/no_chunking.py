from tqdm import tqdm

from src.chunking.chunking_strategy import ChunkingStrategy
from src.model.chunk import Chunk
from src.model.doc import Doc


class NoChunkingStrategy(ChunkingStrategy):
    """Returns one chunk per doc, with the title prepended to the text."""

    def chunk(self, docs: list[Doc]) -> list[Chunk]:
        return [
            Chunk(id=f"{d.id}_0", doc_id=d.id, text=f"{d.title}. {d.text}")
            for d in tqdm(docs, desc="NoChunking")
        ]
