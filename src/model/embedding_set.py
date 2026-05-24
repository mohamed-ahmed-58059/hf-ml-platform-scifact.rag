from pydantic import BaseModel

from src.model.embedded_chunk import EmbeddedChunk


class EmbeddingSet(BaseModel):
    model_id: str
    embedded_chunks: list[EmbeddedChunk]
