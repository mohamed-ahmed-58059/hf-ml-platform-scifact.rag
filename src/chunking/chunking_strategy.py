from abc import ABC, abstractmethod

from src.model.chunk import Chunk
from src.model.doc import Doc


class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, docs: list[Doc]) -> list[Chunk]:
        ...
