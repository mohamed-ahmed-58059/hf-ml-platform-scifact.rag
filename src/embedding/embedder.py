from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Encodes texts into vectors. Same object handles passages (build) and queries (retrieval)."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def embed_passages(self, texts: list[str]) -> np.ndarray:
        ...

    @abstractmethod
    def embed_queries(self, texts: list[str]) -> np.ndarray:
        ...
