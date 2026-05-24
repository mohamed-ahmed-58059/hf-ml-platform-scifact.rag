import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.embedding.embedder import Embedder


class SentenceTransformerEmbedder(Embedder):
    """Wraps any sentence-transformers model. Handles per-model query/passage prefixes."""

    def __init__(
        self,
        model_id: str,
        query_prefix: str = "",
        passage_prefix: str = "",
        batch_size: int = 32,
        device: str | None = None,
        trust_remote_code: bool = False,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
    ):
        super().__init__(model_id)
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(
            model_id,
            device=resolved_device,
            trust_remote_code=trust_remote_code,
            model_kwargs=model_kwargs or {},
            tokenizer_kwargs=tokenizer_kwargs or {},
        )
        print(f"[SentenceTransformerEmbedder] {model_id} loaded on {self.model.device}")

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        prefixed = [f"{prefix}{t}" for t in texts] if prefix else texts
        return self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.passage_prefix)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix)
