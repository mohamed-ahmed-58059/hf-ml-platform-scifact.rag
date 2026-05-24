"""Chroma-backed vector store. Indexes one collection per embedder; searches by vector."""
import re
from typing import Optional

import chromadb
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from tqdm import tqdm

from src.model.embedding_set import EmbeddingSet
from src.model.retrieved_chunk import RetrievedChunk

RRF_K = 60


class _StaticDocsRetriever(BaseRetriever):
    """Returns a pre-computed list of Documents. Used to feed each per-model ranking
    into LangChain's EnsembleRetriever so the RRF math comes from the library."""

    docs: list[Document]

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        return self.docs


def _sanitize(model_id: str) -> str:
    """Chroma collection names allow only [a-zA-Z0-9_-]."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model_id)


class VectorStore(Runnable[list[EmbeddingSet], list[EmbeddingSet]]):
    """Indexer + searcher. invoke() persists as a side effect and passes the input through.

    Holds no embedders — query embedding is the QueryEmbedder's job. Callers feed
    pre-computed query vectors into search().
    """

    def __init__(self, persist_directory: str = "./chroma_data"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)

    def invoke(
        self,
        input: list[EmbeddingSet],
        config: Optional[RunnableConfig] = None,
    ) -> list[EmbeddingSet]:
        for s in tqdm(input, desc="Indexing collections"):
            name = _sanitize(s.model_id)
            # Drop existing so re-runs don't duplicate
            try:
                self.client.delete_collection(name)
            except Exception:
                pass
            collection = self.client.create_collection(
                name=name,
                metadata={"model_id": s.model_id, "hnsw:space": "cosine"},
            )
            collection.add(
                ids=[c.id for c in s.embedded_chunks],
                embeddings=[c.vector.tolist() for c in s.embedded_chunks],
                documents=[c.text for c in s.embedded_chunks],
                metadatas=[{"doc_id": c.doc_id} for c in s.embedded_chunks],
            )
        return input

    def search(
        self,
        query_vectors: dict[str, list[float]],
        top_k: int = 30,
        oversample: int = 3,
    ) -> list[RetrievedChunk]:
        """Search using pre-computed query vectors.

        Keys are normally a model_id. They may also be a fan-out key like
        '{model_id}__orig' or '{model_id}__rewritten'; the '__suffix' is stripped
        to locate the Chroma collection. Each (key, vector) becomes one ranking
        and all rankings are RRF-fused.
        """
        if not query_vectors:
            raise ValueError("search requires at least one query vector")

        n_rankings = len(query_vectors)
        n_per = top_k if n_rankings == 1 else top_k * oversample
        rankings: list[list[dict]] = []
        for key, q_vec in query_vectors.items():
            model_id = key.split("__", 1)[0]   # strip any fan-out variant suffix
            name = _sanitize(model_id)
            try:
                collection = self.client.get_collection(name=name)
            except Exception as e:
                raise ValueError(
                    f"No collection found for model '{model_id}' "
                    f"(expected collection '{name}'). Was it indexed?"
                ) from e

            res = collection.query(
                query_embeddings=[q_vec],
                n_results=n_per,
                include=["documents", "metadatas", "distances"],
            )
            rankings.append([
                {"id": id_, "doc_id": meta["doc_id"], "text": doc, "distance": dist}
                for id_, doc, meta, dist in zip(
                    res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
                )
            ])

        if n_rankings == 1:
            return [
                RetrievedChunk(id=r["id"], doc_id=r["doc_id"], text=r["text"], score=1.0 - r["distance"])
                for r in rankings[0][:top_k]
            ]

        # RRF via LangChain's EnsembleRetriever
        doc_lists = [
            [
                Document(
                    page_content=r["text"],
                    metadata={"id": r["id"], "doc_id": r["doc_id"]},
                )
                for r in ranking
            ]
            for ranking in rankings
        ]
        retrievers = [_StaticDocsRetriever(docs=docs) for docs in doc_lists]
        ensemble = EnsembleRetriever(retrievers=retrievers, c=RRF_K)
        fused = ensemble.invoke("")[:top_k]
        return [
            RetrievedChunk(
                id=d.metadata["id"],
                doc_id=d.metadata["doc_id"],
                text=d.page_content,
                score=1.0 / (rank + 1),
            )
            for rank, d in enumerate(fused)
        ]
