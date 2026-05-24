"""Production chat path: one query in, one grounded answer out.

Reuses the retrieval/generation Runnables from the eval pipeline, wired to the
winning SciFact configuration plus query rewriting — which helps real answers in
practice even though it was metrics-neutral on the benchmark:

    rewrite (fused) -> dense ensemble + RRF -> top-k
        -> cross-encoder rerank (original query) -> top-n -> generate

On a single 12 GB GPU the models can't all stay resident, so manage_gpu (default
True) interleaves vLLM sleep/wake with ToCpu/ToGpu exactly like the eval chain.
Set manage_gpu=False when vLLM and the local models live on separate devices
(multi-GPU, or a remote vLLM server).
"""
from typing import Optional

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from sentence_transformers import CrossEncoder

from src.embedding.embedder import Embedder
from src.generation import (
    Generator,
    LLMClientGroup,
    ToCpu,
    ToGpu,
    VllmSleep,
    VllmWake,
)
from src.model.eval_item import EvalItem
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker, QuerySource
from src.retrieval.fused_query_embedder import FusedQueryEmbedder
from src.retrieval.query_rewriter import QueryRewriter
from src.retrieval.store_lookup import StoreLookup
from src.storage.vector_store import VectorStore


class ChatChain(Runnable[str, str]):
    """Single-query RAG over an existing index: query string -> answer string.

    Components are passed in already-loaded so they are reused across queries
    (construct once, call invoke() many times).
    """

    def __init__(
        self,
        store: VectorStore,
        embedders: list[Embedder],
        reranker_model: CrossEncoder,
        llm_group: LLMClientGroup,
        top_k: int = 50,
        top_n: int = 10,
        query_source: QuerySource = "original",
        manage_gpu: bool = True,
    ):
        embedder_models = [e.model for e in embedders]
        reranker_models = [reranker_model.model]

        rewrite = QueryRewriter(llm_group=llm_group)
        embed = FusedQueryEmbedder(embedders)
        lookup = StoreLookup(store, top_k=top_k)
        rerank = CrossEncoderReranker(reranker_model, top_n=top_n, query_source=query_source)
        generate = Generator(llm_group=llm_group, top_n=top_n)

        if manage_gpu:
            self._pipeline = (
                VllmWake()
                | rewrite
                | VllmSleep()
                | ToGpu(embedder_models)
                | embed
                | lookup
                | ToCpu(embedder_models)
                | ToGpu(reranker_models)
                | rerank
                | ToCpu(reranker_models)
                | VllmWake()
                | generate
            )
        else:
            self._pipeline = rewrite | embed | lookup | rerank | generate

    def invoke(self, input: str, config: Optional[RunnableConfig] = None) -> str:
        item = EvalItem(query_id="chat", query_text=input, expected_doc_ids=[])
        contexts = self._pipeline.invoke([item], config=config)
        answer = contexts[0].answer
        if answer is None:
            raise RuntimeError("Generation produced no answer")
        return answer
