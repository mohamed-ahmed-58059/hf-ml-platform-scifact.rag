"""Experiment driver for the SciFact RAG framework.

Defines a 7-experiment ablation and runs it through ExperimentRunner, which
checkpoints each stage and prints a master metrics table:

  1. baseline_dense        dense multi-encoder ensemble + RRF (owns the build)
  2. rewrite_fused         + LLM query rewriting (fused with the original)
  3. expand_n3             + LLM query expansion (3 variants, fan-out + RRF)
  4. rerank_only           + cross-encoder reranking            <-- best on SciFact
  5. rewrite_rerank_orig   rewriting + rerank (original query)
  6. rewrite_rerank_fused  rewriting + rerank (fused query)
  7. rewrite_rerank_wide   rewrite + rerank(fused) + a wider reranked window

Only experiment 1 carries a build_chain; the rest reuse the persisted index.

ranx emits SyntaxWarnings on import; src.evaluation.scorer suppresses them
before importing ranx, so no warning filter is needed here.
"""
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder

from src.chunking.chunking_orchestrator import ChunkingOrchestrator
from src.chunking.no_chunking import NoChunkingStrategy
from src.corpus.loader import CorpusLoader
from src.corpus.qrel_loader import QrelLoader
from src.embedding.configs import EMBEDDER_PRESETS
from src.embedding.embedding_orchestrator import EmbeddingOrchestrator
from src.embedding.sentence_transformer import SentenceTransformerEmbedder
from src.evaluation.eval_generator import EvalGenerator
from src.evaluation.judge import Judge
from src.evaluation.reporter import Reporter
from src.evaluation.scorer import Scorer
from src.experiments import Experiment, ExperimentRunner, cache_path
from src.generation import (
    Generator,
    LLMClient,
    LLMClientGroup,
    ToCpu,
    ToGpu,
    VllmSleep,
    VllmWake,
)
from src.model.eval_item import EvalItem
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.expanded_fanout_query_embedder import ExpandedFanOutQueryEmbedder
from src.retrieval.fused_query_embedder import FusedQueryEmbedder
from src.retrieval.query_embedder import QueryEmbedder
from src.retrieval.query_expander import QueryExpander
from src.retrieval.query_rewriter import QueryRewriter
from src.retrieval.store_lookup import StoreLookup
from src.storage.clear_store import ClearStore
from src.storage.jsonl_reader import JsonlReader
from src.storage.jsonl_writer import JsonlWriter
from src.storage.vector_store import VectorStore


def main():
    store = VectorStore()

    # A single local vLLM lane. LLMClientGroup can load-balance generation and
    # judging across several backends by weight — add more LLMClient entries
    # (weights summing to 1.0) to fan out to additional servers.
    llm_group = LLMClientGroup([
        LLMClient(
            llm=ChatOpenAI(
                base_url="http://localhost:8000/v1", api_key="EMPTY",
                model="NousResearch/Hermes-3-Llama-3.1-8B", temperature=0.0, max_tokens=1024,
            ),
            weight=1.0,
            method="json_schema",
            name="vllm",
        ),
    ], max_retries=2)

    # Sleep vLLM so the embedders fit on the GPU during construction.
    VllmSleep().invoke(None)

    embedders = [
        SentenceTransformerEmbedder(**EMBEDDER_PRESETS["e5-large"], device="cuda"),
        SentenceTransformerEmbedder(**EMBEDDER_PRESETS["bge-large"], device="cuda"),
        SentenceTransformerEmbedder(**EMBEDDER_PRESETS["minilm"], device="cuda"),
    ]

    # Load the cross-encoder once and share it across all reranker experiments.
    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cuda")

    # Split residency: only the models needed in the current phase live on the GPU.
    # Halves peak local VRAM vs holding embedders + reranker simultaneously, which
    # matters when vLLM wakes to claim the card for generation.
    embedder_models = [e.model for e in embedders]
    reranker_models = [reranker_model.model]

    experiments = [
        # 1. Baseline: dense retrieval over the embedder ensemble. No rewrite, no rerank.
        #    Owns the build chain — all other experiments reuse this index.
        Experiment(
            name="baseline_dense",
            build_chain=RunnableParallel(
                generated=(
                    ClearStore(store)
                    | CorpusLoader()
                    | ChunkingOrchestrator({"text": NoChunkingStrategy()})
                    | VllmSleep()
                    | ToGpu(embedder_models)
                    | EmbeddingOrchestrator(embedders)
                    | ToCpu(embedder_models)
                    | VllmWake()
                    | store
                    | EvalGenerator(
                        llm_group=llm_group, group_size=8, qa_per_group=3,
                        cache_path=cache_path("eval_gen", "baseline_dense"),
                    )
                    | JsonlWriter("eval_generated.jsonl")
                ),
                qrels=QrelLoader(splits=["train", "test"]) | JsonlWriter("eval_qrels.jsonl"),
            ),
            eval_chain=(
                JsonlReader(["eval_qrels.jsonl", "eval_generated.jsonl"], EvalItem)
                | VllmSleep()
                | ToGpu(embedder_models)
                | QueryEmbedder(embedders)
                | StoreLookup(store, top_k=50)
                | Scorer()
                | ToCpu(embedder_models + reranker_models)
                | VllmWake()
                | Generator(llm_group=llm_group, top_n=10,
                            cache_path=cache_path("generation", "baseline_dense"))
                | Judge(llm_group=llm_group, top_n=10,
                        cache_path=cache_path("judge", "baseline_dense"))
                | Reporter()
            ),
            rest_minutes=5.0,
        ),
        # 2. Rewrite only (fused): adds an LLM rewriter; no rerank.
        Experiment(
            name="rewrite_fused",
            eval_chain=(
                JsonlReader(["eval_qrels.jsonl", "eval_generated.jsonl"], EvalItem)
                | VllmWake()
                | QueryRewriter(llm_group=llm_group,
                                cache_path=cache_path("query_rewrite", "rewrite_fused"))
                | VllmSleep()
                | ToGpu(embedder_models)
                | FusedQueryEmbedder(embedders)
                | StoreLookup(store, top_k=50)
                | Scorer()
                | ToCpu(embedder_models + reranker_models)
                | VllmWake()
                | Generator(llm_group=llm_group, top_n=10,
                            cache_path=cache_path("generation", "rewrite_fused"))
                | Judge(llm_group=llm_group, top_n=10,
                        cache_path=cache_path("judge", "rewrite_fused"))
                | Reporter()
            ),
            rest_minutes=5.0,
        ),
        # 3. Expansion only: N=3 query variants, fan-out + RRF.
        Experiment(
            name="expand_n3",
            eval_chain=(
                JsonlReader(["eval_qrels.jsonl", "eval_generated.jsonl"], EvalItem)
                | VllmWake()
                | QueryExpander(llm_group=llm_group, n_variants=3,
                                cache_path=cache_path("query_expand", "expand_n3"))
                | VllmSleep()
                | ToGpu(embedder_models)
                | ExpandedFanOutQueryEmbedder(embedders)
                | StoreLookup(store, top_k=50)
                | Scorer()
                | ToCpu(embedder_models + reranker_models)
                | VllmWake()
                | Generator(llm_group=llm_group, top_n=10,
                            cache_path=cache_path("generation", "expand_n3"))
                | Judge(llm_group=llm_group, top_n=10,
                        cache_path=cache_path("judge", "expand_n3"))
                | Reporter()
            ),
            rest_minutes=5.0,
        ),
        # 4. Rerank only: dense retrieval + cross-encoder rerank (no rewrite). Best on SciFact.
        Experiment(
            name="rerank_only",
            eval_chain=(
                JsonlReader(["eval_qrels.jsonl", "eval_generated.jsonl"], EvalItem)
                | VllmSleep()
                | ToGpu(embedder_models)
                | QueryEmbedder(embedders)
                | StoreLookup(store, top_k=50)
                | ToCpu(embedder_models)
                | ToGpu(reranker_models)
                | CrossEncoderReranker(reranker_model, top_n=10, query_source="original")
                | Scorer()
                | ToCpu(embedder_models + reranker_models)
                | VllmWake()
                | Generator(llm_group=llm_group, top_n=10,
                            cache_path=cache_path("generation", "rerank_only"))
                | Judge(llm_group=llm_group, top_n=10,
                        cache_path=cache_path("judge", "rerank_only"))
                | Reporter()
            ),
            rest_minutes=5.0,
        ),
        # 5. Rewrite (fused) + rerank with the original query.
        Experiment(
            name="rewrite_rerank_orig",
            eval_chain=(
                JsonlReader(["eval_qrels.jsonl", "eval_generated.jsonl"], EvalItem)
                | VllmWake()
                | QueryRewriter(llm_group=llm_group,
                                cache_path=cache_path("query_rewrite", "rewrite_rerank_orig"))
                | VllmSleep()
                | ToGpu(embedder_models)
                | FusedQueryEmbedder(embedders)
                | StoreLookup(store, top_k=50)
                | ToCpu(embedder_models)
                | ToGpu(reranker_models)
                | CrossEncoderReranker(reranker_model, top_n=10, query_source="original")
                | Scorer()
                | ToCpu(embedder_models + reranker_models)
                | VllmWake()
                | Generator(llm_group=llm_group, top_n=10,
                            cache_path=cache_path("generation", "rewrite_rerank_orig"))
                | Judge(llm_group=llm_group, top_n=10,
                        cache_path=cache_path("judge", "rewrite_rerank_orig"))
                | Reporter()
            ),
            rest_minutes=5.0,
        ),
        # 6. Rewrite (fused) + rerank with the fused query.
        Experiment(
            name="rewrite_rerank_fused",
            eval_chain=(
                JsonlReader(["eval_qrels.jsonl", "eval_generated.jsonl"], EvalItem)
                | VllmWake()
                | QueryRewriter(llm_group=llm_group,
                                cache_path=cache_path("query_rewrite", "rewrite_rerank_fused"))
                | VllmSleep()
                | ToGpu(embedder_models)
                | FusedQueryEmbedder(embedders)
                | StoreLookup(store, top_k=50)
                | ToCpu(embedder_models)
                | ToGpu(reranker_models)
                | CrossEncoderReranker(reranker_model, top_n=10, query_source="fused")
                | Scorer()
                | ToCpu(embedder_models + reranker_models)
                | VllmWake()
                | Generator(llm_group=llm_group, top_n=10,
                            cache_path=cache_path("generation", "rewrite_rerank_fused"))
                | Judge(llm_group=llm_group, top_n=10,
                        cache_path=cache_path("judge", "rewrite_rerank_fused"))
                | Reporter()
            ),
            rest_minutes=5.0,
        ),
        # 7. Rewrite + rerank(fused) + a wider reranked window (top_n=15) + bigger context.
        Experiment(
            name="rewrite_rerank_wide",
            eval_chain=(
                JsonlReader(["eval_qrels.jsonl", "eval_generated.jsonl"], EvalItem)
                | VllmWake()
                | QueryRewriter(llm_group=llm_group,
                                cache_path=cache_path("query_rewrite", "rewrite_rerank_wide"))
                | VllmSleep()
                | ToGpu(embedder_models)
                | FusedQueryEmbedder(embedders)
                | StoreLookup(store, top_k=50)
                | ToCpu(embedder_models)
                | ToGpu(reranker_models)
                | CrossEncoderReranker(reranker_model, top_n=15, query_source="fused")
                | Scorer()
                | ToCpu(embedder_models + reranker_models)
                | VllmWake()
                | Generator(llm_group=llm_group, top_n=15,
                            cache_path=cache_path("generation", "rewrite_rerank_wide"))
                | Judge(llm_group=llm_group, top_n=15,
                        cache_path=cache_path("judge", "rewrite_rerank_wide"))
                | Reporter()
            ),
            rest_minutes=5.0,
        ),
    ]

    ExperimentRunner(experiments).run()


if __name__ == "__main__":
    main()
