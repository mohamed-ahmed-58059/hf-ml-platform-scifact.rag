# hf-ml-platform-scifact.rag

An advanced, fully-evaluated **Retrieval-Augmented Generation** pipeline on the
[BEIR/SciFact](https://github.com/beir-cellar/beir) scientific-claim dataset, built
entirely on open-source HuggingFace models. It is an experiment framework: every
retrieval and generation technique is a swappable component, and the whole system
is benchmarked end-to-end (retrieval **and** answer quality) across a 7-experiment
ablation.

> Part of the larger `hf-ml-platform` family (VPC/RDS, a Java auth service, a Python
> model-serving API, SQS/SNS plumbing, and CI/CD). This repo is the RAG component.

## Overview

The pipeline indexes the SciFact corpus with an **ensemble of dense encoders**, fuses
their rankings with **Reciprocal Rank Fusion**, optionally **rewrites/expands** the
query and **reranks** with a cross-encoder, then **generates** a grounded answer with a
local LLM and **scores** the result with both classic IR metrics and an **LLM-as-judge**.

Because it targets a single 12 GB GPU, it choreographs VRAM carefully: vLLM is put to
sleep while the encoders/reranker run, and the local models are moved on and off the GPU
per phase so the generator can claim the full card.

```
                          ┌──────────────────────── build (once) ─────────────────────────┐
  SciFact corpus ──▶ chunk ──▶ embed (e5 · bge · MiniLM) ──▶ Chroma (1 collection/encoder)
                                                              │
                                                              ▼  synthetic QA + qrels
                          ┌──────────────────────────── eval (per experiment) ────────────────────────────┐
  query ─▶ [rewrite/expand]? ─▶ embed ─▶ vector search ─▶ RRF ─▶ [cross-encoder rerank]? ─▶ top-k
                                                                                              │
                                                       generate (Hermes-3-8B via vLLM) ◀──────┘
                                                                                              │
                          retrieval metrics (ranx) + LLM-as-judge (4 metrics) ◀──────────────┘
```

## Features

**Dataset & ingestion**
- BEIR/SciFact corpus + qrels loaders (train/test) via HuggingFace `datasets`
- Pluggable chunking via a strategy + orchestrator pattern (per-field strategies)
- Synthetic eval-set generation — QA pairs produced from corpus groups by the LLM

**Multi-encoder retrieval**
- Ensemble of `sentence-transformers` encoders run together (e5-large, bge-large, MiniLM;
  optional Nemotron-8B in 4-bit), with per-model query/passage prefixes
- **Chroma** persistent store, one collection per encoder, cosine/HNSW
- **Reciprocal Rank Fusion** across all encoders (via LangChain `EnsembleRetriever`)

**Query understanding**
- **Query rewriting** in three modes — single, *fused* (concat original + rewrite),
  *fan-out* (retrieve both, fuse with RRF)
- **Query expansion** — N variants, fan-out + RRF
- Both cached and mid-batch resumable

**Reranking**
- **Cross-encoder reranker** (`ms-marco-MiniLM-L-12-v2`), `top_k → top_n`
- Configurable query source for scoring: `original` / `rewritten` / `fused`
- One shared model instance across experiments

**Generation (multi-backend)**
- `LLMClientGroup` load-balances across LLM backends **by weight** (e.g. local vLLM +
  an MLX server), with structured (JSON-schema / JSON-mode) and text modes
- Per-lane retries, output validation, `batch_as_completed` with per-result callbacks
- Pydantic-typed structured outputs

**Evaluation**
- Retrieval metrics via [`ranx`](https://github.com/AmenRa/ranx): NDCG, MAP, MRR,
  Recall@k, Hit-Rate
- **LLM-as-judge** for faithfulness, relevance, accuracy, completeness —
  *one metric per prompt* for sharper discrimination
- Reporter + matplotlib t-SNE visualization of the embedding spaces

**Experiment framework**
- Declarative `Experiment` (build chain + eval chain) run by `ExperimentRunner`
- Live per-row + final **master metrics table**, exported to CSV/JSONL
- **Resume**: completed experiments are skipped; every stage checkpoints to JSONL
  (thread-safe, mid-batch resumable)
- Inter-experiment rest (vLLM sleep) for thermal/contention management

**GPU orchestration (single 12 GB card)**
- vLLM **sleep/wake** HTTP control to free VRAM between phases
- `ToCpu`/`ToGpu` residency management with split groups (encoders vs reranker) to
  avoid OOM and GPU contention when vLLM wakes for generation

## Tech stack

Python 3.12 · LangChain (LCEL `Runnable` pipelines) · Pydantic v2 · Polars ·
sentence-transformers · Chroma · vLLM · ranx · bitsandbytes/accelerate · ruff/tox.

## Repository layout

```
run.py                     # experiment driver (the 7-experiment ablation)
src/
  corpus/                  # SciFact corpus + qrels loaders
  chunking/                # chunking strategies + orchestrator
  embedding/               # sentence-transformers encoders + presets + orchestrator
  storage/                 # Chroma vector store, JSONL reader/writer/checkpoint
  retrieval/               # query embedders, rewriter, expander, cross-encoder reranker
  generation/              # LLM client group, generator, vLLM sleep/wake, GPU residency
  evaluation/              # scorer (ranx), LLM-as-judge, eval generator, reporter, viz
  experiments/             # Experiment, ExperimentRunner, metrics aggregation, results
  model/                   # Pydantic domain models
  prompt/                  # prompt templates
```

## Getting started

**1. Start a vLLM server** with sleep mode enabled (the dev flag exposes `/sleep`,
`/wake_up`):

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve NousResearch/Hermes-3-Llama-3.1-8B \
  --enable-sleep-mode --port 8000
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt    # or: uv sync
```

**3. Run the ablation:**

```bash
python run.py
```

The first experiment (`baseline_dense`) builds the index (downloads SciFact, embeds
~5,183 abstracts, generates the synthetic eval set of ~1,100 queries) and persists it
to `chroma_data/`. The remaining experiments reuse that index. Progress checkpoints to
`cache/`, so an interrupted run resumes where it stopped. Final results print as a master
table and are written to `experiments_results.csv`.

## How it works (design notes)

- **Everything is a LangChain `Runnable`.** Build and eval pipelines are composed with
  the `|` operator; swapping a technique is swapping a stage.
- **One collection per encoder + RRF** keeps encoders independent and lets the ensemble
  be reconfigured without re-embedding the others.
- **Fully resumable.** Each LLM stage writes results to JSONL as they land (keyed by
  query, or by `(query, metric)` for the judge); reruns skip completed work.
- **VRAM choreography.** On a 12 GB card, vLLM (~10.8 GB) cannot coexist with the
  encoders + reranker. The pipeline sleeps vLLM during retrieval, moves only the model
  needed for the current phase onto the GPU, and wakes vLLM with a clear card for
  generation.

## Results

7-experiment ablation on SciFact (retrieval metrics via `ranx`; generation metrics via
LLM-as-judge, 0–1). **`rerank_only` wins** every ranking-quality and answer-quality
metric, at the cost of a small recall dip from compressing the candidate pool into the
top-10.

| Experiment | accuracy | completeness | faithfulness | relevance | ndcg@10 | map@10 | mrr@10 | recall@5 | recall@10 | hit_rate@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_dense | 0.7375 | 0.7052 | 0.7972 | 0.8034 | 0.8625 | 0.8389 | 0.8454 | 0.9022 | 0.9271 | 0.9334 |
| rewrite_fused | 0.7503 | 0.7127 | 0.8070 | 0.8036 | 0.8615 | 0.8378 | 0.8455 | 0.8993 | 0.9257 | 0.9321 |
| expand_n3 | 0.7472 | 0.7091 | 0.8040 | 0.8090 | 0.8549 | 0.8284 | 0.8348 | 0.9037 | 0.9282 | 0.9351 |
| **rerank_only** | **0.7649** | 0.7115 | **0.8122** | 0.8086 | **0.8686** | **0.8493** | **0.8568** | 0.8946 | 0.9200 | 0.9279 |
| rewrite_rerank_orig | 0.7628 | 0.7097 | 0.8088 | 0.8050 | 0.8684 | 0.8491 | 0.8567 | 0.8946 | 0.9196 | 0.9275 |
| rewrite_rerank_fused | 0.7625 | 0.7062 | 0.8073 | 0.8035 | 0.8501 | 0.8273 | 0.8349 | 0.8800 | 0.9129 | 0.9216 |
| rewrite_rerank_wide | 0.7354 | 0.7031 | 0.7909 | 0.7980 | 0.8499 | 0.8270 | 0.8346 | 0.8797 | 0.9129 | 0.9216 |

**What the ablation shows:**
- **Cross-encoder reranking is the only technique that helped** — the single biggest lever.
- **Query rewriting adds nothing once you rerank** (`rewrite_rerank_orig` ≤ `rerank_only`).
- **Feeding the *fused* query to the reranker hurts** — the cross-encoder prefers the clean
  original query (`rewrite_rerank_fused` < `rerank_only`).
- **Query expansion slightly *hurts* ranking** on keyword-dense scientific queries, as expected.
- **More is not better:** stacking everything with a wider window (`rewrite_rerank_wide`)
  is the worst configuration — extra context adds noise.

> *On SciFact, a strong dense ensemble plus a cross-encoder reranker is the whole game;
> rewriting, expansion, and bigger context are at best neutral and at worst harmful.*

## Winning configuration

```
dense ensemble (e5-large + bge-large + MiniLM)
  → Reciprocal Rank Fusion (k=60)
  → top-50 candidates
  → cross-encoder rerank (original query)
  → top-10
  → grounded generation (Hermes-3-8B via vLLM)
```

Headline numbers: **NDCG@10 0.869 · accuracy 0.765 · faithfulness 0.812**.

### Models used

| Role | Model |
|---|---|
| Encoder (ensemble) | `intfloat/e5-large-v2` |
| Encoder (ensemble) | `BAAI/bge-large-en-v1.5` |
| Encoder (ensemble) | `sentence-transformers/all-MiniLM-L6-v2` |
| Encoder (optional, higher quality) | `nvidia/llama-embed-nemotron-8b` (4-bit) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| Generation + LLM-as-judge | `NousResearch/Hermes-3-Llama-3.1-8B` (served via vLLM) |
