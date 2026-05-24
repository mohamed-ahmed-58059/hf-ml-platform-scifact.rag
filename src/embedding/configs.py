"""Per-model presets for SentenceTransformerEmbedder.

Pick from EMBEDDER_PRESETS by short name and pass **EMBEDDER_PRESETS[name] into
SentenceTransformerEmbedder(...). Add new entries here when bringing in a new model.
"""
import torch
from transformers import BitsAndBytesConfig


EMBEDDER_PRESETS: dict[str, dict] = {
    # Best single encoder on SciFact in our prior runs
    "bge-large": {
        "model_id": "BAAI/bge-large-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
    },
    # Strong complement to bge in the ensemble
    "mxbai": {
        "model_id": "mixedbread-ai/mxbai-embed-large-v1",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
    },
    # Needs query/passage prefixes — without them, recall drops ~2 pts
    "e5-large": {
        "model_id": "intfloat/e5-large-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    # Tiny (22M params, 384-dim). Fast baseline / sanity check; weaker than the larges
    "minilm": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "passage_prefix": "",
    },
    # NVIDIA Llama-Embed-Nemotron 8B (4096-dim). Top of multilingual MTEB as of Oct 2025.
    # Loaded in 4-bit via bitsandbytes to fit on a 12 GB card. Slow — small batch.
    "nemotron-8b": {
        "model_id": "nvidia/llama-embed-nemotron-8b",
        "query_prefix": (
            "Instruct: Given a question, retrieve passages that answer the question\n"
            "Query: "
        ),
        "passage_prefix": "",
        "batch_size": 4,
        "trust_remote_code": True,
        "model_kwargs": {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            ),
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "eager",
        },
        "tokenizer_kwargs": {"padding_side": "left"},
    },
}
