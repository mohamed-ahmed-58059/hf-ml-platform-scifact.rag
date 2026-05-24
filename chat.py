"""Interactive RAG chat over the SciFact index (the production inference path).

Loads the winning configuration once, then answers queries — single-shot or in a
REPL. Requires a built index (chroma_data/) and a running vLLM server.

Usage:
    python chat.py "Does vitamin D reduce the risk of respiratory infections?"
    python chat.py            # interactive loop (Ctrl-D or 'exit' to quit)
"""
import sys

from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder

from src.chat import ChatChain
from src.embedding.configs import EMBEDDER_PRESETS
from src.embedding.sentence_transformer import SentenceTransformerEmbedder
from src.generation import LLMClient, LLMClientGroup, VllmSleep
from src.storage.vector_store import VectorStore


def build_chat_chain() -> ChatChain:
    store = VectorStore()

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

    # Free the GPU before loading the local encoders/reranker.
    VllmSleep().invoke(None)
    embedders = [
        SentenceTransformerEmbedder(**EMBEDDER_PRESETS["e5-large"], device="cuda"),
        SentenceTransformerEmbedder(**EMBEDDER_PRESETS["bge-large"], device="cuda"),
        SentenceTransformerEmbedder(**EMBEDDER_PRESETS["minilm"], device="cuda"),
    ]
    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cuda")

    return ChatChain(store, embedders, reranker_model, llm_group)


def main():
    chain = build_chat_chain()

    if len(sys.argv) > 1:
        print(chain.invoke(" ".join(sys.argv[1:])))
        return

    print("SciFact RAG chat — ask a question (Ctrl-D or 'exit' to quit).")
    while True:
        try:
            query = input("\n> ").strip()
        except EOFError:
            break
        if not query or query.lower() in {"exit", "quit"}:
            break
        print(f"\n{chain.invoke(query)}")


if __name__ == "__main__":
    main()
