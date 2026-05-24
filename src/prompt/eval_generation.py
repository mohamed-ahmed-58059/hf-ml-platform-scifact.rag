"""Prompt for the EvalGenerator: produce qa_per_group Q&A pairs from a chunk window."""
from langchain_core.prompts import PromptTemplate


EVAL_GENERATION_SYSTEM = "detailed thinking off"

EVAL_GENERATION_USER = PromptTemplate.from_template(
    "You are given {n_chunks} scientific abstracts, each prefixed by a unique chunk ID. "
    "Generate exactly {qa_per_group} question-answer pairs grounded ONLY in the provided abstracts. "
    "Each pair must include the chunk_ids it draws from. Use ONLY chunk IDs that appear below.\n\n"
    "Abstracts:\n{chunks_text}\n\n"
    "Return JSON with an 'items' field — a list of {qa_per_group} objects, each with "
    "'question', 'answer', and 'chunk_ids' fields."
)
