"""Prompt template for the Generator: answer a question from retrieved chunks."""
from langchain_core.prompts import PromptTemplate


GENERATION_SYSTEM = (
    "You answer scientific questions strictly from the provided context. "
    "You do not use outside knowledge. You do not speculate."
)

GENERATION_USER = PromptTemplate.from_template(
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Instructions:\n"
    "- Use ONLY the context above. Do not invoke outside knowledge.\n"
    "- If the context does not answer the question, reply exactly: \"Insufficient context.\"\n"
    "- Be concise (1-3 sentences).\n"
    "- After your answer, cite the chunk IDs you relied on in square brackets, e.g. [12345_0].\n"
    "- Do not preface or repeat the question."
)
