"""Prompt template for the QueryRewriter: turn a question into a search-friendly query."""
from langchain_core.prompts import PromptTemplate


QUERY_REWRITE_SYSTEM = (
    "You rewrite user questions into concise, keyword-dense search queries for "
    "retrieving scientific abstracts. Strip filler words and conversational phrasing. "
    "Keep specific technical terms, entity names, and numbers verbatim."
)

QUERY_REWRITE_USER = PromptTemplate.from_template(
    "Original question:\n{question}\n\n"
    "Rewrite this as a focused search query (typically 4-12 words). "
    "Do not invent new facts. Do not change scientific meaning.\n\n"
    'Return ONLY JSON in this exact shape:\n{{"rewritten": "..."}}'
)
