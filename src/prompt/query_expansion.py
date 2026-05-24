"""Prompt template for the QueryExpander: produce N diverse search query variants."""
from langchain_core.prompts import PromptTemplate


QUERY_EXPANSION_SYSTEM = (
    "You generate diverse search query variants for retrieving scientific abstracts. "
    "Each variant should approach the question from a different angle or use different "
    "terminology (synonyms, related entities, alternative phrasings). "
    "Keep specific technical terms, names, and numbers verbatim. Do not invent new facts."
)

QUERY_EXPANSION_USER = PromptTemplate.from_template(
    "Original question:\n{question}\n\n"
    "Produce {n_variants} different search query variants for this question. "
    "Each variant should be 4-12 words, keyword-dense, and approach the same information "
    "need from a distinct angle (e.g., synonyms, related concepts, narrower or broader scope).\n\n"
    'Return ONLY JSON in this exact shape:\n{{"queries": ["variant 1", "variant 2", "..."]}}'
)
