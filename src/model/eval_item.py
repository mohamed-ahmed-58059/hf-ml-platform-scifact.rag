from pydantic import BaseModel


class EvalItem(BaseModel):
    """Common eval shape for both qrel-based and agent-generated sets.

    reference_answer is None for qrel-based items (no answer ground-truth),
    populated for agent-generated items.
    """
    query_id: str
    query_text: str
    expected_doc_ids: list[str]
    reference_answer: str | None = None
    rewritten_query: str | None = None   # set by QueryRewriter; used for retrieval if present
    expanded_queries: list[str] = []     # set by QueryExpander; used by ExpandedFanOutQueryEmbedder
