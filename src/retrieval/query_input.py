"""Entry point for the prod rag chain. Wraps a query string into a RagContext."""
from typing import Optional

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from src.model.eval_item import EvalItem
from src.model.rag_context import RagContext


class QueryInput(Runnable[str, RagContext]):
    """Real-user entry: wraps a query into a RagContext via a placeholder EvalItem.

    For eval, callers construct RagContext directly from existing EvalItems
    (no placeholder needed because the EvalItem already exists on disk).
    """

    def invoke(
        self,
        input: str,
        config: Optional[RunnableConfig] = None,
    ) -> RagContext:
        return RagContext(item=EvalItem(
            query_id="0",
            query_text=input,
            expected_doc_ids=[],
            reference_answer=None,
        ))
