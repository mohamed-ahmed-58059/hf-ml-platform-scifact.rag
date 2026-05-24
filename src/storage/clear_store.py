"""Wipes all collections from a VectorStore. Pass-through.

Use as the first stage of a build chain to ensure no stale collections survive
from a previous experiment's build.
"""
from typing import Any, Optional

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from src.storage.vector_store import VectorStore


class ClearStore(Runnable[Any, Any]):
    """Deletes every Chroma collection in the store. Returns its input unchanged."""

    def __init__(self, store: VectorStore):
        self.store = store

    def invoke(
        self,
        input: Any = None,
        config: Optional[RunnableConfig] = None,
    ) -> Any:
        collections = self.store.client.list_collections()
        if not collections:
            print("[ClearStore] No collections to clear")
            return input
        for c in collections:
            name = c.name if hasattr(c, "name") else c
            self.store.client.delete_collection(name)
            print(f"[ClearStore] Deleted collection '{name}'")
        return input
