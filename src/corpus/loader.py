from typing import Any, Optional

from datasets import load_dataset
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from tqdm import tqdm

from src.model.doc import Doc

DATASET_ID = "BeIR/scifact"


class CorpusLoader(Runnable[Any, list[Doc]]):
    """Loads the BEIR/scifact corpus from HuggingFace and returns it as a list of Docs.

    Pass limit=N to truncate to the first N docs (useful for fast iteration);
    None (default) returns the whole corpus.
    """

    def __init__(self, dataset_id: str = DATASET_ID, limit: int | None = None):
        self.dataset_id = dataset_id
        self.limit = limit

    def invoke(
        self, input: Any = None, config: Optional[RunnableConfig] = None
    ) -> list[Doc]:
        ds = load_dataset(self.dataset_id, "corpus", split="corpus")
        if self.limit is not None:
            ds = ds.select(range(min(self.limit, len(ds))))
        return [
            Doc(id=row["_id"], title=row["title"], text=row["text"], doc_type="text")
            for row in tqdm(ds, desc=f"Loading {self.dataset_id}")
        ]
