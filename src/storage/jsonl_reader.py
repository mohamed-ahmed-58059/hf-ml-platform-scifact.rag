"""Generic JSONL reader Runnable. Inverse of JsonlWriter."""
import random
from pathlib import Path
from typing import Any, Optional, TypeVar

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class JsonlReader(Runnable[Any, list[T]]):
    """Reads one or more JSONL files and parses each non-empty line into model_cls.

    Pass a single path or a list of paths. Items from all files are pooled.
    limit=N takes a random sample of N items from the combined pool
    (seed for reproducibility). limit=None returns everything in file order.
    """

    def __init__(
        self,
        paths: str | list[str],
        model_cls: type[T],
        limit: Optional[int] = None,
        seed: int = 42,
    ):
        self.paths = [Path(paths)] if isinstance(paths, str) else [Path(p) for p in paths]
        self.model_cls = model_cls
        self.limit = limit
        self.seed = seed

    def invoke(
        self,
        input: Any = None,
        config: Optional[RunnableConfig] = None,
    ) -> list[T]:
        items: list[T] = []
        for path in self.paths:
            count = 0
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(self.model_cls.model_validate_json(line))
                        count += 1
            print(f"[JsonlReader] Read {count} items from {path}")

        if self.limit is not None and self.limit < len(items):
            items = random.Random(self.seed).sample(items, self.limit)
            print(f"[JsonlReader] Sampled {len(items)} items (limit={self.limit}, seed={self.seed})")

        return items
