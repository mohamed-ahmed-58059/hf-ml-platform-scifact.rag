"""JSONL sink Runnable. Writes a list of pydantic models to disk and passes through."""
from pathlib import Path
from typing import Optional, TypeVar

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class JsonlWriter(Runnable[list[T], list[T]]):
    """Writes each item as a JSONL line, then passes the input through.

    Default mode "w" — file is truncated/overwritten on each invoke.
    Use append=True to append instead.
    """

    def __init__(self, path: str, append: bool = False):
        self.path = Path(path)
        self.append = append

    def invoke(
        self,
        input: list[T],
        config: Optional[RunnableConfig] = None,
    ) -> list[T]:
        mode = "a" if self.append else "w"
        with self.path.open(mode, encoding="utf-8") as f:
            for item in input:
                f.write(item.model_dump_json() + "\n")
        print(f"[JsonlWriter] Wrote {len(input)} items to {self.path}")
        return input
