"""Append-only JSONL checkpoint helper with thread-safe writes.

Used by stages that need mid-batch resumable caching. Holds the file handle and
a lock so concurrent worker threads can append safely.
"""
import json
import threading
from pathlib import Path
from typing import Optional


class JsonlCheckpoint:
    """Thread-safe append-only JSONL writer with bulk read for resume."""

    def __init__(self, path: Optional[str]):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._file = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def enabled(self) -> bool:
        return self.path is not None

    def load(self) -> list[dict]:
        """Returns existing rows (parsed dicts) in file order. Empty if no file."""
        if not self.path or not self.path.exists():
            return []
        out: list[dict] = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def append(self, row: dict) -> None:
        """Append one row to the JSONL. No-op if disabled."""
        if not self.path:
            return
        with self._lock:
            if self._file is None:
                self._file = self.path.open("a", encoding="utf-8")
            self._file.write(json.dumps(row) + "\n")
            self._file.flush()
