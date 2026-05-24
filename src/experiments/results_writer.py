"""Persists experiment results: per-row JSONL (append as you go) + final CSV.

Does NOT truncate the JSONL on construction so the runner can resume across runs.
Use ResultsWriter(...).clear() if you want a fresh start.
"""
import json
from pathlib import Path

import polars as pl


class ResultsWriter:
    def __init__(
        self,
        jsonl_path: str = "experiments_results.jsonl",
        csv_path: str = "experiments_results.csv",
    ):
        self.jsonl_path = Path(jsonl_path)
        self.csv_path = Path(csv_path)

    def load(self) -> list[tuple[str, dict[str, float]]]:
        """Returns previously persisted (name, metrics) pairs in append order."""
        if not self.jsonl_path.exists():
            return []
        out: list[tuple[str, dict[str, float]]] = []
        with self.jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                name = row.pop("experiment")
                out.append((name, row))
        return out

    def append_row(self, name: str, metrics: dict[str, float]) -> None:
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"experiment": name, **metrics}) + "\n")

    def write_csv(self, results: list[tuple[str, dict[str, float]]]) -> None:
        if not results:
            return
        all_keys = sorted({k for _, m in results for k in m})
        rows = [{"experiment": name, **{k: m.get(k) for k in all_keys}} for name, m in results]
        pl.DataFrame(rows).write_csv(self.csv_path)
        print(f"[ResultsWriter] Wrote master CSV to {self.csv_path}")

    def clear(self) -> None:
        """Truncate the JSONL (used when resume=False)."""
        if self.jsonl_path.exists():
            self.jsonl_path.write_text("")
