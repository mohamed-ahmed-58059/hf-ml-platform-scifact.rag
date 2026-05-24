"""Sequential experiment runner. Persists + prints results as they finish.

Uses ResultsWriter (JSONL append + final CSV) and ResultsTablePrinter (immediate
per-row + final master table). Sleeps vLLM between experiments when rest_minutes > 0.
On resume=True (default), skips experiments whose name already appears in the JSONL.
"""
import time
from pathlib import Path
from typing import Optional

import chromadb

from src.experiments.experiment import Experiment
from src.experiments.metrics_aggregator import MetricsAggregator
from src.experiments.results_table import ResultsTablePrinter
from src.experiments.results_writer import ResultsWriter
from src.generation.vllm_server import VllmSleep, VllmWake


class ExperimentRunner:
    def __init__(
        self,
        experiments: list[Experiment],
        persist_directory: str = "./chroma_data",
        aggregator: Optional[MetricsAggregator] = None,
        writer: Optional[ResultsWriter] = None,
        printer: Optional[ResultsTablePrinter] = None,
        resume: bool = True,
    ):
        names = [e.name for e in experiments]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"Duplicate experiment names: {dupes}")

        self.experiments = experiments
        self.persist_directory = persist_directory
        self.aggregator = aggregator or MetricsAggregator()
        self.writer = writer or ResultsWriter()
        self.printer = printer or ResultsTablePrinter()
        self.resume = resume
        self.results: list[tuple[str, dict[str, float]]] = []

    def run(self) -> list[tuple[str, dict[str, float]]]:
        completed = self._load_completed()
        try:
            for i, exp in enumerate(self.experiments):
                self._announce(i, exp)

                if exp.name in completed:
                    print(f"[{exp.name}] already completed — skipping (resume)")
                    metrics = completed[exp.name]
                    self.results.append((exp.name, metrics))
                    self.printer.print_row(exp.name, metrics)
                    continue

                self._run_one(exp)

                if exp.rest_minutes > 0 and i < len(self.experiments) - 1:
                    self._rest(exp.rest_minutes)
        except KeyboardInterrupt:
            print("\n\n[Runner] Interrupted — emitting partial master table")

        self.printer.print_master(self.results)
        self.writer.write_csv(self.results)
        return self.results

    def _load_completed(self) -> dict[str, dict[str, float]]:
        if not self.resume:
            self.writer.clear()
            return {}
        prior = self.writer.load()
        if prior:
            print(f"[Runner] Resuming — {len(prior)} experiments already completed")
        return dict(prior)

    def _run_one(self, exp: Experiment) -> None:
        if exp.build_chain is not None:
            print(f"[{exp.name}] build...")
            exp.build_chain.invoke(None)
        elif not self._has_index():
            raise ValueError(
                f"'{exp.name}' is eval-only but no index found at {self.persist_directory}"
            )

        print(f"[{exp.name}] eval...")
        contexts = exp.eval_chain.invoke(None)
        metrics = self.aggregator.aggregate(contexts)

        self.results.append((exp.name, metrics))
        self.writer.append_row(exp.name, metrics)
        self.printer.print_row(exp.name, metrics)

    def _announce(self, i: int, exp: Experiment) -> None:
        n = len(self.experiments)
        print(f"\n{'=' * 60}\n  EXPERIMENT {i + 1}/{n}: {exp.name}\n{'=' * 60}")

    def _has_index(self) -> bool:
        path = Path(self.persist_directory)
        if not path.exists():
            return False
        try:
            client = chromadb.PersistentClient(path=str(path))
            return len(client.list_collections()) > 0
        except Exception:
            return False

    def _rest(self, minutes: float) -> None:
        print(f"\n[Rest] vLLM sleeping for {minutes} min...")
        try:
            VllmSleep().invoke(None)
            time.sleep(minutes * 60)
        finally:
            VllmWake().invoke(None)
