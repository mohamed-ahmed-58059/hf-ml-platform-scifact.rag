"""LLMClientGroup: orchestrates batched LLM calls across multiple lanes.

Partitions prompts by weight, runs each lane in a thread, optionally retries
per-lane failures (exception, None, validation reject). Mode-agnostic; the
per-call-shape variation lives in BatchMode (StructuredMode, TextMode).

If on_result is provided, fires per successful response (after extract + validate)
so callers can incrementally checkpoint results to disk.
"""
import concurrent.futures
import time
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel
from tqdm import tqdm

from src.generation.batch_mode import StructuredMode, TextMode
from src.generation.llm_client import LLMClient


T = TypeVar("T", bound=BaseModel)


class LLMClientGroup:
    """Splits prompts across multiple LLMClients, runs each lane in parallel."""

    def __init__(self, clients: list[LLMClient], max_retries: int = 0):
        if not clients:
            raise ValueError("LLMClientGroup requires at least one client")
        total = sum(c.weight for c in clients)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0; got {total}")
        self.clients = clients
        self.max_retries = max_retries

    def batch_structured(
        self,
        prompts: list,
        schema_cls: type[T],
        validate_fn: Optional[Callable[[T, int], bool]] = None,
        on_result: Optional[Callable[[int, T], None]] = None,
    ) -> list[Optional[T]]:
        return self._dispatch(prompts, StructuredMode(schema_cls), validate_fn, on_result)

    def batch_text(
        self,
        prompts: list,
        validate_fn: Optional[Callable[[str, int], bool]] = None,
        on_result: Optional[Callable[[int, str], None]] = None,
    ) -> list[Optional[str]]:
        return self._dispatch(prompts, TextMode(), validate_fn, on_result)

    def _dispatch(self, prompts, mode, validate_fn, on_result) -> list:
        partitions = self._partition(len(prompts))
        results: dict[int, Any] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.clients)) as pool:
            futures = [
                pool.submit(self._run_lane, client, indices, prompts, mode, validate_fn, on_result)
                for client, indices in zip(self.clients, partitions)
            ]
            for f in concurrent.futures.as_completed(futures):
                results.update(f.result())
        return [results.get(i) for i in range(len(prompts))]

    def _partition(self, n: int) -> list[list[int]]:
        counts = [int(c.weight * n) for c in self.clients]
        diff = n - sum(counts)
        if diff:
            heaviest = max(range(len(self.clients)), key=lambda i: self.clients[i].weight)
            counts[heaviest] += diff
        out, offset = [], 0
        for c in counts:
            out.append(list(range(offset, offset + c)))
            offset += c
        return out

    def _run_lane(self, client, indices, prompts, mode, validate_fn, on_result) -> dict[int, Any]:
        if not indices:
            return {}

        runnable = mode.make_runnable(client.llm, client.method)
        results: dict[int, Any] = {}
        pending = list(indices)

        for round_idx in range(self.max_retries + 1):
            if not pending:
                break
            successes, pending = self._run_round(
                client, runnable, mode, validate_fn, on_result, prompts, pending, round_idx,
            )
            results.update(successes)

        for idx in pending:
            results[idx] = None
        if pending:
            print(f"[{client.name}] {len(pending)} prompts failed after {self.max_retries + 1} rounds")
        return results

    def _run_round(self, client, runnable, mode, validate_fn, on_result, prompts, pending, round_idx):
        """One pass over pending indices. Returns (successes_dict, new_pending_list)."""
        lane_prompts = self._prepare_lane_prompts(client, prompts, pending)
        config = self._batch_config(client)

        self._log_start(client, mode, round_idx, len(lane_prompts))
        t0 = time.perf_counter()

        successes: dict[int, Any] = {}
        new_pending: list[int] = []
        bar = tqdm(
            runnable.batch_as_completed(lane_prompts, config=config, return_exceptions=True),
            total=len(lane_prompts),
            desc=f"[{client.name}] {mode.label} r{round_idx + 1}",
            leave=False,
        )
        for local_idx, raw in bar:
            global_idx = pending[local_idx]
            value = mode.extract(raw)
            if value is None or (validate_fn is not None and not validate_fn(value, global_idx)):
                new_pending.append(global_idx)
                continue
            successes[global_idx] = value
            self._fire_on_result(client, on_result, global_idx, value)

        self._log_finish(client, mode, round_idx, len(lane_prompts), len(new_pending), time.perf_counter() - t0)
        return successes, new_pending

    @staticmethod
    def _prepare_lane_prompts(client, prompts, pending):
        lane_prompts = [prompts[i] for i in pending]
        if client.prepare_prompt is not None:
            lane_prompts = [client.prepare_prompt(p) for p in lane_prompts]
        return lane_prompts

    @staticmethod
    def _batch_config(client):
        return {"max_concurrency": client.max_concurrency} if client.max_concurrency else None

    @staticmethod
    def _fire_on_result(client, on_result, global_idx, value):
        if on_result is None:
            return
        try:
            on_result(global_idx, value)
        except Exception as e:
            print(f"[{client.name}] on_result raised: {e!r}")

    def _log_start(self, client, mode, round_idx, n_prompts):
        round_label = f"round {round_idx + 1}/{self.max_retries + 1}" if self.max_retries > 0 else ""
        print(f"[{client.name}] starting ({mode.label}) {round_label}: "
              f"{n_prompts} prompts (max_concurrency={client.max_concurrency})")

    def _log_finish(self, client, mode, round_idx, n_total, n_failed, elapsed):
        round_label = f"round {round_idx + 1}/{self.max_retries + 1}" if self.max_retries > 0 else ""
        print(f"[{client.name}] finished ({mode.label}) {round_label}: "
              f"{n_total - n_failed}/{n_total} succeeded "
              f"in {elapsed:.1f}s ({elapsed / max(n_total, 1):.2f}s/prompt)")
