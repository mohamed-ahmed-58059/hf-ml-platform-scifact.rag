"""Move local PyTorch models on/off the GPU around vLLM wake/sleep.

The 12 GB card can't hold vLLM (~10.8 GB) + our embedders + reranker (~3.8 GB)
simultaneously. We co-locate by:

    ... | retrieve | rerank | ToCpu(models) | VllmWake | generate | VllmSleep | ToGpu(models) | ...

`.to('cpu')` alone leaves PyTorch's caching allocator holding the freed blocks,
which keeps `nvidia-smi` usage high and prevents vLLM from claiming the slack.
The full pattern needed to actually release VRAM to another process is:
    model.to('cpu') → gc.collect() → torch.cuda.empty_cache() → torch.cuda.ipc_collect()
(See pytorch/pytorch#72117 and the vLLM sleep-mode docs.)

Pinned CPU buffers would shave the return-trip copy further, but plain pageable
RAM costs <1s for 3.8 GB on PCIe 4.0 — not worth the bookkeeping.
"""
import gc
from typing import Any, Optional

import torch
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig


class ToCpu(Runnable[Any, Any]):
    """Move the given models to CPU and force the CUDA allocator to release. Pass-through."""

    def __init__(self, models: list):
        self.models = models

    def invoke(self, input: Any = None, config: Optional[RunnableConfig] = None) -> Any:
        for m in self.models:
            m.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print(f"[ToCpu] moved {len(self.models)} models to CPU")
        return input


class ToGpu(Runnable[Any, Any]):
    """Move the given models back to CUDA. Pass-through."""

    def __init__(self, models: list, device: str = "cuda"):
        self.models = models
        self.device = device

    def invoke(self, input: Any = None, config: Optional[RunnableConfig] = None) -> Any:
        for m in self.models:
            m.to(self.device)
        print(f"[ToGpu] moved {len(self.models)} models to {self.device}")
        return input
