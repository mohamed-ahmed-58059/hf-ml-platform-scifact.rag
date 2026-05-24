"""HTTP control for a vLLM serve instance: sleep (free GPU) and wake_up.

vLLM must be started with --enable-sleep-mode for these endpoints to exist:
    vllm serve <model> --enable-sleep-mode ...

Both Runnables are pass-throughs — they perform a side effect and forward the input.
"""
import time
from typing import Any, Optional

import httpx
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig


class VllmSleep(Runnable[Any, Any]):
    """POST /sleep — vLLM releases GPU memory. Pass-through.

    level=1 → discard KV cache, swap weights to CPU RAM. Fast wake (~5-10s).
    level=2 → discard everything. Slow wake (~30-60s, reloads from disk).
    """

    def __init__(self, base_url: str = "http://localhost:8000", level: int = 1):
        self.base_url = base_url.rstrip("/")
        self.level = level

    def invoke(
        self,
        input: Any = None,
        config: Optional[RunnableConfig] = None,
    ) -> Any:
        url = f"{self.base_url}/sleep"
        print(f"[VllmSleep] POST {url}?level={self.level}")
        r = httpx.post(url, params={"level": self.level}, timeout=30.0)
        r.raise_for_status()
        print("[VllmSleep] vLLM sleeping; GPU released")
        return input


class VllmWake(Runnable[Any, Any]):
    """POST /wake_up — vLLM reloads model. Polls /v1/models until ready. Pass-through."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        poll_interval_s: float = 2.0,
        timeout_s: float = 180.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s

    def invoke(
        self,
        input: Any = None,
        config: Optional[RunnableConfig] = None,
    ) -> Any:
        wake_url = f"{self.base_url}/wake_up"
        models_url = f"{self.base_url}/v1/models"
        print(f"[VllmWake] POST {wake_url}")
        r = httpx.post(wake_url, timeout=30.0)
        r.raise_for_status()

        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            try:
                resp = httpx.get(models_url, timeout=5.0)
                if resp.status_code == 200:
                    print(f"[VllmWake] vLLM ready ({models_url})")
                    return input
            except httpx.HTTPError:
                pass
            time.sleep(self.poll_interval_s)

        raise TimeoutError(f"vLLM did not wake within {self.timeout_s}s")
