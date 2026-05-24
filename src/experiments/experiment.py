"""One experiment definition: name + eval chain (required) + build chain (optional)."""
from dataclasses import dataclass
from typing import Optional

from langchain_core.runnables import Runnable


@dataclass
class Experiment:
    """A named, run-once-and-record experiment.

    build_chain is optional — when None, the experiment reuses whichever index
    a previous experiment built (or fails if no index exists yet).
    rest_minutes is a post-experiment cool-down: vLLM is slept and the runner
    waits before continuing to the next experiment.
    """

    name: str
    eval_chain: Runnable
    build_chain: Optional[Runnable] = None
    rest_minutes: float = 0.0
