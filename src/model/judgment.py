"""Schema for one-metric-per-call LLM-as-judge output."""
from pydantic import BaseModel


class MetricScore(BaseModel):
    score: float    # 0.0–1.0
