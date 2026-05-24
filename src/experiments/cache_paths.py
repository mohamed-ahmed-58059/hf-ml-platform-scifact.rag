"""Convention helper for per-stage, per-experiment cache file paths."""


def cache_path(stage: str, experiment_name: str, root: str = "cache") -> str:
    """Returns f'{root}/{stage}/{experiment_name}.jsonl'.

    Stages used in this project: 'eval_gen', 'generation', 'judge'.
    """
    return f"{root}/{stage}/{experiment_name}.jsonl"
