"""LLMClient: one lane in an LLMClientGroup. Plus a built-in prompt-prep helper."""
from dataclasses import dataclass
from typing import Any, Callable, Optional

from langchain_core.language_models import BaseChatModel


STRICT_JSON_SUFFIX = (
    "\n\nIMPORTANT: Output ONLY the JSON object. "
    "No preamble, no markdown fences, no explanatory text."
)


def append_strict_json_suffix(messages: list) -> list:
    """Returns a new message list with STRICT_JSON_SUFFIX appended to the last message's content.

    Useful as a prepare_prompt callback for lanes whose server doesn't enforce
    JSON output at the decoder level (mlx, ollama, etc.).
    """
    if not messages:
        return messages
    last = messages[-1]
    new_last = type(last)(content=last.content + STRICT_JSON_SUFFIX)
    return messages[:-1] + [new_last]


@dataclass
class LLMClient:
    llm: BaseChatModel
    weight: float                                  # fraction of work (group weights sum to 1.0)
    method: str = "json_schema"                    # structured-output method
    max_concurrency: Optional[int] = None          # None = LangChain default
    name: str = "lane"                             # for logs
    prepare_prompt: Optional[Callable[[Any], Any]] = None   # per-lane prompt transformer
