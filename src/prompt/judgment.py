"""Prompt templates for the Judge: one focused prompt per metric.

Research (Eugene Yan, Arize) shows small judges (≤8B) produce more discriminating
scores when asked to evaluate one criterion at a time vs. a multi-metric JSON.
"""
from langchain_core.prompts import PromptTemplate


JUDGMENT_SYSTEM = (
    "You are a precise evaluator. Return a single float score between 0.0 and 1.0 "
    "for the requested metric. Use the full continuous range — do not default to a "
    "round number unless the evidence genuinely sits at that level."
)


METRIC_RUBRICS: dict[str, str] = {
    "faithfulness": (
        "Faithfulness measures whether every claim in the generated answer is supported by the context.\n"
        "  0.0 = the answer contains claims that are not in the context (fabrication).\n"
        "  1.0 = every claim in the answer is directly grounded in the context.\n"
        "  Score continuously based on the proportion of claims that are supported."
    ),
    "relevance": (
        "Relevance measures whether the generated answer addresses the question that was asked.\n"
        "  0.0 = off-topic; the answer does not address the question.\n"
        "  1.0 = the answer directly and fully addresses what the question asks.\n"
        "  Score continuously based on how on-point the answer is."
    ),
    "accuracy": (
        "Accuracy measures factual agreement between the generated answer and the reference answer.\n"
        "  0.0 = the answer contradicts the reference.\n"
        "  1.0 = the answer is factually consistent with the reference.\n"
        "  Score continuously based on the proportion of facts that align."
    ),
    "completeness": (
        "Completeness measures how much of the reference's key information the generated answer covers.\n"
        "  0.0 = the answer misses all key points in the reference.\n"
        "  1.0 = the answer covers every key point in the reference.\n"
        "  Score continuously based on coverage."
    ),
}


JUDGMENT_USER = PromptTemplate.from_template(
    "Question:\n{question}\n\n"
    "Context (chunks the generated answer was based on):\n{context}\n\n"
    "Generated Answer:\n{generated_answer}\n\n"
    "Reference Answer:\n{reference_answer}\n\n"
    "Metric to evaluate: {metric_name}\n\n"
    "{rubric}\n\n"
    'Return ONLY JSON in this exact shape:\n{{"score": 0.0}}'
)
