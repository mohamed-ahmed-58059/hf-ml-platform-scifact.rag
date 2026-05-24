"""Loads BEIR/scifact queries + qrels from HuggingFace, joins into list[EvalItem]."""
from typing import Any, Optional

import polars as pl
from datasets import load_dataset
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from src.model.eval_item import EvalItem

DATASET_ID = "BeIR/scifact"
QRELS_DATASET_ID = "BeIR/scifact-qrels"


class QrelLoader(Runnable[Any, list[EvalItem]]):
    """Joins BEIR queries + qrels into EvalItems with reference_answer=None.

    Pass splits=["train", "test"] (default) to load both. SciFact's train and test
    queries are disjoint, so the merged list has no duplicates.
    """

    def __init__(self, splits: list[str] = None):
        self.splits = splits if splits is not None else ["train", "test"]

    def invoke(
        self, input: Any = None, config: Optional[RunnableConfig] = None
    ) -> list[EvalItem]:
        queries_ds = load_dataset(DATASET_ID, "queries", split="queries")
        queries_df = (
            pl.from_arrow(queries_ds.data.table)
            .select(
                pl.col("_id").cast(pl.String).alias("query_id"),
                pl.col("text").alias("query_text"),
            )
        )

        qrels_frames = []
        for split in self.splits:
            split_ds = load_dataset(QRELS_DATASET_ID, split=split)
            qrels_frames.append(
                pl.from_arrow(split_ds.data.table)
                .filter(pl.col("score") >= 1)
                .select(
                    pl.col("query-id").cast(pl.String).alias("query_id"),
                    pl.col("corpus-id").cast(pl.String).alias("doc_id"),
                )
            )
        qrels_df = pl.concat(qrels_frames) if qrels_frames else pl.DataFrame()

        joined = (
            qrels_df
            .join(queries_df, on="query_id", how="inner")
            .group_by(["query_id", "query_text"])
            .agg(pl.col("doc_id").unique().sort().alias("expected_doc_ids"))
        )

        items = [
            EvalItem(
                query_id=row["query_id"],
                query_text=row["query_text"],
                expected_doc_ids=list(row["expected_doc_ids"]),
                reference_answer=None,
            )
            for row in joined.iter_rows(named=True)
        ]
        print(f"[QrelLoader] Loaded {len(items)} eval items from splits {self.splits}")
        return items
