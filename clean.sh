#!/bin/bash
# Dev caches
rm -rf .tox
rm -rf .ruff_cache
rm -rf hf_ml_platform_scifact_rag.egg-info
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name ".pytest_cache" -exec rm -rf {} +

# Pipeline artifacts
rm -rf chroma_data
rm -rf cache
rm -f eval_qrels.jsonl
rm -f eval_generated.jsonl
rm -f experiments_results.jsonl
rm -f experiments_results.csv
rm -f tsne.png

echo "Cleaned."
