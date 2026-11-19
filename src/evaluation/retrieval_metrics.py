"""Retrieval quality metrics: Precision@K, Recall@K, MRR, and nDCG@K."""

from __future__ import annotations

import math
from statistics import mean
from typing import Any


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k retrieved documents that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for doc_id in top_k if doc_id in relevant_ids) / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant documents found in top-k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return sum(1 for doc_id in top_k if doc_id in relevant_ids) / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant document."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    top_k = retrieved_ids[:k]

    dcg = 0.0
    for rank, doc_id in enumerate(top_k):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / math.log2(rank + 2)  # rank is 0-indexed, so +2

    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval(
    retrieved_chunks: list[dict[str, Any]],
    relevant_chunk_ids: set[str],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute all retrieval metrics for a single query at multiple k values."""
    if k_values is None:
        k_values = [3, 5, 10]

    retrieved_ids = [str(chunk.get("chunk_id", "")) for chunk in retrieved_chunks]

    results: dict[str, float] = {}
    for k in k_values:
        results[f"precision@{k}"] = round(precision_at_k(retrieved_ids, relevant_chunk_ids, k), 4)
        results[f"recall@{k}"] = round(recall_at_k(retrieved_ids, relevant_chunk_ids, k), 4)
        results[f"ndcg@{k}"] = round(ndcg_at_k(retrieved_ids, relevant_chunk_ids, k), 4)

    results["mrr"] = round(mean_reciprocal_rank(retrieved_ids, relevant_chunk_ids), 4)
    return results


def aggregate_retrieval_metrics(all_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Average retrieval metrics across multiple queries."""
    if not all_metrics:
        return {}

    keys = all_metrics[0].keys()
    return {key: round(mean(m.get(key, 0.0) for m in all_metrics), 4) for key in keys}
