from __future__ import annotations

from src.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    evaluate_retrieval,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# Precision@K
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def test_perfect_precision(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_no_relevant(self) -> None:
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_partial_precision(self) -> None:
        retrieved = ["a", "x", "b", "y"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, k=4) == 0.5

    def test_k_larger_than_retrieved(self) -> None:
        retrieved = ["a", "b"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, k=10) == 1.0

    def test_empty_retrieved(self) -> None:
        assert precision_at_k([], {"a"}, k=5) == 0.0


# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_no_recall(self) -> None:
        retrieved = ["x", "y"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, k=2) == 0.0

    def test_partial_recall(self) -> None:
        retrieved = ["a", "x", "y"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, k=3) == 0.5

    def test_empty_relevant(self) -> None:
        assert recall_at_k(["a"], set(), k=1) == 0.0


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------


class TestMRR:
    def test_first_position(self) -> None:
        assert mean_reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_second_position(self) -> None:
        assert mean_reciprocal_rank(["x", "a", "b"], {"a"}) == 0.5

    def test_third_position(self) -> None:
        result = mean_reciprocal_rank(["x", "y", "a"], {"a"})
        assert abs(result - 1 / 3) < 1e-9

    def test_no_relevant_found(self) -> None:
        assert mean_reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0

    def test_multiple_relevant_returns_first(self) -> None:
        # MRR returns reciprocal of first relevant hit
        assert mean_reciprocal_rank(["x", "a", "b"], {"a", "b"}) == 0.5


# ---------------------------------------------------------------------------
# nDCG@K
# ---------------------------------------------------------------------------


class TestNDCG:
    def test_perfect_ranking(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert ndcg_at_k(retrieved, relevant, k=3) == 1.0

    def test_no_relevant(self) -> None:
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert ndcg_at_k(retrieved, relevant, k=3) == 0.0

    def test_suboptimal_ranking(self) -> None:
        # Relevant doc at position 3 instead of 1 → nDCG < 1.0
        retrieved = ["x", "y", "a"]
        relevant = {"a"}
        result = ndcg_at_k(retrieved, relevant, k=3)
        assert 0.0 < result < 1.0

    def test_empty_relevant_set(self) -> None:
        assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0


# ---------------------------------------------------------------------------
# evaluate_retrieval (all-in-one)
# ---------------------------------------------------------------------------


class TestEvaluateRetrieval:
    def test_returns_all_metric_keys(self) -> None:
        chunks = [
            {"chunk_id": "a"},
            {"chunk_id": "b"},
            {"chunk_id": "c"},
        ]
        relevant = {"a", "b"}
        result = evaluate_retrieval(chunks, relevant, k_values=[3, 5])
        assert "precision@3" in result
        assert "precision@5" in result
        assert "recall@3" in result
        assert "recall@5" in result
        assert "ndcg@3" in result
        assert "ndcg@5" in result
        assert "mrr" in result

    def test_values_are_rounded(self) -> None:
        chunks = [{"chunk_id": "a"}, {"chunk_id": "x"}, {"chunk_id": "b"}]
        relevant = {"a", "b"}
        result = evaluate_retrieval(chunks, relevant, k_values=[3])
        for value in result.values():
            # Values should be rounded to 4 decimal places
            assert round(value, 4) == value


# ---------------------------------------------------------------------------
# aggregate_retrieval_metrics
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    def test_averages_across_queries(self) -> None:
        metrics_list = [
            {"precision@3": 1.0, "mrr": 1.0},
            {"precision@3": 0.5, "mrr": 0.5},
        ]
        result = aggregate_retrieval_metrics(metrics_list)
        assert result["precision@3"] == 0.75
        assert result["mrr"] == 0.75

    def test_empty_list_returns_empty(self) -> None:
        assert aggregate_retrieval_metrics([]) == {}
