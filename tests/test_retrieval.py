from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.pipeline.retriever import BM25Okapi, ClauseAwareRetriever


@dataclass
class _Doc:
    page_content: str
    metadata: dict[str, Any]


class _FakeChromaStore:
    """Fake LangChain Chroma vector store for testing.

    Supports ``similarity_search_with_score`` for dense retrieval and exposes a
    ``_collection`` attribute whose ``.get()`` returns all documents for the BM25
    corpus builder.
    """

    def __init__(
        self,
        results_by_query: dict[str, list[tuple[_Doc, float]]],
        all_docs: list[_Doc] | None = None,
    ) -> None:
        self.results_by_query = results_by_query
        # Build the internal collection data from *all_docs* when provided,
        # otherwise fall back to the union of all query result documents.
        if all_docs is not None:
            self._all_docs = list(all_docs)
        else:
            seen: set[str] = set()
            self._all_docs: list[_Doc] = []
            for docs_scores in results_by_query.values():
                for doc, _ in docs_scores:
                    cid = str(doc.metadata.get("chunk_id", id(doc)))
                    if cid not in seen:
                        seen.add(cid)
                        self._all_docs.append(doc)

        # Expose a fake _collection for _build_bm25_index
        self._collection = _FakeCollection(self._all_docs)

    def similarity_search_with_score(
        self,
        query: str,
        k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[_Doc, float]]:
        results = list(self.results_by_query.get(query, []))
        if filter:
            filtered_results: list[tuple[_Doc, float]] = []
            for document, score in results:
                matches = True
                for key, value in filter.items():
                    if str(document.metadata.get(key)) != str(value):
                        matches = False
                        break
                if matches:
                    filtered_results.append((document, score))
            results = filtered_results
        return results[:k]


class _FakeCollection:
    """Mimics the Chroma collection ``get()`` API for the BM25 corpus builder."""

    def __init__(self, docs: list[_Doc]) -> None:
        self._docs = docs

    def get(self, include: list[str] | None = None) -> dict[str, Any]:
        ids = [str(doc.metadata.get("chunk_id", f"doc_{i}")) for i, doc in enumerate(self._docs)]
        result: dict[str, Any] = {"ids": ids}
        if include and "documents" in include:
            result["documents"] = [doc.page_content for doc in self._docs]
        if include and "metadatas" in include:
            result["metadatas"] = [doc.metadata for doc in self._docs]
        return result


class _FakeVectorStore:
    def __init__(self, store: _FakeChromaStore) -> None:
        self.store = store

    def get_store(self) -> _FakeChromaStore:
        return self.store


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------


def test_clause_aware_retriever_applies_contract_filter() -> None:
    query = "termination notice"
    store = _FakeChromaStore(
        {
            query: [
                (_Doc("Contract A termination terms", {"chunk_id": "a", "contract_id": "contract_a"}), 0.92),
                (_Doc("Contract B termination terms", {"chunk_id": "b", "contract_id": "contract_b"}), 0.91),
            ]
        }
    )
    retriever = ClauseAwareRetriever(
        vector_store=_FakeVectorStore(store),
        default_k=2,
        candidate_k=2,
        enable_sparse_rerank=False,
        enable_hybrid=False,
    )

    results = retriever.get_top_k(query=query, contract_id="contract_b", k=2, clause_hints=[])

    assert results
    assert all(item.get("metadata", {}).get("contract_id") == "contract_b" for item in results)


def test_clause_aware_retriever_sparse_rerank_changes_order() -> None:
    query = "obligation breach remedy"
    store = _FakeChromaStore(
        {
            query: [
                (_Doc("obligation breach remedy", {"chunk_id": "a", "contract_id": "contract_a"}), 0.76),
                (
                    _Doc(
                        "obligation obligation obligation breach breach remedy remedy remedy remedy",
                        {"chunk_id": "b", "contract_id": "contract_a"},
                    ),
                    0.75,
                ),
            ]
        }
    )
    vector_store = _FakeVectorStore(store)

    without_sparse = ClauseAwareRetriever(
        vector_store=vector_store,
        default_k=2,
        candidate_k=2,
        enable_sparse_rerank=False,
        enable_hybrid=False,
    )
    with_sparse = ClauseAwareRetriever(
        vector_store=vector_store,
        default_k=2,
        candidate_k=2,
        enable_sparse_rerank=True,
        sparse_rerank_weight=0.3,
        enable_hybrid=False,
    )

    without_sparse_results = without_sparse.get_top_k(query=query, k=2, clause_hints=[])
    with_sparse_results = with_sparse.get_top_k(query=query, k=2, clause_hints=[])

    assert without_sparse_results[0]["chunk_id"] == "a"

    if BM25Okapi is None:
        # Optional dependency path: rerank is skipped when rank_bm25 is unavailable.
        assert with_sparse_results[0]["chunk_id"] == "a"
        return

    baseline_by_id = {item["chunk_id"]: item for item in without_sparse_results}
    sparse_by_id = {item["chunk_id"]: item for item in with_sparse_results}
    assert sparse_by_id["b"]["rerank_score"] >= baseline_by_id["b"]["rerank_score"]


# ---------------------------------------------------------------------------
# New hybrid / RRF tests
# ---------------------------------------------------------------------------


def _make_hybrid_retriever(
    dense_results: dict[str, list[tuple[_Doc, float]]],
    all_docs: list[_Doc],
    enable_hybrid: bool = True,
    candidate_k: int = 10,
    default_k: int = 5,
) -> ClauseAwareRetriever:
    """Helper to build a ``ClauseAwareRetriever`` wired to fakes for testing."""
    store = _FakeChromaStore(results_by_query=dense_results, all_docs=all_docs)
    return ClauseAwareRetriever(
        vector_store=_FakeVectorStore(store),
        default_k=default_k,
        candidate_k=candidate_k,
        enable_sparse_rerank=False,
        enable_hybrid=enable_hybrid,
    )


def test_rrf_fusion_combines_independent_rankings() -> None:
    """A document ranked high by BM25 but low by dense gets a combined RRF
    score higher than its dense-only score."""
    if BM25Okapi is None:
        return

    # doc_a: high dense, low BM25 relevance
    doc_a = _Doc("alpha beta gamma delta", {"chunk_id": "a", "contract_id": "c1"})
    # doc_b: low dense, high BM25 relevance (lots of query terms)
    doc_b = _Doc(
        "contract obligation breach breach remedy remedy obligation breach",
        {"chunk_id": "b", "contract_id": "c1"},
    )
    # doc_c: mid dense, no BM25 relevance
    doc_c = _Doc("unrelated content here", {"chunk_id": "c", "contract_id": "c1"})

    query = "obligation breach remedy"

    dense_results = {
        query: [
            (doc_a, 0.95),  # rank 1 dense
            (doc_c, 0.80),  # rank 2 dense
            (doc_b, 0.40),  # rank 3 dense — low
        ],
    }
    all_docs = [doc_a, doc_b, doc_c]

    retriever_hybrid = _make_hybrid_retriever(dense_results, all_docs, enable_hybrid=True)
    retriever_dense = _make_hybrid_retriever(dense_results, all_docs, enable_hybrid=False)

    hybrid_results = retriever_hybrid.get_top_k(query=query, k=5, clause_hints=[])
    dense_results_out = retriever_dense.get_top_k(query=query, k=5, clause_hints=[])

    hybrid_by_id = {r["chunk_id"]: r for r in hybrid_results}
    dense_by_id = {r["chunk_id"]: r for r in dense_results_out}

    # doc_b's RRF score (hybrid) should be higher than its dense-only RRF score
    assert hybrid_by_id["b"]["rerank_score"] > dense_by_id["b"]["rerank_score"]


def test_rrf_rescues_bm25_only_document() -> None:
    """A document found by BM25 but NOT found by dense retrieval still appears
    in the final hybrid results."""
    if BM25Okapi is None:
        return

    doc_a = _Doc("alpha beta gamma delta", {"chunk_id": "a", "contract_id": "c1"})
    # doc_bm25_only is in the full corpus but NOT returned by dense search
    doc_bm25_only = _Doc(
        "obligation breach remedy obligation breach remedy",
        {"chunk_id": "bm25_only", "contract_id": "c1"},
    )

    query = "obligation breach remedy"

    # Dense retrieval only returns doc_a
    dense_results = {
        query: [
            (doc_a, 0.90),
        ],
    }
    # But the full corpus has both
    all_docs = [doc_a, doc_bm25_only]

    retriever = _make_hybrid_retriever(dense_results, all_docs, enable_hybrid=True)
    results = retriever.get_top_k(query=query, k=5, clause_hints=[])

    result_ids = {r["chunk_id"] for r in results}
    assert "bm25_only" in result_ids, (
        "BM25-only document should be rescued into the final results"
    )


def test_dense_only_mode_skips_bm25() -> None:
    """When ``enable_hybrid=False``, results contain no BM25-only documents
    (the BM25 path is not used)."""
    if BM25Okapi is None:
        return

    doc_a = _Doc("alpha beta gamma delta", {"chunk_id": "a", "contract_id": "c1"})
    doc_bm25_only = _Doc(
        "obligation breach remedy obligation breach remedy",
        {"chunk_id": "bm25_only", "contract_id": "c1"},
    )

    query = "obligation breach remedy"

    dense_results = {
        query: [
            (doc_a, 0.90),
        ],
    }
    all_docs = [doc_a, doc_bm25_only]

    retriever = _make_hybrid_retriever(dense_results, all_docs, enable_hybrid=False)
    results = retriever.get_top_k(query=query, k=5, clause_hints=[])

    result_ids = {r["chunk_id"] for r in results}
    # doc_bm25_only should NOT appear because BM25 is disabled
    assert "bm25_only" not in result_ids, (
        "BM25-only document must not appear when hybrid is disabled"
    )


def test_hybrid_vs_dense_returns_different_ordering() -> None:
    """The hybrid and dense-only modes produce different rankings for the same
    query when BM25 would rank documents differently from dense."""
    if BM25Okapi is None:
        return

    # doc_a: high dense score, low BM25 relevance (no query terms)
    doc_a = _Doc("alpha beta gamma delta epsilon", {"chunk_id": "a", "contract_id": "c1"})
    # doc_b: lower dense score, high BM25 relevance (many query terms)
    doc_b = _Doc(
        "contract breach remedy obligation breach remedy obligation",
        {"chunk_id": "b", "contract_id": "c1"},
    )
    # doc_c: medium dense score, medium BM25
    doc_c = _Doc(
        "obligation overview section alpha",
        {"chunk_id": "c", "contract_id": "c1"},
    )

    query = "obligation breach remedy"

    dense_results = {
        query: [
            (doc_a, 0.95),  # dense rank 1
            (doc_c, 0.85),  # dense rank 2
            (doc_b, 0.50),  # dense rank 3
        ],
    }
    all_docs = [doc_a, doc_b, doc_c]

    retriever = _make_hybrid_retriever(dense_results, all_docs, enable_hybrid=True)

    hybrid_results = retriever.get_top_k(query=query, k=3, clause_hints=[])
    dense_results_out = retriever.dense_only_top_k(query=query, k=3, clause_hints=[])

    hybrid_order = [r["chunk_id"] for r in hybrid_results]
    dense_order = [r["chunk_id"] for r in dense_results_out]

    # The two orderings should differ because BM25 favours doc_b
    assert hybrid_order != dense_order, (
        f"Hybrid and dense-only should produce different orderings, "
        f"but both returned: {hybrid_order}"
    )
