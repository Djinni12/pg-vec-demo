"""Unit tests for legal dense + BM25 hybrid retrieval."""

from unittest.mock import Mock, patch

import pytest

from src.retrievers.legal_hybrid_retriever import (
    DEFAULT_RETRIEVE_LIMIT,
    DEFAULT_RRF_K,
    chunk_key,
    fuse_legal_results,
    hybrid_search_legal_corpus,
)


def result(chunk_id, document_type="act", reference="Section 1", title="Title", score=0.7):
    return {
        "chunk_id": chunk_id,
        "document_type": document_type,
        "reference": reference,
        "title": title,
        "score": score,
    }


def bm25_result(chunk_id, document_type="act", reference="Section 1", title="Title", score=5.0):
    return {
        "chunk_id": chunk_id,
        "document_type": document_type,
        "reference": reference,
        "title": title,
        "bm25_score": score,
    }


def test_chunk_key_requires_chunk_id():
    with pytest.raises(ValueError, match="chunk_id"):
        chunk_key({"title": "missing key"})


def test_fuse_legal_results_uses_rank_positions_not_raw_scores():
    dense = [
        result("shared", score=0.1),
        result("dense-only", score=0.99),
    ]
    bm25 = [
        bm25_result("shared", score=1.0),
        bm25_result("bm25-only", score=99.0),
    ]

    fused = fuse_legal_results(dense, bm25, top_k=3, k=60)

    assert [item["chunk_id"] for item in fused] == ["shared", "bm25-only", "dense-only"]
    assert fused[0]["rrf_score"] == pytest.approx((1 / 61) + (1 / 61))
    assert fused[1]["rrf_score"] == pytest.approx(1 / 62)
    assert fused[2]["rrf_score"] == pytest.approx(1 / 62)


def test_fuse_legal_results_preserves_required_fields_and_missing_side_values():
    dense = [result("dense-only", document_type="rule", reference="Rule 23", title="Revocation", score=0.82)]
    bm25 = [bm25_result("bm25-only", document_type="form", reference="REG-16", title="Cancellation", score=7.5)]

    fused = fuse_legal_results(dense, bm25, top_k=2, k=60)

    dense_only = next(item for item in fused if item["chunk_id"] == "dense-only")
    assert dense_only == {
        "chunk_id": "dense-only",
        "document_type": "rule",
        "reference": "Rule 23",
        "title": "Revocation",
        "dense_rank": 1,
        "dense_score": 0.82,
        "bm25_rank": None,
        "bm25_score": None,
        "rrf_score": pytest.approx(1 / 61),
    }

    bm25_only = next(item for item in fused if item["chunk_id"] == "bm25-only")
    assert bm25_only["dense_rank"] is None
    assert bm25_only["dense_score"] is None
    assert bm25_only["bm25_rank"] == 1
    assert bm25_only["bm25_score"] == 7.5


def test_fuse_legal_results_keeps_duplicate_chunk_once_per_retriever():
    duplicate = result("same")
    fused = fuse_legal_results([duplicate, duplicate], [bm25_result("same")], top_k=10, k=60)

    assert len(fused) == 1
    assert fused[0]["rrf_score"] == pytest.approx((1 / 61) + (1 / 61))


@patch("src.retrievers.legal_hybrid_retriever.BM25Retriever")
@patch("src.retrievers.legal_hybrid_retriever.dense_search_legal_corpus")
def test_hybrid_search_runs_dense_and_bm25_with_top_30_then_fuses(mock_dense_search, mock_bm25_cls):
    conn = object()
    conn_params = {"host": "localhost"}
    model = object()
    dense_rows = [result("dense"), result("shared", score=0.5)]
    bm25_rows = [bm25_result("shared", score=6.0), bm25_result("bm25")]
    mock_dense_search.return_value = (dense_rows, {"act": 2, "rule": 0, "form": 0})
    mock_bm25 = Mock()
    mock_bm25.retrieve.return_value = (bm25_rows, {"retrieval_time": 0.01})
    mock_bm25_cls.return_value = mock_bm25

    fused, stats = hybrid_search_legal_corpus(conn, "cancel registration", conn_params, model=model)

    mock_dense_search.assert_called_once_with(
        conn,
        "cancel registration",
        top_k=DEFAULT_RETRIEVE_LIMIT,
        per_table_limit=DEFAULT_RETRIEVE_LIMIT,
        model=model,
        rerank=False,
    )
    mock_bm25_cls.assert_called_once_with(conn_params)
    mock_bm25.retrieve.assert_called_once_with("cancel registration", top_k=DEFAULT_RETRIEVE_LIMIT)
    assert fused[0]["chunk_id"] == "shared"
    assert fused[0]["dense_rank"] == 2
    assert fused[0]["bm25_rank"] == 1
    assert stats["retrieve_limit"] == DEFAULT_RETRIEVE_LIMIT
    assert stats["rrf_k"] == DEFAULT_RRF_K
    assert stats["reranked"] is False


@patch("src.retrievers.legal_hybrid_retriever.rerank_legal_results")
@patch("src.retrievers.legal_hybrid_retriever.BM25Retriever")
@patch("src.retrievers.legal_hybrid_retriever.dense_search_legal_corpus")
def test_hybrid_search_reranks_after_rrf_not_dense_only(mock_dense_search, mock_bm25_cls, mock_rerank):
    conn = object()
    conn_params = {"host": "localhost"}
    reranker = object()
    dense_rows = [
        {**result("dense"), "content": "dense only"},
        {**result("shared", score=0.5), "content": "shared dense"},
    ]
    bm25_rows = [
        {**bm25_result("shared", score=6.0), "content": "shared bm25"},
        {**bm25_result("bm25"), "content": "bm25 only"},
    ]
    mock_dense_search.return_value = (dense_rows, {"act": 2, "rule": 0, "form": 0})
    mock_bm25 = Mock()
    mock_bm25.retrieve.return_value = (bm25_rows, {"retrieval_time": 0.01})
    mock_bm25_cls.return_value = mock_bm25
    mock_rerank.side_effect = lambda query, candidates, reranker=None: [
        {**candidate, "rerank_score": 10.0 - index}
        for index, candidate in enumerate(reversed(candidates))
    ]

    fused, stats = hybrid_search_legal_corpus(
        conn,
        "cancel registration",
        conn_params,
        top_k=2,
        reranker=reranker,
        rerank_candidate_limit=3,
    )

    candidates_passed_to_reranker = mock_rerank.call_args.args[1]
    mock_rerank.assert_called_once()
    assert mock_rerank.call_args.args[0] == "cancel registration"
    assert mock_rerank.call_args.kwargs["reranker"] is reranker
    assert [candidate["chunk_id"] for candidate in candidates_passed_to_reranker] == ["shared", "dense", "bm25"]
    assert candidates_passed_to_reranker[0]["dense_rank"] == 2
    assert candidates_passed_to_reranker[0]["bm25_rank"] == 1
    assert candidates_passed_to_reranker[0]["rrf_score"] == pytest.approx((1 / 62) + (1 / 61))
    assert [item["chunk_id"] for item in fused] == ["bm25", "dense"]
    assert len(fused) == 2
    assert stats["reranked"] is True
    assert stats["hybrid_candidate_count"] == 3
    assert stats["rerank_candidate_limit"] == 3


def test_reranked_hybrid_results_preserve_retrieval_metadata():
    candidate = {
        **result("shared", score=0.8),
        "content": "candidate text",
        "dense_rank": 1,
        "dense_score": 0.8,
        "bm25_rank": 2,
        "bm25_score": 4.2,
        "rrf_score": 0.03,
    }
    reranker = Mock()
    reranker.predict.return_value = [0.9]

    from src.retrievers.legal_dense_retriever import rerank_legal_results

    reranked = rerank_legal_results("query", [candidate], reranker=reranker)

    assert reranked[0]["dense_rank"] == 1
    assert reranked[0]["dense_score"] == 0.8
    assert reranked[0]["bm25_rank"] == 2
    assert reranked[0]["bm25_score"] == 4.2
    assert reranked[0]["rrf_score"] == 0.03
    assert reranked[0]["rerank_score"] == 0.9
