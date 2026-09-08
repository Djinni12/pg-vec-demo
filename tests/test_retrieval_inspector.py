from unittest.mock import Mock, patch

import pytest

from src.retrieval_inspector import LoadedModels, inspect_retrieval


def test_inspect_retrieval_returns_stage_timings_and_metadata():
    models = LoadedModels(embedding_model=object(), reranker=object(), initialization_ms=123.4)
    dense_rows = [
        {
            "chunk_id": "dense",
            "document_type": "act",
            "reference": "Section 1",
            "title": "Registration",
            "content": "dense text",
            "score": 0.7,
        }
    ]
    bm25_rows = [
        {
            "chunk_id": "bm25",
            "document_type": "rule",
            "reference": "Rule 22",
            "title": "Cancellation",
            "content": "bm25 text",
            "bm25_score": 5.0,
            "rank": 1,
        }
    ]

    with patch("src.retrieval_inspector.psycopg.connect") as connect, patch(
        "src.retrieval_inspector.dense_search_legal_corpus", return_value=(dense_rows, {"act": 1})
    ) as dense_search, patch("src.retrieval_inspector.BM25Retriever") as bm25_cls, patch(
        "src.retrieval_inspector.rerank_legal_results"
    ) as rerank:
        connect.return_value.__enter__.return_value = object()
        bm25 = Mock()
        bm25.retrieve.return_value = (bm25_rows, {"backend": "pg_textsearch"})
        bm25_cls.return_value = bm25
        rerank.return_value = [{**bm25_rows[0], "rrf_score": 1 / 61, "rerank_score": 0.9}]

        response = inspect_retrieval(
            "cancel registration",
            top_k=1,
            models=models,
            db_url="dbname=hybrid_rag user=postgres",
        )

    dense_search.assert_called_once()
    bm25.retrieve.assert_called_once_with("cancel registration", top_k=30)
    rerank.assert_called_once()
    assert response["timings_ms"]["total"] >= 0
    assert response["metadata"]["embedding_model"] == "BAAI/bge-m3"
    assert response["metadata"]["reranker_model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert response["metadata"]["embedding_dimension"] == 1024
    assert response["metadata"]["bm25_method"] == "pg_textsearch BM25"
    assert response["metadata"]["dense_candidate_count"] == 1
    assert response["metadata"]["bm25_candidate_count"] == 1
    assert response["metadata"]["hybrid_candidate_count"] == 2
    assert response["results"][0]["reranker_score"] == 0.9


def test_inspect_retrieval_rejects_empty_query():
    with pytest.raises(ValueError, match="query cannot be empty"):
        inspect_retrieval(" ", models=LoadedModels(None, None, 0.0))
