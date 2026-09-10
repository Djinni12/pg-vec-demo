"""Hybrid dense + BM25 retrieval for GST legal Act, Rule, and Form chunks."""

from __future__ import annotations

from typing import Any

from .bm25_retriever import BM25Retriever
from .legal_dense_retriever import dense_search_legal_corpus, rerank_legal_results
from .rrf import reciprocal_rank_fusion_by_key


DEFAULT_RETRIEVE_LIMIT = 30
DEFAULT_TOP_K = 10
DEFAULT_RRF_K = 60
DEFAULT_RERANK_CANDIDATE_LIMIT = 10


def chunk_key(result: dict[str, Any]) -> str:
    """Return the stable legal chunk key."""
    try:
        return result["chunk_id"]
    except KeyError as exc:
        raise ValueError("Legal retrieval results must include chunk_id") from exc


def _rank_by_chunk_id(results: list[dict[str, Any]]) -> dict[str, int]:
    return {chunk_key(result): rank for rank, result in enumerate(results, 1)}


def _score_by_chunk_id(results: list[dict[str, Any]], score_field: str) -> dict[str, float]:
    return {chunk_key(result): float(result[score_field]) for result in results if result.get(score_field) is not None}


def _base_result(row: dict[str, Any]) -> dict[str, Any]:
    result = {
        "chunk_id": row["chunk_id"],
        "document_type": row["document_type"],
        "reference": row["reference"],
        "title": row["title"],
    }
    for field in ("content", "source_metadata", "snippet"):
        if field in row:
            result[field] = row[field]
    return result


def fuse_legal_results(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
    k: int = DEFAULT_RRF_K,
) -> list[dict[str, Any]]:
    """Fuse dense and BM25 legal chunk results by chunk_id using RRF."""
    dense_ranks = _rank_by_chunk_id(dense_results)
    bm25_ranks = _rank_by_chunk_id(bm25_results)
    dense_scores = _score_by_chunk_id(dense_results, "score")
    bm25_scores = _score_by_chunk_id(bm25_results, "bm25_score")

    fused = reciprocal_rank_fusion_by_key([dense_results, bm25_results], chunk_key, k=k, top_k=top_k)
    hybrid_results = []
    for item in fused:
        chunk_id = chunk_key(item["row"])
        result = _base_result(item["row"])
        result.update(
            {
                "dense_rank": dense_ranks.get(chunk_id),
                "dense_score": dense_scores.get(chunk_id),
                "bm25_rank": bm25_ranks.get(chunk_id),
                "bm25_score": bm25_scores.get(chunk_id),
                "rrf_score": item["rrf_score"],
            }
        )
        hybrid_results.append(result)
    return hybrid_results


def hybrid_search_legal_corpus(
    conn,
    query: str,
    conn_params: dict[str, Any],
    retrieve_limit: int = DEFAULT_RETRIEVE_LIMIT,
    top_k: int = DEFAULT_TOP_K,
    k: int = DEFAULT_RRF_K,
    model=None,
    reranker=None,
    rerank: bool = False,
    rerank_candidate_limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run dense and BM25 retrievers, fuse by chunk_id, and optionally rerank after RRF."""
    if retrieve_limit < 1:
        raise ValueError("retrieve_limit must be positive")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if k < 0:
        raise ValueError("k must be non-negative")
    if not query.strip():
        return [], {"dense_count": 0, "bm25_count": 0}
    if rerank_candidate_limit is None:
        rerank_candidate_limit = DEFAULT_RERANK_CANDIDATE_LIMIT
    if rerank_candidate_limit < 1:
        raise ValueError("rerank_candidate_limit must be positive")

    dense_results, dense_counts = dense_search_legal_corpus(
        conn,
        query,
        top_k=retrieve_limit,
        per_table_limit=retrieve_limit,
        model=model,
        rerank=False,
    )
    bm25_results, bm25_stats = BM25Retriever(conn_params).retrieve(query, top_k=retrieve_limit)
    should_rerank = rerank or reranker is not None
    fusion_limit = rerank_candidate_limit if should_rerank else top_k
    hybrid_candidates = fuse_legal_results(dense_results, bm25_results, top_k=fusion_limit, k=k)
    if should_rerank:
        hybrid_results = rerank_legal_results(query, hybrid_candidates, reranker=reranker)[:top_k]
    else:
        hybrid_results = hybrid_candidates[:top_k]

    stats = {
        "dense_count": len(dense_results),
        "bm25_count": len(bm25_results),
        "hybrid_candidate_count": len(hybrid_candidates),
        "dense_counts": dense_counts,
        "bm25_stats": bm25_stats,
        "retrieve_limit": retrieve_limit,
        "top_k": top_k,
        "rrf_k": k,
        "reranked": should_rerank,
        "rerank_candidate_limit": rerank_candidate_limit,
    }
    return hybrid_results, stats
