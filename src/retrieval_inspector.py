"""Inspection wrapper for the GST legal hybrid retrieval pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import time
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.conninfo import conninfo_to_dict

from src.embedders.act_embedder import MODEL_NAME as EMBEDDING_MODEL_NAME
from src.ingestors.ingest_gst import DATABASE_URL as DEFAULT_DATABASE_URL
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.legal_dense_retriever import (
    DEFAULT_RERANKER_MODEL,
    dense_search_legal_corpus,
    load_query_model,
    load_reranker_model,
    rerank_legal_results,
)
from src.retrievers.legal_hybrid_retriever import (
    DEFAULT_RETRIEVE_LIMIT,
    DEFAULT_RRF_K,
    DEFAULT_TOP_K,
    fuse_legal_results,
)


EMBEDDING_DIMENSION = 1024
DENSE_METHOD = "pgvector + HNSW"
BM25_METHOD = "pg_textsearch BM25 or pg_search BM25, detected by existing BM25 retriever"


@dataclass
class LoadedModels:
    embedding_model: Any
    reranker: Any
    initialization_ms: float


def database_url() -> str:
    """Return the configured PostgreSQL connection string."""
    return os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL


def load_models() -> LoadedModels:
    """Load retrieval models once for reuse across requests."""
    start = time.perf_counter()
    embedding_model = load_query_model(EMBEDDING_MODEL_NAME)
    reranker = load_reranker_model(DEFAULT_RERANKER_MODEL)
    return LoadedModels(
        embedding_model=embedding_model,
        reranker=reranker,
        initialization_ms=_elapsed_ms(start),
    )


DEFAULT_RERANK_CANDIDATE_LIMIT = 10


def inspect_retrieval(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    models: LoadedModels,
    db_url: str | None = None,
    retrieve_limit: int = DEFAULT_RETRIEVE_LIMIT,
    rrf_k: int = DEFAULT_RRF_K,
    rerank_candidate_limit: int = DEFAULT_RERANK_CANDIDATE_LIMIT,
) -> dict[str, Any]:
    """Run dense, BM25, RRF, and reranker stages with per-query timings."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if retrieve_limit < 1:
        raise ValueError("retrieve_limit must be positive")
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    if not query.strip():
        raise ValueError("query cannot be empty")

    db_url = db_url or database_url()
    conn_params = conninfo_to_dict(db_url)
    total_start = time.perf_counter()

    def _run_dense():
        d_start = time.perf_counter()
        with psycopg.connect(db_url) as conn:
            d_res, d_counts = dense_search_legal_corpus(
                conn,
                query,
                top_k=retrieve_limit,
                per_table_limit=retrieve_limit,
                model=models.embedding_model,
                rerank=False,
            )
        return d_res, d_counts, _elapsed_ms(d_start)

    def _run_bm25():
        b_start = time.perf_counter()
        b_res, b_stats = BM25Retriever(conn_params).retrieve(query, top_k=retrieve_limit)
        return b_res, b_stats, _elapsed_ms(b_start)

    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_dense = executor.submit(_run_dense)
        fut_bm25 = executor.submit(_run_bm25)
        dense_results, dense_counts, dense_ms = fut_dense.result()
        bm25_results, bm25_stats, bm25_ms = fut_bm25.result()

    rrf_start = time.perf_counter()
    hybrid_candidates = fuse_legal_results(
        dense_results,
        bm25_results,
        top_k=retrieve_limit,
        k=rrf_k,
    )
    rrf_ms = _elapsed_ms(rrf_start)

    reranker_start = time.perf_counter()
    candidates_to_rerank = hybrid_candidates[:rerank_candidate_limit]
    reranked_results = rerank_legal_results(query, candidates_to_rerank, reranker=models.reranker)[:top_k]
    reranker_ms = _elapsed_ms(reranker_start)

    ranked_final = [_normalize_result(result, index) for index, result in enumerate(reranked_results, 1)]
    total_ms = _elapsed_ms(total_start)

    metadata = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker_model": DEFAULT_RERANKER_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "dense_candidate_count": len(dense_results),
        "bm25_candidate_count": len(bm25_results),
        "hybrid_candidate_count": len(hybrid_candidates),
        "final_top_k": top_k,
        "rrf_k": rrf_k,
        "dense_method": DENSE_METHOD,
        "bm25_method": _bm25_method(bm25_stats),
        "model_initialization_ms": models.initialization_ms,
        "retrieve_limit": retrieve_limit,
        "dense_counts": dense_counts,
        "bm25_stats": bm25_stats,
    }

    return {
        "query": query,
        "results": ranked_final,
        "dense_results": [_normalize_result(result, index) for index, result in enumerate(dense_results, 1)],
        "bm25_results": [_normalize_result(result, index) for index, result in enumerate(bm25_results, 1)],
        "hybrid_results": [_normalize_result(result, index) for index, result in enumerate(hybrid_candidates, 1)],
        "timings_ms": {
            "total": total_ms,
            "dense": dense_ms,
            "bm25": bm25_ms,
            "rrf": rrf_ms,
            "reranker": reranker_ms,
        },
        "metadata": metadata,
        "models": {
            "embedding": EMBEDDING_MODEL_NAME,
            "reranker": DEFAULT_RERANKER_MODEL,
        },
        "config": {
            "dense_candidate_count": retrieve_limit,
            "bm25_candidate_count": retrieve_limit,
            "rrf_k": rrf_k,
            "final_top_k": top_k,
            "dense_method": DENSE_METHOD,
            "bm25_method": metadata["bm25_method"],
        },
    }


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def _bm25_method(stats: dict[str, Any]) -> str:
    backend = stats.get("backend")
    if backend:
        return f"{backend} BM25"
    return BM25_METHOD


def _normalize_result(result: dict[str, Any], rank: int) -> dict[str, Any]:
    content = result.get("content") or result.get("snippet") or ""
    return {
        "rank": rank,
        "document_type": result.get("document_type"),
        "reference": result.get("reference"),
        "title": result.get("title"),
        "content": content,
        "snippet": result.get("snippet") or _snippet(content),
        "chunk_id": result.get("chunk_id"),
        "dense_rank": result.get("dense_rank"),
        "dense_score": result.get("dense_score", result.get("score")),
        "bm25_rank": result.get("bm25_rank", result.get("rank") if result.get("bm25_score") is not None else None),
        "bm25_score": result.get("bm25_score"),
        "rrf_score": result.get("rrf_score"),
        "reranker_score": result.get("rerank_score"),
    }


def _snippet(content: str, max_length: int = 280) -> str:
    if len(content) <= max_length:
        return content
    truncate_at = content.rfind(" ", 0, max_length)
    if truncate_at == -1:
        truncate_at = max_length
    return content[:truncate_at] + "..."
