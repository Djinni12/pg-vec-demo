#!/usr/bin/env python3
"""Compare dense, BM25, and hybrid legal retrieval for a fixed GST query."""

import os
import sys

import psycopg

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.legal_dense_retriever import dense_search_legal_corpus, load_query_model, load_reranker_model, rerank_legal_results
from src.retrievers.legal_hybrid_retriever import (
    DEFAULT_RETRIEVE_LIMIT,
    DEFAULT_TOP_K,
    fuse_legal_results,
)


QUERY = "How can GST registration be cancelled?"


def conn_params_from_env():
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "dbname": os.getenv("POSTGRES_DB", "hybrid_rag"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    }


def print_result_list(title, rows, score_field, limit=10):
    print(f"\n{title}")
    print("-" * len(title))
    for rank, row in enumerate(rows[:limit], 1):
        score = row.get(score_field)
        score_text = f"{score:.4f}" if isinstance(score, float) else str(score)
        print(
            f"{rank:>2}. [{row['document_type'].upper()}] {row['reference']} | "
            f"{row['title']} | {score_field}={score_text} | {row['chunk_id']}"
        )


def print_hybrid(title, rows):
    print(f"\n{title}")
    print("-" * len(title))
    for rank, row in enumerate(rows, 1):
        dense_rank = row["dense_rank"] if row["dense_rank"] is not None else "-"
        bm25_rank = row["bm25_rank"] if row["bm25_rank"] is not None else "-"
        rerank = f" | rerank={row['rerank_score']:.4f}" if "rerank_score" in row else ""
        print(
            f"{rank:>2}. [{row['document_type'].upper()}] {row['reference']} | {row['title']} | "
            f"rrf={row['rrf_score']:.6f} | dense_rank={dense_rank} | bm25_rank={bm25_rank} | "
            f"{row['chunk_id']}{rerank}"
        )


def main():
    conn_params = conn_params_from_env()
    print(f"Query: {QUERY}")
    print(f"Retrieve limit per retriever: {DEFAULT_RETRIEVE_LIMIT}")
    print(f"Hybrid top-k: {DEFAULT_TOP_K}")

    model = load_query_model()
    with psycopg.connect(**conn_params) as conn:
        dense_rows, dense_counts = dense_search_legal_corpus(
            conn,
            QUERY,
            top_k=DEFAULT_RETRIEVE_LIMIT,
            per_table_limit=DEFAULT_RETRIEVE_LIMIT,
            model=model,
            rerank=False,
        )

    bm25_rows, bm25_stats = BM25Retriever(conn_params).retrieve(QUERY, top_k=DEFAULT_RETRIEVE_LIMIT)
    hybrid_candidates = fuse_legal_results(dense_rows, bm25_rows, top_k=DEFAULT_RETRIEVE_LIMIT)
    reranker = load_reranker_model()
    reranked_rows = rerank_legal_results(QUERY, hybrid_candidates, reranker=reranker)[:DEFAULT_TOP_K]
    hybrid_rows = hybrid_candidates[:DEFAULT_TOP_K]

    print(f"\nDense candidates: {len(dense_rows)} {dense_counts}")
    print(
        "BM25 candidates: "
        f"{len(bm25_rows)} acts={bm25_stats['acts_fetched']} "
        f"rules={bm25_stats['rules_fetched']} forms={bm25_stats['forms_fetched']}"
    )
    print(f"Hybrid candidates: dense={len(dense_rows)} bm25={len(bm25_rows)}")

    print_result_list("DENSE TOP 10", dense_rows, "score")
    print_result_list("BM25 TOP 10", bm25_rows, "bm25_score")
    print_hybrid("HYBRID TOP 10", hybrid_rows)
    print_hybrid("HYBRID+RERANKER TOP 10", reranked_rows)


if __name__ == "__main__":
    main()
