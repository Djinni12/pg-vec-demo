"""Print a small GST BM25 and vector-search smoke test."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse

import psycopg

from ingest_gst import DATABASE_URL, MODEL_NAME
from keyword_search import bm25_search
from rrf import reciprocal_rank_fusion, rerank_rows
from search_pgvec import print_result
from vector_search import vector_search


def print_section(title):
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def print_error(error):
    print(f"{error.__class__.__name__}: {error}")


def run_bm25(conn, query, limit):
    print_section("BM25 KEYWORD SEARCH")
    try:
        rows = bm25_search(conn, query, limit)
    except Exception as error:
        print("BM25 search failed.")
        print_error(error)
        return []

    if not rows:
        print("No BM25 matches found.")
        return []

    for index, row in enumerate(rows, 1):
        print(f"{index}.", end=" ")
        print_result(row, "bm25_score")
    return rows


def run_vector(conn, query, limit):
    print_section("VECTOR SEARCH")
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME)
        rows = vector_search(conn, query, limit, model=model)
    except Exception as error:
        print("Vector search failed.")
        print_error(error)
        return []

    if not rows:
        print("No vector matches found.")
        return []

    for index, row in enumerate(rows, 1):
        print(f"{index}.", end=" ")
        print_result(row, "vector_distance")
    return rows


def run_rrf(query, bm25_rows, vector_rows, k, top_k, rerank_vector):
    if rerank_vector:
        print_section("VECTOR-RERANKED RRF FUSED SEARCH")
        try:
            from sentence_transformers import CrossEncoder

            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            vector_rows = rerank_rows(query, vector_rows, reranker)
        except Exception as error:
            print("Vector reranker failed.")
            print_error(error)
            return
    else:
        print_section("RRF FUSED SEARCH")
    rows = reciprocal_rank_fusion([bm25_rows, vector_rows], k=k, top_k=top_k)
    if not rows:
        print("No RRF matches found.")
        return
    for index, item in enumerate(rows, 1):
        print(f"{index}.", end=" ")
        print_result(item["row"])
        print(f"  rrf_score={item['rrf_score']:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="coffee beans")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rerank-vector-before-rrf", action="store_true")
    parser.add_argument("--rerank-before-rrf", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--database-url", default=DATABASE_URL)
    args = parser.parse_args()

    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.rrf_k < 0:
        raise SystemExit("--rrf-k must be non-negative")

    print(f"Query: {args.query}")
    print(f"Limit: {args.limit}")
    rerank_vector = args.rerank_vector_before_rrf or args.rerank_before_rrf

    try:
        with psycopg.connect(args.database_url, autocommit=True) as conn:
            bm25_rows = run_bm25(conn, args.query, args.limit)
            vector_rows = run_vector(conn, args.query, args.limit)
            run_rrf(args.query, bm25_rows, vector_rows, args.rrf_k, args.limit, rerank_vector)
    except Exception as error:
        print("Database connection failed.")
        print_error(error)


if __name__ == "__main__":
    main()
