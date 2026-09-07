"""Print a small GST BM25 and vector-search smoke test."""

import argparse

import psycopg
from pgvector.psycopg import register_vector

from ingest_gst import DATABASE_URL, MODEL_NAME
from keyword_search import bm25_search
from search_pgvec import print_result


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
        return

    if not rows:
        print("No BM25 matches found.")
        return

    for index, row in enumerate(rows, 1):
        print(f"{index}.", end=" ")
        print_result(row, "bm25_score")


def run_vector(conn, query, limit):
    print_section("VECTOR SEARCH")
    try:
        from sentence_transformers import SentenceTransformer

        register_vector(conn)
        model = SentenceTransformer(MODEL_NAME)
        query_embedding = model.encode(query)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT code, description, embedding <=> %s AS distance,
                       cgst_rate_pct, sgst_utgst_rate_pct, igst_rate_pct, metadata
                FROM gst_documents
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_embedding, query_embedding, limit),
            )
            rows = cur.fetchall()
    except Exception as error:
        print("Vector search failed.")
        print_error(error)
        return

    if not rows:
        print("No vector matches found.")
        return

    for index, row in enumerate(rows, 1):
        print(f"{index}.", end=" ")
        print_result(row, "vector_distance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="coffee beans")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--database-url", default=DATABASE_URL)
    args = parser.parse_args()

    if args.limit < 1:
        raise SystemExit("--limit must be positive")

    print(f"Query: {args.query}")
    print(f"Limit: {args.limit}")

    try:
        with psycopg.connect(args.database_url, autocommit=True) as conn:
            run_bm25(conn, args.query, args.limit)
            run_vector(conn, args.query, args.limit)
    except Exception as error:
        print("Database connection failed.")
        print_error(error)


if __name__ == "__main__":
    main()
