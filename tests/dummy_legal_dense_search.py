"""Print reranked unified GST legal retrieval results from act/rule/form pgvector tables.

Run examples:
    python tests/dummy_legal_dense_search.py --examples
    python tests/dummy_legal_dense_search.py "How do I register for GST?" --top-k 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

from ingest_gst import DATABASE_URL
from legal_dense_retriever import dense_search_legal_corpus, load_query_model, load_reranker_model

load_dotenv(dotenv_path=ROOT_DIR / ".env")

EXAMPLE_QUERIES = [
    "How do I register for GST?",
    "Which form is used for GST registration?",
    "What are the conditions for input tax credit?",
    "How can registration be cancelled?",
    "Which form is used to claim a refund?",
]


def print_result(rank, result):
    metadata = result["source_metadata"] or {}
    if result["document_type"] == "form":
        reference = metadata.get("form_code") or result["reference"]
    else:
        reference = result["reference"]
    score_label = "rerank_score" if "rerank_score" in result else "score"
    print(
        f"{rank}. {result['document_type']} | {reference} | "
        f"{result['title']} | {score_label}={result['score']:.4f}"
    )
    if "dense_score" in result:
        print(f"   dense_score={result['dense_score']:.4f}")
    if result["document_type"] == "form":
        print(
            f"   chunk_id={result['chunk_id']} | form_uid={metadata.get('form_uid')} | "
            f"pages={metadata.get('source_start_page')}-{metadata.get('source_end_page')}"
        )
    else:
        print(f"   chunk_id={result['chunk_id']} | metadata={metadata}")


def run_query(conn, query, top_k, per_table_limit, model, reranker):
    results, counts = dense_search_legal_corpus(
        conn,
        query,
        top_k=top_k,
        per_table_limit=per_table_limit,
        model=model,
        reranker=reranker,
    )
    print()
    print(f"Query: {query}")
    print("=" * (len(query) + 7))
    print("Mode: RERANKED")
    print(f"Candidates: Act={counts.get('act', 0)} Rules={counts.get('rule', 0)} Forms={counts.get('form', 0)}")
    for rank, result in enumerate(results, 1):
        print_result(rank, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default=EXAMPLE_QUERIES[0])
    parser.add_argument("--examples", action="store_true", help="Run the built-in sanity queries")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--per-table-limit", type=int, default=None)
    parser.add_argument("--database-url", default=DATABASE_URL)
    args = parser.parse_args()

    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")
    per_table_limit = args.per_table_limit or args.top_k
    if per_table_limit < 1:
        raise SystemExit("--per-table-limit must be positive")

    queries = EXAMPLE_QUERIES if args.examples else [args.query]

    try:
        import psycopg

        model = load_query_model()
        reranker = load_reranker_model()
        with psycopg.connect(args.database_url, autocommit=True) as conn:
            for query in queries:
                run_query(conn, query, args.top_k, per_table_limit, model, reranker)
    except Exception as error:
        print("Unified legal dense search failed.")
        print(f"{error.__class__.__name__}: {error}")
        print("\nIf local Python cannot reach Docker Postgres, verify the tables with:")
        print('sudo docker compose exec db psql -U postgres -d hybrid_rag -c "SELECT count(*) FROM act_chunks; SELECT count(*) FROM rule_chunks; SELECT count(*) FROM form_chunks;"')


if __name__ == "__main__":
    main()
