"""Print a small GST Forms pgvector semantic-search smoke test.

Run examples:
    python tests/dummy_form_search_test.py "Which form is used for GST registration?"
    python tests/dummy_form_search_test.py --examples
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

from act_embedder import MODEL_NAME
from ingest_gst import DATABASE_URL

load_dotenv(dotenv_path=ROOT_DIR / ".env")

EXAMPLE_QUERIES = [
    "Which form is used for GST registration?",
    "How do I withdraw from the composition scheme?",
    "Which form is used for a GST refund application?",
    "Which form is used for TDS return?",
]


def embed_query(query):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    return model.encode(query, normalize_embeddings=True)


def search_forms(conn, query, limit):
    from pgvector.psycopg import register_vector

    register_vector(conn)
    query_embedding = embed_query(query)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                form_code,
                form_uid,
                title,
                source_start_page,
                source_end_page,
                chunk_strategy,
                token_count,
                embedding <=> %s AS cosine_distance,
                content
            FROM form_chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query_embedding, query_embedding, limit),
        )
        return cur.fetchall()


def print_results(query, rows):
    print()
    print(f"Query: {query}")
    print("=" * (len(query) + 7))
    if not rows:
        print("No results.")
        return

    for rank, row in enumerate(rows, 1):
        (
            form_code,
            form_uid,
            title,
            source_start_page,
            source_end_page,
            chunk_strategy,
            token_count,
            cosine_distance,
            content,
        ) = row
        similarity = 1 - float(cosine_distance)
        preview = " ".join((content or "").split())[:260]
        print(f"{rank}. form_code={form_code} | form_uid={form_uid}")
        print(f"   title={title}")
        print(f"   pages={source_start_page}-{source_end_page} | strategy={chunk_strategy} | tokens={token_count}")
        print(f"   cosine_distance={float(cosine_distance):.4f} | cosine_similarity={similarity:.4f}")
        print(f"   preview={preview}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default=EXAMPLE_QUERIES[0])
    parser.add_argument("--examples", action="store_true", help="Run four built-in sanity queries")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--database-url", default=DATABASE_URL)
    args = parser.parse_args()

    if args.limit < 1:
        raise SystemExit("--limit must be positive")

    queries = EXAMPLE_QUERIES if args.examples else [args.query]

    try:
        import psycopg

        with psycopg.connect(args.database_url, autocommit=True) as conn:
            for query in queries:
                rows = search_forms(conn, query, args.limit)
                print_results(query, rows)
    except Exception as error:
        print("Form search failed.")
        print(f"{error.__class__.__name__}: {error}")
        print("\nIf your Docker DB needs sudo, first verify from the terminal:")
        print('sudo docker compose exec db psql -U postgres -d hybrid_rag -c "SELECT count(*) FROM form_chunks;"')


if __name__ == "__main__":
    main()
