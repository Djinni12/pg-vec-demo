"""Run BGE-M3 dense retrieval over the stored GST legal corpus.

This script embeds the user query with BAAI/bge-m3 and searches the existing
pgvector tables: act_chunks, rule_chunks, and form_chunks. It prints retrieved
results directly in the terminal.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv(dotenv_path=ROOT_DIR / ".env")

from sentence_transformers import SentenceTransformer

from act_embedder import MODEL_NAME as BGE_M3_MODEL
from ingest_gst import DATABASE_URL

DEFAULT_QUERIES = [
    "How do I register for GST?",
    "Which form is used for GST registration?",
    "What are the conditions for input tax credit?",
    "How can registration be cancelled?",
    "Which form is used to claim a refund?",
]


def sql_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def vector_literal(embedding):
    return "[" + ",".join(str(float(value)) for value in embedding) + "]"


def build_search_sql(vector, top_k, per_table_limit, include_content):
    vector_sql = sql_quote(vector) + "::vector"
    content_select = ", left(regexp_replace(content, E'\\\\s+', ' ', 'g'), 500) AS snippet" if include_content else ""
    return f"""
WITH act AS (
    SELECT
        'act' AS document_type,
        chunk_id,
        'Section ' || COALESCE(section_number, '') AS reference,
        COALESCE(section_title, '') AS title,
        1 - (embedding <=> {vector_sql}) AS score,
        embedding <=> {vector_sql} AS distance
        {content_select}
    FROM act_chunks
    ORDER BY embedding <=> {vector_sql}
    LIMIT {int(per_table_limit)}
), rules AS (
    SELECT
        'rule' AS document_type,
        chunk_id,
        'Rule ' || COALESCE(rule_number, '') AS reference,
        COALESCE(rule_title, '') AS title,
        1 - (embedding <=> {vector_sql}) AS score,
        embedding <=> {vector_sql} AS distance
        {content_select}
    FROM rule_chunks
    ORDER BY embedding <=> {vector_sql}
    LIMIT {int(per_table_limit)}
), forms AS (
    SELECT
        'form' AS document_type,
        chunk_id,
        COALESCE(form_number, '') AS reference,
        COALESCE(title, form_title, '') AS title,
        1 - (embedding <=> {vector_sql}) AS score,
        embedding <=> {vector_sql} AS distance
        {content_select}
    FROM form_chunks
    ORDER BY embedding <=> {vector_sql}
    LIMIT {int(per_table_limit)}
), candidates AS (
    SELECT * FROM act
    UNION ALL SELECT * FROM rules
    UNION ALL SELECT * FROM forms
)
SELECT
    document_type,
    chunk_id,
    replace(reference, E'\\t', ' ') AS reference,
    replace(title, E'\\t', ' ') AS title,
    score,
    distance{', replace(snippet, chr(9), chr(32)) AS snippet' if include_content else ''}
FROM candidates
ORDER BY score DESC
LIMIT {int(top_k)};
"""


def parse_rows(output, include_content):
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if include_content:
            document_type, chunk_id, reference, title, score, distance, snippet = parts[:7]
        else:
            document_type, chunk_id, reference, title, score, distance = parts[:6]
            snippet = ""
        rows.append(
            {
                "document_type": document_type,
                "chunk_id": chunk_id,
                "reference": reference,
                "title": title,
                "score": float(score),
                "distance": float(distance),
                "snippet": snippet,
            }
        )
    return rows


def search_with_psycopg(database_url, sql, include_content):
    import psycopg

    with psycopg.connect(database_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = []
            for row in cur.fetchall():
                item = {
                    "document_type": row[0],
                    "chunk_id": row[1],
                    "reference": row[2],
                    "title": row[3],
                    "score": float(row[4]),
                    "distance": float(row[5]),
                    "snippet": row[6] if include_content else "",
                }
                rows.append(item)
            return rows


def search_with_docker(sql, include_content):
    cmd = [
        "docker",
        "compose",
        "exec",
        "-T",
        "db",
        "psql",
        "-U",
        "postgres",
        "-d",
        "hybrid_rag",
        "-At",
        "-F",
        "\t",
    ]
    completed = subprocess.run(cmd, input=sql, text=True, capture_output=True, check=True)
    return parse_rows(completed.stdout, include_content)


def print_results(query, rows, counts, embedding_time, retrieval_time, query_dimension, show_snippets):
    print()
    print(f"Query: {query}")
    print("=" * (len(query) + 7))
    print(f"Model: {BGE_M3_MODEL}")
    print(f"Query embedding dimension: {query_dimension}")
    print(f"Query embedding time: {embedding_time:.3f}s")
    print(f"Retrieval time: {retrieval_time:.3f}s")
    print(f"Candidates fetched: Act={counts['act']} Rules={counts['rule']} Forms={counts['form']}")
    print()
    for rank, row in enumerate(rows, 1):
        print(
            f"{rank}. {row['document_type']} | {row['reference']} | {row['title']} | "
            f"score={row['score']:.4f} | distance={row['distance']:.4f}"
        )
        print(f"   chunk_id={row['chunk_id']}")
        if show_snippets and row["snippet"]:
            print(f"   snippet={row['snippet']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Run one query instead of the built-in sample queries")
    parser.add_argument("--examples", action="store_true", help="Run the built-in sanity queries")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--per-table-limit", type=int, default=None)
    parser.add_argument("--no-snippets", action="store_true", help="Do not print content snippets")
    parser.add_argument("--database-url", default=DATABASE_URL)
    args = parser.parse_args()

    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")
    per_table_limit = args.per_table_limit or args.top_k
    if per_table_limit < 1:
        raise SystemExit("--per-table-limit must be positive")

    if args.examples:
        queries = DEFAULT_QUERIES
    elif args.query:
        queries = [args.query]
    else:
        queries = [DEFAULT_QUERIES[0]]

    print("Loading BGE-M3 query embedding model...")
    model = SentenceTransformer(BGE_M3_MODEL, local_files_only=True)

    use_docker = False
    try:
        import psycopg

        with psycopg.connect(args.database_url, autocommit=True):
            pass
    except Exception as error:
        print(f"Direct psycopg connection failed; using docker compose psql fallback: {error}")
        use_docker = True

    for query in queries:
        start = time.perf_counter()
        embedding = model.encode(query, normalize_embeddings=True, show_progress_bar=False)
        embedding_time = time.perf_counter() - start
        sql = build_search_sql(
            vector_literal(embedding),
            top_k=args.top_k,
            per_table_limit=per_table_limit,
            include_content=not args.no_snippets,
        )
        start = time.perf_counter()
        rows = search_with_docker(sql, not args.no_snippets) if use_docker else search_with_psycopg(args.database_url, sql, not args.no_snippets)
        retrieval_time = time.perf_counter() - start
        counts = {"act": per_table_limit, "rule": per_table_limit, "form": per_table_limit}
        print_results(query, rows, counts, embedding_time, retrieval_time, len(embedding), not args.no_snippets)


if __name__ == "__main__":
    main()
