"""Experimental query-encoder comparison over existing BGE-M3 document vectors.

This script is strictly query-only. It does not create document embeddings, does not
write files, and does not modify PostgreSQL. It searches the existing BGE-M3
VECTOR(1024) document tables with two query encoders:

1. BGE-M3 query encoder: normal valid same-model retrieval.
2. multilingual-e5-large query encoder: CROSS-MODEL / INVALID FOR FORMAL
   RETRIEVAL EVALUATION because E5 query vectors are compared against BGE-M3
   document vectors.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv(dotenv_path=ROOT_DIR / ".env")
if os.getenv("HF_TOKEN") and not os.getenv("HUGGINGFACE_HUB_TOKEN"):
    os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

from sentence_transformers import SentenceTransformer

from act_embedder import MODEL_NAME as BGE_M3_MODEL
from ingest_gst import DATABASE_URL

E5_MODEL = "intfloat/multilingual-e5-large"
CORPUS_VECTOR_DIMENSION = 1024
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


def build_search_sql(vector, top_k, per_table_limit):
    vector_sql = sql_quote(vector) + "::vector"
    return f"""
WITH act AS (
    SELECT
        'act' AS document_type,
        chunk_id,
        'Section ' || COALESCE(section_number, '') AS reference,
        COALESCE(section_title, '') AS title,
        1 - (embedding <=> {vector_sql}) AS score
    FROM act_chunks
    ORDER BY embedding <=> {vector_sql}
    LIMIT {int(per_table_limit)}
), rules AS (
    SELECT
        'rule' AS document_type,
        chunk_id,
        'Rule ' || COALESCE(rule_number, '') AS reference,
        COALESCE(rule_title, '') AS title,
        1 - (embedding <=> {vector_sql}) AS score
    FROM rule_chunks
    ORDER BY embedding <=> {vector_sql}
    LIMIT {int(per_table_limit)}
), forms AS (
    SELECT
        'form' AS document_type,
        chunk_id,
        COALESCE(form_number, '') AS reference,
        COALESCE(title, form_title, '') AS title,
        1 - (embedding <=> {vector_sql}) AS score
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
    score
FROM candidates
ORDER BY score DESC
LIMIT {int(top_k)};
"""


def encode_query(model, query, prefix=None):
    model_input = f"{prefix}{query}" if prefix else query
    start = time.perf_counter()
    embedding = model.encode(model_input, normalize_embeddings=True, show_progress_bar=False)
    elapsed = time.perf_counter() - start
    dimension = len(embedding)
    if dimension != CORPUS_VECTOR_DIMENSION:
        raise ValueError(
            f"Query embedding dimension {dimension} does not match existing corpus dimension {CORPUS_VECTOR_DIMENSION}"
        )
    return embedding, elapsed, dimension


def rows_from_psycopg(conn, sql):
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


def rows_from_docker(sql, compose_service="db"):
    cmd = [
        "docker",
        "compose",
        "exec",
        "-T",
        compose_service,
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
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        document_type, chunk_id, reference, title, score = line.split("\t", 4)
        rows.append((document_type, chunk_id, reference, title, float(score)))
    return rows


def search_existing_bge_vectors(model, query, top_k, per_table_limit, prefix=None, conn=None, use_docker=False):
    embedding, embedding_time, dimension = encode_query(model, query, prefix=prefix)
    sql = build_search_sql(vector_literal(embedding), top_k=top_k, per_table_limit=per_table_limit)
    start = time.perf_counter()
    rows = rows_from_docker(sql) if use_docker else rows_from_psycopg(conn, sql)
    retrieval_time = time.perf_counter() - start
    return rows, embedding_time, retrieval_time, dimension


def print_rows(label, rows):
    print(label)
    print("-" * len(label))
    for rank, row in enumerate(rows, 1):
        document_type, chunk_id, reference, title, score = row
        print(f"{rank}. {document_type} | {reference} | {title} | score={float(score):.4f} | {chunk_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Run one query instead of the built-in query set")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--per-table-limit", type=int, default=None)
    parser.add_argument("--database-url", default=DATABASE_URL)
    args = parser.parse_args()

    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")
    per_table_limit = args.per_table_limit or args.top_k
    queries = [args.query] if args.query else DEFAULT_QUERIES

    print("Loading query encoders...")
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    models = {
        "BGE-M3 query → BGE-M3 document vectors": SentenceTransformer(BGE_M3_MODEL, local_files_only=True),
        "E5 query → BGE-M3 document vectors": SentenceTransformer(E5_MODEL, token=hf_token),
    }
    print(f"BGE-M3 model: {BGE_M3_MODEL}")
    print(f"E5 model: {E5_MODEL}")
    print(f"Existing document vector tables: act_chunks, rule_chunks, form_chunks")
    print(f"Existing document vector dimension: {CORPUS_VECTOR_DIMENSION}")
    print("E5 query format: query: <user query>")
    print("CROSS-MODEL / INVALID FOR FORMAL RETRIEVAL EVALUATION: E5 query → BGE-M3 document-vector results")

    import psycopg

    use_docker = False
    conn_context = None
    conn = None
    try:
        conn_context = psycopg.connect(args.database_url, autocommit=True)
        conn = conn_context.__enter__()
    except psycopg.OperationalError as error:
        print(f"Direct psycopg connection failed; using docker compose psql fallback: {error}")
        use_docker = True

    try:
        for query in queries:
            print()
            print(f"Query: {query}")
            print("=" * (len(query) + 7))

            bge_rows, bge_embed_time, bge_retrieval_time, bge_dim = search_existing_bge_vectors(
                models["BGE-M3 query → BGE-M3 document vectors"],
                query,
                top_k=args.top_k,
                per_table_limit=per_table_limit,
                conn=conn,
                use_docker=use_docker,
            )
            e5_rows, e5_embed_time, e5_retrieval_time, e5_dim = search_existing_bge_vectors(
                models["E5 query → BGE-M3 document vectors"],
                query,
                top_k=args.top_k,
                per_table_limit=per_table_limit,
                prefix="query: ",
                conn=conn,
                use_docker=use_docker,
            )

            print(f"BGE-M3: query_dim={bge_dim} embedding_time={bge_embed_time:.3f}s retrieval_time={bge_retrieval_time:.3f}s")
            print(
                "E5 CROSS-MODEL / INVALID FOR FORMAL RETRIEVAL EVALUATION: "
                f"query_dim={e5_dim} embedding_time={e5_embed_time:.3f}s retrieval_time={e5_retrieval_time:.3f}s"
            )
            print()
            print_rows("BGE-M3 query → BGE-M3 document vectors", bge_rows)
            print()
            print_rows("E5 query → BGE-M3 document vectors (CROSS-MODEL / INVALID FOR FORMAL RETRIEVAL EVALUATION)", e5_rows)
    finally:
        if conn_context is not None:
            conn_context.__exit__(None, None, None)


if __name__ == "__main__":
    main()
