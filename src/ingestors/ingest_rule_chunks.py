"""Load CGST Rules chunk embeddings into PostgreSQL/pgvector."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from ingest_act_chunks import EXPECTED_EMBEDDING_DIMENSION, load_embedding_records
from ingest_gst import DATABASE_URL


load_dotenv()

DEFAULT_EMBEDDINGS_PATH = Path("data/rules/gst_rules_bge_m3_embeddings.jsonl")
RULESET_NAME = "Central Goods and Services Tax Rules, 2017"


def validate_rule_record(record, source_label):
    """Validate and normalize one rule chunk embedding record."""
    if not isinstance(record, dict):
        raise ValueError(f"{source_label}: expected object, got {type(record).__name__}")

    chunk_id = str(record.get("chunk_id") or "").strip()
    content = str(record.get("text") or "").strip()
    token_count = record.get("token_count")
    metadata = record.get("metadata") or {}
    embedding = record.get("embedding")

    if not chunk_id:
        raise ValueError(f"{source_label}: missing chunk_id")
    if not content:
        raise ValueError(f"{source_label}: empty content for {chunk_id}")
    if not isinstance(token_count, int):
        raise ValueError(f"{source_label}: token_count must be an integer for {chunk_id}")
    if not isinstance(metadata, dict):
        raise ValueError(f"{source_label}: metadata must be an object for {chunk_id}")
    if not isinstance(embedding, list):
        raise ValueError(f"{source_label}: embedding must be a list for {chunk_id}")
    if len(embedding) != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            f"{source_label}: {chunk_id} embedding dimension {len(embedding)} != {EXPECTED_EMBEDDING_DIMENSION}"
        )

    try:
        embedding = [float(value) for value in embedding]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source_label}: non-numeric embedding value for {chunk_id}") from error

    subrule_numbers = metadata.get("subrule_numbers") or []
    if not isinstance(subrule_numbers, list):
        raise ValueError(f"{source_label}: subrule_numbers must be a list for {chunk_id}")

    normalized = {
        "chunk_id": chunk_id,
        "rule_number": str(metadata.get("rule_number") or "").strip(),
        "rule_title": str(metadata.get("rule_title") or "").strip(),
        "chapter": metadata.get("chapter"),
        "chapter_title": metadata.get("chapter_title"),
        "subrule_numbers": [str(value) for value in subrule_numbers],
        "status": str(metadata.get("status") or "").strip(),
        "content": content,
        "token_count": token_count,
        "chunk_strategy": str(metadata.get("chunk_strategy") or "").strip(),
        "embedding": embedding,
    }

    for field in ("rule_number", "rule_title", "status", "chunk_strategy"):
        if not normalized[field]:
            raise ValueError(f"{source_label}: missing {field} for {chunk_id}")

    return normalized


def load_rule_records(path=DEFAULT_EMBEDDINGS_PATH):
    """Read and validate rule chunk embedding records."""
    records = []
    failures = []
    try:
        raw_records = load_embedding_records(path)
        for index, raw in enumerate(raw_records, 1):
            records.append(validate_rule_record(raw, f"{path}:{index}"))
    except Exception as error:
        failures.append(f"{path}: {error}")
    return records, failures


def create_rule_chunks_table(conn):
    """Create the rule_chunks table and vector index if schema.sql was not run."""
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rule_chunks (
            chunk_id TEXT PRIMARY KEY,
            rule_number TEXT NOT NULL,
            rule_title TEXT NOT NULL,
            chapter TEXT,
            chapter_title TEXT,
            subrule_numbers TEXT[] NOT NULL DEFAULT '{}',
            status TEXT NOT NULL,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL CHECK (token_count > 0),
            chunk_strategy TEXT NOT NULL,
            embedding VECTOR(1024) NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS rule_chunks_embedding_hnsw_idx
            ON rule_chunks USING hnsw (embedding vector_cosine_ops)
        """
    )


def existing_chunk_ids(conn, chunk_ids):
    """Return chunk ids already present in rule_chunks."""
    if not chunk_ids:
        return set()
    rows = conn.execute("SELECT chunk_id FROM rule_chunks WHERE chunk_id = ANY(%s)", (chunk_ids,)).fetchall()
    return {row[0] for row in rows}


def save_rule_records(conn, records):
    """Upsert rule chunks; caller owns transaction commit/rollback."""
    create_rule_chunks_table(conn)
    existing = existing_chunk_ids(conn, [record["chunk_id"] for record in records])
    inserted = 0
    updated = 0

    with conn.cursor() as cur:
        for record in records:
            cur.execute(
                """
                INSERT INTO rule_chunks (
                    chunk_id, rule_number, rule_title, chapter, chapter_title,
                    subrule_numbers, status, content, token_count, chunk_strategy, embedding
                ) VALUES (
                    %(chunk_id)s, %(rule_number)s, %(rule_title)s, %(chapter)s,
                    %(chapter_title)s, %(subrule_numbers)s, %(status)s, %(content)s,
                    %(token_count)s, %(chunk_strategy)s, %(embedding)s
                ) ON CONFLICT (chunk_id) DO UPDATE SET
                    rule_number = EXCLUDED.rule_number,
                    rule_title = EXCLUDED.rule_title,
                    chapter = EXCLUDED.chapter,
                    chapter_title = EXCLUDED.chapter_title,
                    subrule_numbers = EXCLUDED.subrule_numbers,
                    status = EXCLUDED.status,
                    content = EXCLUDED.content,
                    token_count = EXCLUDED.token_count,
                    chunk_strategy = EXCLUDED.chunk_strategy,
                    embedding = EXCLUDED.embedding
                """,
                record,
            )
            if record["chunk_id"] in existing:
                updated += 1
            else:
                inserted += 1
    return {"inserted": inserted, "updated": updated, "failed": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embeddings_path", nargs="?", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--database-url", default=DATABASE_URL)
    args = parser.parse_args()

    records, failures = load_rule_records(args.embeddings_path)
    if failures:
        for failure in failures:
            print(f"FAILED: {failure}")
        print("Rows read: 0")
        print("Inserted: 0")
        print("Updated: 0")
        print(f"Failed: {len(failures)}")
        print(f"Total: {len(failures)}")
        raise SystemExit(1)

    try:
        import psycopg
        from pgvector.psycopg import register_vector

        with psycopg.connect(args.database_url) as conn:
            create_rule_chunks_table(conn)
            register_vector(conn)
            report = save_rule_records(conn, records)
    except Exception as error:
        failed_count = len(records) or 1
        print(f"Database load failed; transaction rolled back: {error}")
        print(f"Rows read: {len(records)}")
        print("Inserted: 0")
        print("Updated: 0")
        print(f"Failed: {failed_count}")
        print(f"Total: {failed_count}")
        raise SystemExit(1)

    total = report["inserted"] + report["updated"] + report["failed"]
    print(f"Rows read: {len(records)}")
    print(f"Inserted: {report['inserted']}")
    print(f"Updated: {report['updated']}")
    print(f"Failed: {report['failed']}")
    print(f"Total: {total}")


if __name__ == "__main__":
    main()
