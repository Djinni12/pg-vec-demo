"""Load GST Forms chunk embeddings into PostgreSQL/pgvector."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from psycopg.types.json import Jsonb

from ingest_act_chunks import EXPECTED_EMBEDDING_DIMENSION, load_embedding_records
from ingest_gst import DATABASE_URL

load_dotenv()

DEFAULT_EMBEDDINGS_PATH = Path("data/form/gst_forms_bge_m3_embeddings.jsonl")


def validate_form_record(record, source_label):
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

    rule_references = metadata.get("rule_references") or []
    if not isinstance(rule_references, list):
        raise ValueError(f"{source_label}: rule_references must be a list for {chunk_id}")

    normalized = {
        "chunk_id": chunk_id,
        "form_uid": str(metadata.get("form_uid") or "").strip(),
        "form_number": str(metadata.get("form_number") or "").strip(),
        "form_family": str(metadata.get("form_family") or "").strip(),
        "form_code": str(metadata.get("form_code") or "").strip(),
        "form_title": str(metadata.get("form_title") or metadata.get("title") or "").strip(),
        "title": str(metadata.get("title") or metadata.get("form_title") or "").strip(),
        "language": str(metadata.get("language") or "").strip(),
        "rule_references": [str(value) for value in rule_references],
        "part_number": metadata.get("part_number"),
        "section_label": metadata.get("section_label"),
        "source_start_page": metadata.get("source_start_page", metadata.get("page_start")),
        "source_end_page": metadata.get("source_end_page", metadata.get("page_end")),
        "page_start": metadata.get("page_start"),
        "page_end": metadata.get("page_end"),
        "content": content,
        "token_count": token_count,
        "chunk_strategy": str(metadata.get("chunk_strategy") or "").strip(),
        "metadata": Jsonb(metadata),
        "embedding": embedding,
    }
    for field in ("form_uid", "form_number", "form_family", "form_code", "form_title", "title", "language", "chunk_strategy"):
        if not normalized[field]:
            raise ValueError(f"{source_label}: missing {field} for {chunk_id}")
    return normalized


def load_form_records(path=DEFAULT_EMBEDDINGS_PATH):
    records = []
    failures = []
    try:
        for index, raw in enumerate(load_embedding_records(path), 1):
            records.append(validate_form_record(raw, f"{path}:{index}"))
    except Exception as error:
        failures.append(f"{path}: {error}")
    return records, failures


def create_form_chunks_table(conn):
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS form_chunks (
            chunk_id TEXT PRIMARY KEY,
            form_uid TEXT NOT NULL,
            form_number TEXT NOT NULL,
            form_family TEXT NOT NULL,
            form_code TEXT NOT NULL,
            form_title TEXT NOT NULL,
            title TEXT NOT NULL,
            language TEXT NOT NULL,
            rule_references TEXT[] NOT NULL DEFAULT '{}',
            part_number INTEGER,
            section_label TEXT,
            source_start_page INTEGER,
            source_end_page INTEGER,
            page_start INTEGER,
            page_end INTEGER,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL CHECK (token_count > 0),
            chunk_strategy TEXT NOT NULL,
            metadata JSONB NOT NULL,
            embedding VECTOR(1024) NOT NULL
        )
        """
    )
    for statement in (
        "ALTER TABLE form_chunks ADD COLUMN IF NOT EXISTS form_uid TEXT",
        "ALTER TABLE form_chunks ADD COLUMN IF NOT EXISTS title TEXT",
        "ALTER TABLE form_chunks ADD COLUMN IF NOT EXISTS language TEXT",
        "ALTER TABLE form_chunks ADD COLUMN IF NOT EXISTS rule_references TEXT[] NOT NULL DEFAULT '{}'",
        "ALTER TABLE form_chunks ADD COLUMN IF NOT EXISTS part_number INTEGER",
        "ALTER TABLE form_chunks ADD COLUMN IF NOT EXISTS section_label TEXT",
        "ALTER TABLE form_chunks ADD COLUMN IF NOT EXISTS source_start_page INTEGER",
        "ALTER TABLE form_chunks ADD COLUMN IF NOT EXISTS source_end_page INTEGER",
    ):
        conn.execute(statement)
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS form_chunks_embedding_hnsw_idx
            ON form_chunks USING hnsw (embedding vector_cosine_ops)
        """
    )


def existing_chunk_ids(conn, chunk_ids):
    if not chunk_ids:
        return set()
    rows = conn.execute("SELECT chunk_id FROM form_chunks WHERE chunk_id = ANY(%s)", (chunk_ids,)).fetchall()
    return {row[0] for row in rows}


def save_form_records(conn, records):
    create_form_chunks_table(conn)
    existing = existing_chunk_ids(conn, [record["chunk_id"] for record in records])
    inserted = 0
    updated = 0
    with conn.cursor() as cur:
        cur.execute("DELETE FROM form_chunks WHERE NOT (chunk_id = ANY(%s))", ([record["chunk_id"] for record in records],))
        deleted_stale = cur.rowcount
        for record in records:
            cur.execute(
                """
                INSERT INTO form_chunks (
                    chunk_id, form_uid, form_number, form_family, form_code, form_title, title,
                    language, rule_references, part_number, section_label, source_start_page, source_end_page,
                    page_start, page_end, content, token_count, chunk_strategy, metadata, embedding
                ) VALUES (
                    %(chunk_id)s, %(form_uid)s, %(form_number)s, %(form_family)s, %(form_code)s,
                    %(form_title)s, %(title)s, %(language)s, %(rule_references)s, %(part_number)s,
                    %(section_label)s, %(source_start_page)s, %(source_end_page)s, %(page_start)s,
                    %(page_end)s, %(content)s, %(token_count)s, %(chunk_strategy)s, %(metadata)s, %(embedding)s
                ) ON CONFLICT (chunk_id) DO UPDATE SET
                    form_uid = EXCLUDED.form_uid,
                    form_number = EXCLUDED.form_number,
                    form_family = EXCLUDED.form_family,
                    form_code = EXCLUDED.form_code,
                    form_title = EXCLUDED.form_title,
                    title = EXCLUDED.title,
                    language = EXCLUDED.language,
                    rule_references = EXCLUDED.rule_references,
                    part_number = EXCLUDED.part_number,
                    section_label = EXCLUDED.section_label,
                    source_start_page = EXCLUDED.source_start_page,
                    source_end_page = EXCLUDED.source_end_page,
                    page_start = EXCLUDED.page_start,
                    page_end = EXCLUDED.page_end,
                    content = EXCLUDED.content,
                    token_count = EXCLUDED.token_count,
                    chunk_strategy = EXCLUDED.chunk_strategy,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                record,
            )
            if record["chunk_id"] in existing:
                updated += 1
            else:
                inserted += 1
    return {"inserted": inserted, "updated": updated, "failed": 0, "deleted_stale": deleted_stale}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embeddings_path", nargs="?", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--database-url", default=DATABASE_URL)
    args = parser.parse_args()

    records, failures = load_form_records(args.embeddings_path)
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
            create_form_chunks_table(conn)
            register_vector(conn)
            report = save_form_records(conn, records)
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
    print(f"Deleted stale: {report.get('deleted_stale', 0)}")
    print(f"Total: {total}")


if __name__ == "__main__":
    main()
