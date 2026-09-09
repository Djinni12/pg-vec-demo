"""Load GST Central Tax (Rate) notification chunk embeddings into PostgreSQL/pgvector.

Creates the notification_chunks table, indexes (HNSW, B-tree, GIN),
and provides idempotent upsert with comprehensive schema and integrity validations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
import psycopg
from psycopg.types.json import Jsonb

load_dotenv()

EXPECTED_EMBEDDING_DIMENSION = 1024
DEFAULT_EMBEDDINGS_PATH = Path("data/notifications/notification_chunks_bge_m3_embeddings.jsonl")
DEFAULT_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "dbname=hybrid_rag user=postgres password=postgres host=localhost port=5432",
)


def validate_notification_record(record: Dict[str, Any], source_label: str) -> Dict[str, Any]:
    """Validate and normalize one notification chunk embedding record before DB ingestion."""
    if not isinstance(record, dict):
        raise ValueError(f"{source_label}: expected JSON object, got {type(record).__name__}")

    chunk_id = str(record.get("chunk_id") or "").strip()
    if not chunk_id:
        raise ValueError(f"{source_label}: missing chunk_id")
    if "18-2025" in chunk_id:
        raise ValueError(f"{source_label}: Excluded notification 18/2025 chunk found: {chunk_id}")

    content = str(record.get("text") or "").strip()
    if not content:
        raise ValueError(f"{source_label}: empty text content for {chunk_id}")

    token_count = record.get("token_count")
    if not isinstance(token_count, int) or token_count <= 0:
        raise ValueError(f"{source_label}: token_count must be positive integer for {chunk_id}")

    metadata = record.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"{source_label}: metadata must be dict for {chunk_id}")

    embedding = record.get("embedding")
    if not isinstance(embedding, list):
        raise ValueError(f"{source_label}: embedding must be list for {chunk_id}")
    if len(embedding) != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            f"{source_label}: {chunk_id} embedding dimension {len(embedding)} != {EXPECTED_EMBEDDING_DIMENSION}"
        )

    try:
        embedding = [float(v) for v in embedding]
    except (TypeError, ValueError) as err:
        raise ValueError(f"{source_label}: non-numeric embedding value for {chunk_id}") from err

    notif_num = str(metadata.get("notification_number") or "").strip()
    if not notif_num:
        raise ValueError(f"{source_label}: missing notification_number for {chunk_id}")
    if "18/2025" in notif_num:
        raise ValueError(f"{source_label}: Excluded notification 18/2025 found in metadata: {notif_num}")

    doc_type = str(metadata.get("document_type") or "").strip()
    chunk_type = str(metadata.get("chunk_type") or "").strip()
    chunk_strategy = str(metadata.get("chunk_strategy") or "").strip()
    if not doc_type or not chunk_type or not chunk_strategy:
        raise ValueError(f"{source_label}: missing document/chunk type or strategy for {chunk_id}")

    serial_numbers = metadata.get("serial_numbers") or []
    if not isinstance(serial_numbers, list):
        raise ValueError(f"{source_label}: serial_numbers must be list for {chunk_id}")
    serial_numbers = [str(s) for s in serial_numbers]

    normalized_hsn = metadata.get("normalized_hsn") or []
    if not isinstance(normalized_hsn, list):
        raise ValueError(f"{source_label}: normalized_hsn must be list for {chunk_id}")
    normalized_hsn = [str(h) for h in normalized_hsn]

    return {
        "chunk_id": chunk_id,
        "notification_number": notif_num,
        "notification_date": metadata.get("notification_date"),
        "document_type": doc_type,
        "chunk_type": chunk_type,
        "chunk_strategy": chunk_strategy,
        "effective_date": metadata.get("effective_date"),
        "target_notification": metadata.get("target_notification"),
        "schedule": metadata.get("schedule"),
        "schedule_rate_raw": metadata.get("schedule_rate_raw"),
        "tax_treatment": metadata.get("tax_treatment"),
        "serial_numbers": serial_numbers,
        "classification_raw": metadata.get("classification_raw"),
        "normalized_hsn": normalized_hsn,
        "operation_type": metadata.get("operation_type"),
        "source_page_start": metadata.get("source_page_start"),
        "source_page_end": metadata.get("source_page_end"),
        "content": content,
        "token_count": token_count,
        "metadata": Jsonb(metadata),
        "embedding": embedding,
    }


def load_notification_records(path: Path = DEFAULT_EMBEDDINGS_PATH) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load and validate all embedding records from JSONL."""
    records = []
    failures = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                rec = validate_notification_record(raw, f"{path.name}:{line_num}")
                records.append(rec)
            except Exception as err:
                failures.append(f"Line {line_num}: {err}")
    return records, failures


def create_notification_chunks_table(conn: psycopg.Connection) -> None:
    """Create dedicated notification_chunks table and required indexes."""
    from pgvector.psycopg import register_vector
    register_vector(conn)

    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notification_chunks (
            chunk_id TEXT PRIMARY KEY,
            notification_number TEXT NOT NULL,
            notification_date DATE,
            document_type TEXT NOT NULL,
            chunk_type TEXT NOT NULL,
            chunk_strategy TEXT NOT NULL,
            effective_date DATE,
            target_notification TEXT,
            schedule TEXT,
            schedule_rate_raw TEXT,
            tax_treatment TEXT,
            serial_numbers TEXT[],
            classification_raw TEXT,
            normalized_hsn TEXT[],
            operation_type TEXT,
            source_page_start INTEGER,
            source_page_end INTEGER,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            metadata JSONB NOT NULL,
            embedding VECTOR(1024) NOT NULL
        );
        """
    )

    # 1. HNSW cosine index on embedding with m=16, ef_construction=64
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS notification_chunks_embedding_hnsw_idx
            ON notification_chunks USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """
    )

    # 2. Index on notification_number
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_notification_chunks_notification_number
            ON notification_chunks (notification_number);
        """
    )

    # 3. Index on target_notification
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_notification_chunks_target_notification
            ON notification_chunks (target_notification);
        """
    )

    # 4. GIN index on normalized_hsn
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_notification_chunks_normalized_hsn
            ON notification_chunks USING gin (normalized_hsn);
        """
    )


def existing_notification_chunk_ids(conn: psycopg.Connection, chunk_ids: List[str]) -> Set[str]:
    """Return set of chunk_ids already present in notification_chunks."""
    if not chunk_ids:
        return set()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_id FROM notification_chunks WHERE chunk_id = ANY(%s)",
            (chunk_ids,),
        )
        return {row[0] for row in cur.fetchall()}


def save_notification_records(conn: psycopg.Connection, records: List[Dict[str, Any]]) -> Dict[str, int]:
    """Idempotently upsert notification chunks into database."""
    from pgvector.psycopg import register_vector
    register_vector(conn)

    create_notification_chunks_table(conn)

    chunk_ids = [r["chunk_id"] for r in records]
    existing = existing_notification_chunk_ids(conn, chunk_ids)

    inserted = 0
    updated = 0

    insert_sql = """
        INSERT INTO notification_chunks (
            chunk_id, notification_number, notification_date, document_type, chunk_type,
            chunk_strategy, effective_date, target_notification, schedule, schedule_rate_raw,
            tax_treatment, serial_numbers, classification_raw, normalized_hsn, operation_type,
            source_page_start, source_page_end, content, token_count, metadata, embedding
        ) VALUES (
            %(chunk_id)s, %(notification_number)s, %(notification_date)s, %(document_type)s, %(chunk_type)s,
            %(chunk_strategy)s, %(effective_date)s, %(target_notification)s, %(schedule)s, %(schedule_rate_raw)s,
            %(tax_treatment)s, %(serial_numbers)s, %(classification_raw)s, %(normalized_hsn)s, %(operation_type)s,
            %(source_page_start)s, %(source_page_end)s, %(content)s, %(token_count)s, %(metadata)s, %(embedding)s
        ) ON CONFLICT (chunk_id) DO UPDATE SET
            notification_number = EXCLUDED.notification_number,
            notification_date = EXCLUDED.notification_date,
            document_type = EXCLUDED.document_type,
            chunk_type = EXCLUDED.chunk_type,
            chunk_strategy = EXCLUDED.chunk_strategy,
            effective_date = EXCLUDED.effective_date,
            target_notification = EXCLUDED.target_notification,
            schedule = EXCLUDED.schedule,
            schedule_rate_raw = EXCLUDED.schedule_rate_raw,
            tax_treatment = EXCLUDED.tax_treatment,
            serial_numbers = EXCLUDED.serial_numbers,
            classification_raw = EXCLUDED.classification_raw,
            normalized_hsn = EXCLUDED.normalized_hsn,
            operation_type = EXCLUDED.operation_type,
            source_page_start = EXCLUDED.source_page_start,
            source_page_end = EXCLUDED.source_page_end,
            content = EXCLUDED.content,
            token_count = EXCLUDED.token_count,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding;
    """

    with conn.transaction():
        with conn.cursor() as cur:
            for rec in records:
                cur.execute(insert_sql, rec)
                if rec["chunk_id"] in existing:
                    updated += 1
                else:
                    inserted += 1

    return {"inserted": inserted, "updated": updated, "total": len(records)}


def run_database_validation(conn: psycopg.Connection) -> Dict[str, Any]:
    """Run all verification queries required by Stage 4 specification."""
    with conn.cursor() as cur:
        # 1. Total rows
        cur.execute("SELECT COUNT(*) FROM notification_chunks;")
        total_rows = cur.fetchone()[0]

        # 2. Duplicate chunk IDs
        cur.execute("""
            SELECT chunk_id, COUNT(*)
            FROM notification_chunks
            GROUP BY chunk_id
            HAVING COUNT(*) > 1;
        """)
        duplicates = cur.fetchall()

        # 3. Null embeddings
        cur.execute("SELECT COUNT(*) FROM notification_chunks WHERE embedding IS NULL;")
        null_embeddings = cur.fetchone()[0]

        # 4. Wrong embedding dimensions (must be 1024)
        cur.execute("SELECT COUNT(*) FROM notification_chunks WHERE vector_dims(embedding) != 1024;")
        wrong_dims = cur.fetchone()[0]

        # 5. Zero-token chunks
        cur.execute("SELECT COUNT(*) FROM notification_chunks WHERE token_count <= 0;")
        zero_tokens = cur.fetchone()[0]

        # 6. Excluded 18/2025 rows
        cur.execute("""
            SELECT COUNT(*) FROM notification_chunks
            WHERE notification_number LIKE '%18/2025%' OR chunk_id LIKE '%18-2025%';
        """)
        excluded_rows = cur.fetchone()[0]

        # 7. Index verification
        cur.execute("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'notification_chunks';
        """)
        indexes = {r[0]: r[1] for r in cur.fetchall()}

    return {
        "total_rows": total_rows,
        "duplicate_chunk_ids": len(duplicates),
        "null_embeddings": null_embeddings,
        "wrong_embedding_dimensions": wrong_dims,
        "zero_token_chunks": zero_tokens,
        "excluded_18_2025_rows": excluded_rows,
        "indexes": indexes,
    }


def fetch_representative_rows(conn: psycopg.Connection) -> Dict[str, Dict[str, Any]]:
    """Fetch representative rows from 09/2025, 10/2025, 13/2025, 15/2025, 17/2025, 01/2026."""
    target_prefixes = {
        "09/2025": "chunk-09-2025-central_tax_rate-0001",
        "10/2025": "chunk-10-2025-central_tax_rate-0001",
        "13/2025": "chunk-13-2025-central_tax_rate-0001",
        "15/2025": "chunk-15-2025-central_tax_rate-0001",
        "17/2025": "chunk-17-2025-central_tax_rate-0001",
        "01/2026": "chunk-01-2026-central_tax_rate-0001",
    }
    results = {}
    with conn.cursor() as cur:
        for label, cid in target_prefixes.items():
            cur.execute(
                """
                SELECT
                    chunk_id, notification_number, effective_date, target_notification,
                    schedule, schedule_rate_raw, tax_treatment, serial_numbers,
                    classification_raw, normalized_hsn, operation_type,
                    source_page_start, source_page_end, content, token_count,
                    vector_dims(embedding) as emb_dim, metadata
                FROM notification_chunks
                WHERE chunk_id = %s;
                """,
                (cid,),
            )
            row = cur.fetchone()
            if row:
                results[label] = {
                    "chunk_id": row[0],
                    "notification_number": row[1],
                    "effective_date": str(row[2]) if row[2] else None,
                    "target_notification": row[3],
                    "schedule": row[4],
                    "schedule_rate_raw": row[5],
                    "tax_treatment": row[6],
                    "serial_numbers": row[7],
                    "classification_raw": row[8],
                    "normalized_hsn": row[9],
                    "operation_type": row[10],
                    "source_page_start": row[11],
                    "source_page_end": row[12],
                    "content_snippet": row[13][:150] + "..." if len(row[13]) > 150 else row[13],
                    "token_count": row[14],
                    "embedding_dimension": row[15],
                    "has_raw_legal_text_in_metadata": "raw_legal_text" in (row[16] or {}),
                    "has_inherited_header_in_metadata": "inherited_context_header" in (row[16] or {}),
                }
    return results
