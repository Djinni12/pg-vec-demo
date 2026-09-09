"""Tests for Stage 4 Notification Embedding & Ingestion Pipeline."""

from pathlib import Path
import os
import pytest
import psycopg
from dotenv import load_dotenv

from src.ingestors.ingest_notification_chunks import (
    DEFAULT_DATABASE_URL,
    DEFAULT_EMBEDDINGS_PATH,
    fetch_representative_rows,
    load_notification_records,
    run_database_validation,
    save_notification_records,
)

load_dotenv()


@pytest.fixture(scope="module")
def db_conn():
    url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    conn = psycopg.connect(url)
    yield conn
    conn.close()


def test_notification_embeddings_file():
    assert DEFAULT_EMBEDDINGS_PATH.exists(), f"Missing embeddings file: {DEFAULT_EMBEDDINGS_PATH}"
    records, failures = load_notification_records(DEFAULT_EMBEDDINGS_PATH)
    assert len(failures) == 0
    assert len(records) == 1495

    # Check 18/2025 is excluded
    for rec in records:
        assert "18-2025" not in rec["chunk_id"]
        assert "18/2025" not in rec["notification_number"]
        assert len(rec["embedding"]) == 1024
        assert rec["token_count"] > 0
        assert rec["content"]


def test_database_row_count_and_integrity(db_conn):
    val = run_database_validation(db_conn)
    assert val["total_rows"] == 1495
    assert val["duplicate_chunk_ids"] == 0
    assert val["null_embeddings"] == 0
    assert val["wrong_embedding_dimensions"] == 0
    assert val["zero_token_chunks"] == 0
    assert val["excluded_18_2025_rows"] == 0


def test_database_indexes_present(db_conn):
    val = run_database_validation(db_conn)
    indexes = val["indexes"]

    # HNSW cosine index
    assert "notification_chunks_embedding_hnsw_idx" in indexes
    assert "hnsw" in indexes["notification_chunks_embedding_hnsw_idx"].lower()
    assert "vector_cosine_ops" in indexes["notification_chunks_embedding_hnsw_idx"]

    # B-tree indexes
    assert "idx_notification_chunks_notification_number" in indexes
    assert "idx_notification_chunks_target_notification" in indexes

    # GIN index on normalized_hsn
    assert "idx_notification_chunks_normalized_hsn" in indexes
    assert "gin" in indexes["idx_notification_chunks_normalized_hsn"].lower()


def test_representative_rows_schema_and_metadata(db_conn):
    rep = fetch_representative_rows(db_conn)
    expected_keys = {"09/2025", "10/2025", "13/2025", "15/2025", "17/2025", "01/2026"}
    assert set(rep.keys()) == expected_keys

    # 09/2025
    r09 = rep["09/2025"]
    assert r09["notification_number"] == "09/2025-Central Tax (Rate)"
    assert r09["schedule"] == "Schedule I"
    assert r09["schedule_rate_raw"] == "2.5%"
    assert r09["tax_treatment"] == "TAXABLE"
    assert r09["embedding_dimension"] == 1024
    assert r09["has_raw_legal_text_in_metadata"]
    assert r09["has_inherited_header_in_metadata"]

    # 10/2025 (Exempt, Nil, no 0%)
    r10 = rep["10/2025"]
    assert r10["notification_number"] == "10/2025-Central Tax (Rate)"
    assert r10["tax_treatment"] == "EXEMPT"
    assert r10["schedule_rate_raw"] == "Nil"
    assert r10["embedding_dimension"] == 1024

    # 13/2025 (SUBSTITUTE, target 21/2018, row-level page 1)
    r13 = rep["13/2025"]
    assert r13["notification_number"] == "13/2025-Central Tax (Rate)"
    assert r13["target_notification"] == "21/2018-Central Tax (Rate)"
    assert r13["operation_type"] == "SUBSTITUTE"
    assert r13["tax_treatment"] == "CONCESSIONAL"
    assert r13["source_page_start"] == 1
    assert r13["source_page_end"] == 1
    assert r13["embedding_dimension"] == 1024

    # 15/2025
    r15 = rep["15/2025"]
    assert r15["notification_number"] == "15/2025-Central Tax (Rate)"
    assert r15["target_notification"] == "11/2017-Central Tax (Rate)"
    assert r15["embedding_dimension"] == 1024

    # 17/2025
    r17 = rep["17/2025"]
    assert r17["notification_number"] == "17/2025-Central Tax (Rate)"
    assert r17["target_notification"] == "17/2017-Central Tax (Rate)"
    assert r17["operation_type"] == "INSERT"
    assert r17["embedding_dimension"] == 1024

    # 01/2026 (CTR-E-updated-1)
    r01 = rep["01/2026"]
    assert r01["notification_number"] == "01/2026-Central Tax (Rate)"
    assert r01["target_notification"] == "09/2025-Central Tax (Rate)"
    assert r01["embedding_dimension"] == 1024


def test_idempotent_reingestion(db_conn):
    records, _ = load_notification_records(DEFAULT_EMBEDDINGS_PATH)
    stats = save_notification_records(db_conn, records)
    assert stats["inserted"] == 0
    assert stats["updated"] == 1495

    val = run_database_validation(db_conn)
    assert val["total_rows"] == 1495
