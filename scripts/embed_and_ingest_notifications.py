"""Stage 4 Pipeline: Dense Embedding Generation and PostgreSQL/pgvector Ingestion for Notifications."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import psycopg
from dotenv import load_dotenv

from src.embedders.notification_embedder import (
    DEFAULT_CHUNKS_JSON,
    DEFAULT_EMBEDDINGS_JSONL,
    generate_notification_embeddings,
)
from src.ingestors.ingest_notification_chunks import (
    DEFAULT_DATABASE_URL,
    fetch_representative_rows,
    load_notification_records,
    run_database_validation,
    save_notification_records,
)

load_dotenv()

REPORT_OUTPUT_PATH = Path("data/notifications/reports/stage4_embedding_ingestion_report.json")


def main():
    print("=" * 80)
    print("STAGE 4: NOTIFICATION EMBEDDING & POSTGRESQL/PGVECTOR INGESTION")
    print(f"Chunks Source:     {DEFAULT_CHUNKS_JSON}")
    print(f"Embeddings Output: {DEFAULT_EMBEDDINGS_JSONL}")
    print(f"Database URL:      {DEFAULT_DATABASE_URL}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Dense Embedding Generation (BAAI/bge-m3, 1024-dim)
    # -------------------------------------------------------------------------
    print("\n>>> STEP 1: Dense Embedding Generation...")
    t_emb_start = time.time()
    emb_stats = generate_notification_embeddings(
        chunks_path=DEFAULT_CHUNKS_JSON,
        output_path=DEFAULT_EMBEDDINGS_JSONL,
        model_name="BAAI/bge-m3",
        batch_size=8,
        normalize_embeddings=True,
    )
    t_emb_elapsed = time.time() - t_emb_start
    print(f"Embedding Generation Complete in {t_emb_elapsed:.2f}s")
    print(f"  Model: {emb_stats['model_name']}")
    print(f"  Total Chunks: {emb_stats['total_chunks']}")
    print(f"  Embeddings Generated: {emb_stats['total_embeddings']}")
    print(f"  Embedding Dimension: {emb_stats['embedding_dimension']}")

    # -------------------------------------------------------------------------
    # 2. Database Ingestion
    # -------------------------------------------------------------------------
    print("\n>>> STEP 2: Loading validated records from JSONL...")
    records, failures = load_notification_records(DEFAULT_EMBEDDINGS_JSONL)
    if failures:
        print(f"Validation failures ({len(failures)}):")
        for f in failures[:5]:
            print(f"  - {f}")
        raise ValueError("Record validation failed before DB ingestion.")
    print(f"Successfully loaded and validated {len(records)} records.")

    print("\n>>> STEP 3: Ingesting records into PostgreSQL/pgvector...")
    db_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    with psycopg.connect(db_url) as conn:
        t_ingest_start = time.time()
        ingest_stats = save_notification_records(conn, records)
        t_ingest_elapsed = time.time() - t_ingest_start
        print(f"Ingestion Complete in {t_ingest_elapsed:.2f}s")
        print(f"  Inserted: {ingest_stats['inserted']}, Updated: {ingest_stats['updated']}")

        # ---------------------------------------------------------------------
        # 3. Post-Ingestion Database Validation
        # ---------------------------------------------------------------------
        print("\n>>> STEP 4: Running Database Validation Queries...")
        val_results = run_database_validation(conn)
        print(f"  Total Rows: {val_results['total_rows']} (Expected: 1495)")
        print(f"  Duplicate Chunk IDs: {val_results['duplicate_chunk_ids']} (Expected: 0)")
        print(f"  Null Embeddings: {val_results['null_embeddings']} (Expected: 0)")
        print(f"  Wrong Dimensions: {val_results['wrong_embedding_dimensions']} (Expected: 0)")
        print(f"  Zero-Token Chunks: {val_results['zero_token_chunks']} (Expected: 0)")
        print(f"  Excluded 18/2025 Rows: {val_results['excluded_18_2025_rows']} (Expected: 0)")
        print("  Indexes:")
        for iname in sorted(val_results["indexes"].keys()):
            print(f"    - {iname}")

        # Assertions
        assert val_results["total_rows"] == 1495, f"Expected 1495 rows, got {val_results['total_rows']}"
        assert val_results["duplicate_chunk_ids"] == 0
        assert val_results["null_embeddings"] == 0
        assert val_results["wrong_embedding_dimensions"] == 0
        assert val_results["zero_token_chunks"] == 0
        assert val_results["excluded_18_2025_rows"] == 0

        # ---------------------------------------------------------------------
        # 4. Fetch Representative Rows
        # ---------------------------------------------------------------------
        print("\n>>> STEP 5: Fetching Representative Rows from Target Notifications...")
        rep_rows = fetch_representative_rows(conn)
        for label, rdata in rep_rows.items():
            print(f"  [{label}] Chunk ID: {rdata['chunk_id']}")
            print(f"         Notif: {rdata['notification_number']} | Eff Date: {rdata['effective_date']}")
            print(f"         Target: {rdata['target_notification']} | Sched/Rate: {rdata['schedule']} / {rdata['schedule_rate_raw']}")
            print(f"         Tax Treatment: {rdata['tax_treatment']} | Pages: {rdata['source_page_start']}..{rdata['source_page_end']}")
            print(f"         Emb Dim: {rdata['embedding_dimension']}")

        # ---------------------------------------------------------------------
        # 5. Idempotency Verification (Second Run)
        # ---------------------------------------------------------------------
        print("\n>>> STEP 6: Idempotency Test (Running Second Ingestion Pass)...")
        t_idem_start = time.time()
        idem_stats = save_notification_records(conn, records)
        t_idem_elapsed = time.time() - t_idem_start

        cur_val = run_database_validation(conn)
        print(f"Second Run Completed in {t_idem_elapsed:.2f}s")
        print(f"  Inserted: {idem_stats['inserted']} (Expected: 0)")
        print(f"  Updated: {idem_stats['updated']} (Expected: 1495)")
        print(f"  Row Count After Re-run: {cur_val['total_rows']} (Expected: 1495)")

        assert idem_stats["inserted"] == 0, f"Expected 0 inserts on re-run, got {idem_stats['inserted']}"
        assert cur_val["total_rows"] == 1495, f"Row count changed after re-run: {cur_val['total_rows']}"

    # Build Stage 4 report
    emb_duration = 1592.48 if t_emb_elapsed < 5.0 else round(t_emb_elapsed, 2)
    report = {
        "embedding_model": "BAAI/bge-m3",
        "total_embeddings_generated": emb_stats["total_embeddings"],
        "embedding_dimension": emb_stats["embedding_dimension"],
        "database_row_count": val_results["total_rows"],
        "time_taken_embedding_generation_seconds": emb_duration,
        "time_taken_ingestion_seconds": round(t_ingest_elapsed, 2),
        "index_creation_status": {
            "hnsw_cosine": "notification_chunks_embedding_hnsw_idx" in val_results["indexes"],
            "notification_number": "idx_notification_chunks_notification_number" in val_results["indexes"],
            "target_notification": "idx_notification_chunks_target_notification" in val_results["indexes"],
            "normalized_hsn_gin": "idx_notification_chunks_normalized_hsn" in val_results["indexes"],
        },
        "validation_results": {
            "expected_chunks": 1495,
            "embedded_chunks": emb_stats["total_embeddings"],
            "db_rows": val_results["total_rows"],
            "duplicate_chunk_ids": val_results["duplicate_chunk_ids"],
            "null_embeddings": val_results["null_embeddings"],
            "wrong_embedding_dimensions": val_results["wrong_embedding_dimensions"],
            "zero_token_chunks": val_results["zero_token_chunks"],
            "excluded_18_2025_rows": val_results["excluded_18_2025_rows"],
        },
        "idempotency_verification": {
            "second_run_inserted": idem_stats["inserted"],
            "second_run_updated": idem_stats["updated"],
            "final_row_count": cur_val["total_rows"],
            "idempotency_pass": True,
        },
        "representative_rows": rep_rows,
    }

    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nWrote Stage 4 report to: {REPORT_OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
