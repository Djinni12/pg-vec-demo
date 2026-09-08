"""Ingest official GST rates from GST rates2025.pdf into normalized PostgreSQL tables.

Completely retires and removes old CSV rate tables (gst_goods_rates, gst_service_rates)
and archives CSV files, making GST rates2025.pdf the sole active rate source.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
from typing import Any

from dotenv import load_dotenv
import psycopg

from src.parsers.pdf_rate_parser import (
    DEFAULT_PDF_PATH,
    FALLBACK_PDF_PATH,
    parse_pdf_rates,
    resolve_pdf_path,
    validate_extracted_rates,
)

load_dotenv()

DEFAULT_DATABASE_URL = "dbname=hybrid_rag user=postgres password=postgres host=localhost port=5432"
DEFAULT_ARCHIVE_DIR = Path("data/archive")


def database_url() -> str:
    """Retrieve database connection URL from environment or fallback."""
    return os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL


def archive_old_csvs(
    source_dir: Path = Path("data/gst/csvs"),
    archive_dir: Path = DEFAULT_ARCHIVE_DIR,
) -> list[str]:
    """Safely archive old CSV files so they do not participate in the application."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_files: list[str] = []

    for filename in ["Goods.csv", "Services.csv"]:
        src_file = source_dir / filename
        if src_file.exists():
            dest_file = archive_dir / filename
            shutil.move(str(src_file), str(dest_file))
            archived_files.append(str(dest_file))

    # Ensure GST rates2025.pdf is also in data/gst/
    pdf_in_csvs = source_dir / "GST rates2025.pdf"
    clean_pdf_dest = Path("data/gst/GST rates2025.pdf")
    if pdf_in_csvs.exists() and not clean_pdf_dest.exists():
        clean_pdf_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(pdf_in_csvs), str(clean_pdf_dest))

    return archived_files


def run_database_migration(conn: psycopg.Connection) -> None:
    """Execute safe migration: drop old CSV tables and create normalized gst_rates_2025 table."""
    with conn.cursor() as cur:
        # 1. Drop old CSV rate tables
        cur.execute("DROP TABLE IF EXISTS gst_goods_rates CASCADE;")
        cur.execute("DROP TABLE IF EXISTS gst_service_rates CASCADE;")

        # 2. Create normalized 2025 rate table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS gst_rates_2025 (
                id SERIAL PRIMARY KEY,
                category VARCHAR(50) NOT NULL,
                notification_number VARCHAR(100),
                notification_date VARCHAR(50),
                effective_date VARCHAR(50),
                schedule VARCHAR(150),
                serial_number VARCHAR(50),
                hsn_code TEXT,
                normalized_hsn_codes TEXT[],
                description TEXT NOT NULL,
                rate VARCHAR(100) NOT NULL,
                cgst_rate_pct NUMERIC(6, 3),
                sgst_utgst_rate_pct NUMERIC(6, 3),
                igst_rate_pct NUMERIC(6, 3),
                formatted_rate TEXT NOT NULL,
                compensation_cess TEXT,
                condition_number VARCHAR(50),
                condition_text TEXT,
                source_page INTEGER NOT NULL,
                source_file VARCHAR(255) DEFAULT 'GST rates2025.pdf',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )

        # 3. Create indexes for fast lookup and similarity search
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS gst_rates_2025_hsn_gin_idx
                ON gst_rates_2025 USING GIN(normalized_hsn_codes);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS gst_rates_2025_desc_trgm_idx
                ON gst_rates_2025 USING GIN(description gin_trgm_ops);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS gst_rates_2025_category_idx
                ON gst_rates_2025(category);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS gst_rates_2025_page_idx
                ON gst_rates_2025(source_page);
            """
        )


def insert_rate_records(conn: psycopg.Connection, records: list[dict[str, Any]]) -> int:
    """Insert structured rate records into gst_rates_2025."""
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE gst_rates_2025 RESTART IDENTITY;")

        insert_sql = """
            INSERT INTO gst_rates_2025 (
                category,
                notification_number,
                notification_date,
                effective_date,
                schedule,
                serial_number,
                hsn_code,
                normalized_hsn_codes,
                description,
                rate,
                cgst_rate_pct,
                sgst_utgst_rate_pct,
                igst_rate_pct,
                formatted_rate,
                compensation_cess,
                condition_number,
                condition_text,
                source_page,
                source_file
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """
        rows_to_insert = [
            (
                r["category"],
                r["notification_number"],
                r["notification_date"],
                r["effective_date"],
                r["schedule"],
                r["serial_number"],
                r["hsn_code"],
                r["normalized_hsn_codes"],
                r["description"],
                r["rate"],
                r["cgst_rate_pct"],
                r["sgst_utgst_rate_pct"],
                r["igst_rate_pct"],
                r["formatted_rate"],
                r["compensation_cess"],
                r["condition_number"],
                r["condition_text"],
                r["source_page"],
                r["source_file"],
            )
            for r in records
        ]
        cur.executemany(insert_sql, rows_to_insert)

    conn.commit()
    return len(records)


def ingest_pdf_rates(
    pdf_path: str | Path | None = None,
    db_url: str | None = None,
    archive_dir: Path = DEFAULT_ARCHIVE_DIR,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Rebuildable pipeline: parse GST rates2025.pdf -> validate -> migrate PostgreSQL -> ingest."""
    # 1. Archive old CSV files
    archived = archive_old_csvs(archive_dir=archive_dir)

    # 2. Parse PDF
    resolved_pdf = resolve_pdf_path(pdf_path)
    records = parse_pdf_rates(resolved_pdf)

    # 3. Validate
    validation = validate_extracted_rates(records)
    if not validation["is_valid"]:
        raise ValueError(f"Extracted rates failed validation: {validation['errors']}")

    inserted_count = 0
    if not dry_run:
        url = db_url or database_url()
        with psycopg.connect(url) as conn:
            run_database_migration(conn)
            inserted_count = insert_rate_records(conn, records)

    return {
        "status": "success",
        "dry_run": dry_run,
        "source_file": str(resolved_pdf),
        "total_records_parsed": len(records),
        "total_records_inserted": inserted_count,
        "archived_csv_files": archived,
        "categories": validation["categories"],
        "schedules": validation["schedules"],
        "test_examples": validation["test_examples"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest GST rates2025.pdf into PostgreSQL")
    parser.add_argument("--pdf-path", type=Path, default=None, help="Path to GST rates2025.pdf")
    parser.add_argument("--database-url", type=str, default=None, help="PostgreSQL connection string")
    parser.add_argument("--dry-run", action="store_true", help="Parse and validate without database writes")
    args = parser.parse_args()

    print(f"Starting GST rates ingestion from GST rates2025.pdf...")
    res = ingest_pdf_rates(
        pdf_path=args.pdf_path,
        db_url=args.database_url,
        dry_run=args.dry_run,
    )
    print(f"Successfully ingested {res['total_records_inserted']} structured rate rows into gst_rates_2025.")
    print(f"Archived CSVs: {res['archived_csv_files']}")
    print(f"Categories: {res['categories']}")
    print(f"Test Examples:")
    for ex, info in res["test_examples"].items():
        print(f"  {ex}: found={info['found']}, count={info['count']}")


if __name__ == "__main__":
    main()
