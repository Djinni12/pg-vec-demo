"""Load Kaggle Goods.csv and Services.csv without inventing missing tax data."""

import argparse
import csv
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import re

DATABASE_URL = "dbname=hybrid_rag user=postgres password=postgres host=localhost port=5432"
DATASET_URL = "https://www.kaggle.com/datasets/prasad22/goods-and-service-tax-rates-dataset"
MODEL_NAME = "all-MiniLM-L6-v2"
FIELDS = {
    "Goods.csv": (
        "goods", "Chapter / Heading / Sub-heading / Tariff item",
        "Description of Goods", "CGST Rate (%)", "SGST / UTGST Rate (%)",
        "IGST Rate (%)", "Compensation Cess",
    ),
    "Services.csv": (
        "service", "Chapter, Section or Heading", "Description of Service",
        "CGST Rate(%)", "SGST/UTGST Rate(%)", "IGST Rate(%)", "Condition",
    ),
}


def clean(value):
    return " ".join(value.split())


def rate_percent(value, record_type):
    """Goods numbers are fractions; services numbers already are percentages."""
    value = clean(value)
    if value.casefold() == "nil":
        return Decimal(0)
    if not re.fullmatch(r"\d+(?:\.\d+)?", value):
        return None
    return Decimal(value) * (100 if record_type == "goods" else 1)


def exact_codes(classification, record_type):
    """Only parse explicit code lists; never expand ranges or exclusions."""
    text = clean(classification)
    if re.search(r"\b(?:except|other than|to)\b", text, re.I):
        return []
    if record_type == "service":
        # A Section number is a grouping, not an HSN/SAC code.
        if not re.match(r"^(?:Chapter|Heading)\s", text, re.I):
            return []
        # Service labels follow the code in parentheses; no digits from labels.
        text = text.split("(", 1)[0].strip()
        text = re.sub(r"\b(?:Chapter|Heading)\s+", "", text, flags=re.I)
    parts = re.split(r"\s*(?:,|/|\bor\b|\band\b)\s*", text, flags=re.I)
    codes = []
    for part in parts:
        part = part.strip()
        # Accept compact codes or the common 4+2 / 4+2+2 tariff notation.
        if not re.fullmatch(r"(?:\d{2}|\d{4}|\d{6}|\d{8}|\d{4}\s\d{2}(?:\s\d{2})?)", part):
            return []
        codes.append(part.replace(" ", ""))
    return list(dict.fromkeys(codes))


def read_dataset(data_dir):
    """Return validated records and inspection counts; no database/model access."""
    records, report = [], {}
    for filename, fields in FIELDS.items():
        record_type, code_field, desc_field, cgst, sgst, igst, extra = fields
        raw_bytes = (Path(data_dir) / filename).read_bytes()
        try:
            text = raw_bytes.decode("utf-8-sig")
            encoding = "utf-8-sig"
        except UnicodeDecodeError:
            text = raw_bytes.decode("cp1252")
            encoding = "cp1252"
        reader = csv.DictReader(io.StringIO(text, newline=""))
        required = {"S. No.", *fields[1:]}
        if record_type == "goods":
            required.add("Schedules")
        if not required.issubset({clean(h) for h in reader.fieldnames or []}):
            raise ValueError(f"{filename}: unexpected CSV columns: {reader.fieldnames}")
        stats = {"rows": 0, "imported": 0, "omitted": 0, "blank_description": 0,
                 "column_number_row": 0, "encoding": encoding,
                 "sha256": hashlib.sha256(raw_bytes).hexdigest()}
        for row_number, raw in enumerate(reader, 1):
            stats["rows"] += 1
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"{filename} row {row_number}: malformed CSV row")
            row = {clean(key): clean(value) for key, value in raw.items()}
            description = row[desc_field]
            if record_type == "service" and row["S. No."] == "(1)" and description == "(3)":
                stats["column_number_row"] += 1
                continue
            if not description:
                stats["blank_description"] += 1
                continue
            if re.fullmatch(r"\[?omitted\]?\.?", description, re.I):
                stats["omitted"] += 1
                continue
            code = row[code_field]
            extra_text = row[extra]
            if extra_text in ("", "-", "…", "..."):
                extra_text = ""
            search_text = "\n".join(part for part in (description, code, extra_text) if part)
            metadata = {
                "dataset_url": DATASET_URL,
                "source_sha256": stats["sha256"],
                "source_file": filename,
                "source_row": row_number,
                "raw_row": raw,
                "rates_raw": {"cgst": row[cgst], "sgst_utgst": row[sgst], "igst": row[igst]},
            }
            records.append({
                "source_file": filename, "source_row": row_number,
                "record_type": record_type, "code": code,
                "exact_codes": exact_codes(code, record_type),
                "description": description, "search_text": search_text,
                "cgst_rate_pct": rate_percent(row[cgst], record_type),
                "sgst_utgst_rate_pct": rate_percent(row[sgst], record_type),
                "igst_rate_pct": rate_percent(row[igst], record_type),
                "metadata": metadata,
            })
            stats["imported"] += 1
        if not stats["imported"]:
            raise ValueError(f"{filename}: no usable descriptions; refusing empty import")
        report[filename] = stats
    return records, report


def save_records(conn, records, embeddings):
    """Upsert one dataset snapshot; the caller owns commit/rollback."""
    from psycopg.types.json import Jsonb

    if len(records) != len(embeddings):
        raise ValueError("Each record must have one embedding")
    with conn.cursor() as cur:
        for record, embedding in zip(records, embeddings):
            params = {**record, "metadata": Jsonb(record["metadata"]), "embedding": embedding}
            cur.execute(
                """
                INSERT INTO gst_documents (
                    source_file, source_row, record_type, code, exact_codes,
                    description, search_text, cgst_rate_pct, sgst_utgst_rate_pct,
                    igst_rate_pct, metadata, embedding
                ) VALUES (
                    %(source_file)s, %(source_row)s, %(record_type)s, %(code)s,
                    %(exact_codes)s, %(description)s, %(search_text)s, %(cgst_rate_pct)s,
                    %(sgst_utgst_rate_pct)s, %(igst_rate_pct)s, %(metadata)s, %(embedding)s
                ) ON CONFLICT (source_file, source_row) DO UPDATE SET
                    record_type = EXCLUDED.record_type, code = EXCLUDED.code,
                    exact_codes = EXCLUDED.exact_codes, description = EXCLUDED.description,
                    search_text = EXCLUDED.search_text, cgst_rate_pct = EXCLUDED.cgst_rate_pct,
                    sgst_utgst_rate_pct = EXCLUDED.sgst_utgst_rate_pct,
                    igst_rate_pct = EXCLUDED.igst_rate_pct, metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """, params,
            )
        # Remove stale rows from these two imported files, including newly omitted rows.
        for filename in {r["source_file"] for r in records}:
            row_numbers = [r["source_row"] for r in records if r["source_file"] == filename]
            cur.execute(
                "DELETE FROM gst_documents WHERE source_file = %s AND NOT (source_row = ANY(%s))",
                (filename, row_numbers),
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Directory containing Goods.csv and Services.csv")
    parser.add_argument("--dry-run", action="store_true", help="Validate CSVs without models or database writes")
    parser.add_argument("--database-url", default=DATABASE_URL)
    args = parser.parse_args()
    records, report = read_dataset(args.data_dir)
    print(json.dumps(report, indent=2))
    if args.dry_run:
        return

    import psycopg
    from pgvector.psycopg import register_vector
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode([r["description"] for r in records], batch_size=32, show_progress_bar=True)
    with psycopg.connect(args.database_url) as conn:
        register_vector(conn)
        save_records(conn, records, embeddings)
    print(f"Imported {len(records)} GST records.")


if __name__ == "__main__":
    main()
