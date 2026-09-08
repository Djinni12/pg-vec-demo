"""Load GST Act chunk embeddings into PostgreSQL/pgvector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from ingest_gst import DATABASE_URL


load_dotenv()

EXPECTED_EMBEDDING_DIMENSION = 1024
DEFAULT_EMBEDDING_FILES = (
    Path("data/acts/central_gst_act_2017_bge_m3_embeddings.jsonl"),
    Path("data/acts/integrated_gst_2017_bge_m3_embeddings.jsonl"),
    Path("data/acts/gst_act_compensation_to_states_bge_m3_embeddings.jsonl"),
    Path("data/acts/union_territory_gsg_act_bge_m3_embeddings.jsonl"),
)
ACT_NAMES_BY_SLUG = {
    "central_gst_act_2017": "Central Goods and Services Tax Act, 2017",
    "integrated_gst_2017": "Integrated Goods and Services Tax Act, 2017",
    "gst_act_compensation_to_states": "Goods and Services Tax (Compensation to States) Act, 2017",
    "union_territory_gsg_act": "Union Territory Goods and Services Tax Act, 2017",
}


def infer_act_slug(path):
    """Infer the stable Act slug from the embedding filename."""
    stem = Path(path).name
    for slug in ACT_NAMES_BY_SLUG:
        if stem.startswith(slug):
            return slug
    return Path(path).stem.replace("_bge_m3_embeddings", "")


def infer_act_name(path):
    """Infer a readable Act name from the embedding filename."""
    slug = infer_act_slug(path)
    return ACT_NAMES_BY_SLUG.get(slug, slug.replace("_", " ").title())


def load_embedding_records(path):
    """Read embedding records from JSONL or JSON."""
    path = Path(path)
    if path.suffix.casefold() == ".jsonl":
        records = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    records.append(json.loads(line))
        return records

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected JSON array or JSONL records")
    return data


def validate_record(record, source_label, chunk_id_prefix=None):
    """Validate and normalize one embedding record for database insertion."""
    if not isinstance(record, dict):
        raise ValueError(f"{source_label}: expected object, got {type(record).__name__}")

    original_chunk_id = str(record.get("chunk_id") or "").strip()
    chunk_id = f"{chunk_id_prefix}:{original_chunk_id}" if chunk_id_prefix else original_chunk_id
    content = str(record.get("text") or "").strip()
    metadata = record.get("metadata") or {}
    embedding = record.get("embedding")

    if not original_chunk_id:
        raise ValueError(f"{source_label}: missing chunk_id")
    if not content:
        raise ValueError(f"{source_label}: empty content for {chunk_id}")
    if not isinstance(metadata, dict):
        raise ValueError(f"{source_label}: metadata must be an object for {chunk_id}")
    if not isinstance(embedding, list):
        raise ValueError(f"{source_label}: embedding must be a list for {chunk_id}")
    if len(embedding) != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            f"{source_label}: {chunk_id} embedding dimension {len(embedding)} != "
            f"{EXPECTED_EMBEDDING_DIMENSION}"
        )

    try:
        embedding = [float(value) for value in embedding]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source_label}: non-numeric embedding value for {chunk_id}") from error

    token_count = record.get("token_count")
    if not isinstance(token_count, int):
        raise ValueError(f"{source_label}: token_count must be an integer for {chunk_id}")

    subsection_numbers = metadata.get("subsection_numbers") or []
    if not isinstance(subsection_numbers, list):
        raise ValueError(f"{source_label}: subsection_numbers must be a list for {chunk_id}")

    return {
        "chunk_id": chunk_id,
        "chapter": metadata.get("chapter"),
        "section_number": metadata.get("section_number"),
        "section_title": metadata.get("section_title"),
        "subsection_numbers": [str(value) for value in subsection_numbers],
        "status": metadata.get("status"),
        "content": content,
        "token_count": token_count,
        "embedding": embedding,
    }


def load_act_records(files):
    """Read all requested embedding files and attach act names."""
    records = []
    failures = []
    for path, act_name, act_slug in files:
        try:
            raw_records = load_embedding_records(path)
            for index, raw in enumerate(raw_records, 1):
                record = validate_record(raw, f"{path}:{index}", chunk_id_prefix=act_slug)
                record["act_name"] = act_name
                records.append(record)
        except Exception as error:
            failures.append(f"{path}: {error}")
    return records, failures


def create_act_chunks_table(conn):
    """Create the act_chunks table and vector index if schema.sql was not run."""
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS act_chunks (
            chunk_id TEXT PRIMARY KEY,
            act_name TEXT NOT NULL,
            chapter TEXT,
            section_number TEXT NOT NULL,
            section_title TEXT NOT NULL,
            subsection_numbers TEXT[] NOT NULL DEFAULT '{}',
            status TEXT NOT NULL,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL CHECK (token_count > 0),
            embedding VECTOR(1024) NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS act_chunks_embedding_hnsw_idx
            ON act_chunks USING hnsw (embedding vector_cosine_ops)
        """
    )


def existing_chunk_ids(conn, chunk_ids):
    """Return chunk ids already present in act_chunks."""
    if not chunk_ids:
        return set()
    rows = conn.execute("SELECT chunk_id FROM act_chunks WHERE chunk_id = ANY(%s)", (chunk_ids,)).fetchall()
    return {row[0] for row in rows}


def save_act_records(conn, records):
    """Upsert act chunks; caller owns transaction commit/rollback."""
    create_act_chunks_table(conn)
    existing = existing_chunk_ids(conn, [record["chunk_id"] for record in records])
    inserted = 0
    updated = 0

    with conn.cursor() as cur:
        for record in records:
            params = {**record}
            cur.execute(
                """
                INSERT INTO act_chunks (
                    chunk_id, act_name, chapter, section_number, section_title,
                    subsection_numbers, status, content, token_count, embedding
                ) VALUES (
                    %(chunk_id)s, %(act_name)s, %(chapter)s, %(section_number)s,
                    %(section_title)s, %(subsection_numbers)s, %(status)s,
                    %(content)s, %(token_count)s, %(embedding)s
                ) ON CONFLICT (chunk_id) DO UPDATE SET
                    act_name = EXCLUDED.act_name,
                    chapter = EXCLUDED.chapter,
                    section_number = EXCLUDED.section_number,
                    section_title = EXCLUDED.section_title,
                    subsection_numbers = EXCLUDED.subsection_numbers,
                    status = EXCLUDED.status,
                    content = EXCLUDED.content,
                    token_count = EXCLUDED.token_count,
                    embedding = EXCLUDED.embedding
                """,
                params,
            )
            if record["chunk_id"] in existing:
                updated += 1
            else:
                inserted += 1
    return {"inserted": inserted, "updated": updated, "failed": 0}


def parse_file_args(paths, act_name=None):
    """Return (path, act_name) pairs for ingestion."""
    if not paths:
        paths = list(DEFAULT_EMBEDDING_FILES)
    if act_name and len(paths) != 1:
        raise ValueError("--act-name can only be used with one input file")
    return [(Path(path), act_name or infer_act_name(path), infer_act_slug(path)) for path in paths]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embedding_files", nargs="*", type=Path, help="JSONL/JSON files with chunk embeddings")
    parser.add_argument("--act-name", help="Override act name for a single input file")
    parser.add_argument("--database-url", default=DATABASE_URL)
    args = parser.parse_args()

    try:
        files = parse_file_args(args.embedding_files, args.act_name)
        records, failures = load_act_records(files)
    except Exception as error:
        print(f"Failed to read input: {error}")
        raise SystemExit(1)

    if failures:
        for failure in failures:
            print(f"FAILED: {failure}")
        print("Inserted: 0")
        print("Updated: 0")
        print(f"Failed: {len(failures)}")
        raise SystemExit(1)

    try:
        import psycopg
        from pgvector.psycopg import register_vector

        with psycopg.connect(args.database_url) as conn:
            create_act_chunks_table(conn)
            register_vector(conn)
            report = save_act_records(conn, records)
    except Exception as error:
        print(f"Database load failed; transaction rolled back: {error}")
        print("Inserted: 0")
        print("Updated: 0")
        print("Failed: 1")
        raise SystemExit(1)

    print(f"Files: {len(files)}")
    print(f"Rows read: {len(records)}")
    print(f"Inserted: {report['inserted']}")
    print(f"Updated: {report['updated']}")
    print(f"Failed: {report['failed']}")


if __name__ == "__main__":
    main()
