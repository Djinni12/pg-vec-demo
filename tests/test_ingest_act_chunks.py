"""Run with: venv/bin/python -m unittest tests.test_ingest_act_chunks -v."""

import json
import tempfile
import unittest
from pathlib import Path

from ingest_act_chunks import (
    EXPECTED_EMBEDDING_DIMENSION,
    infer_act_name,
    infer_act_slug,
    load_act_records,
    parse_file_args,
    save_act_records,
    validate_record,
)


def embedding(dim=EXPECTED_EMBEDDING_DIMENSION):
    return [0.0] * dim


def raw_record(chunk_id="cgst-1-0001", dim=EXPECTED_EMBEDDING_DIMENSION):
    return {
        "chunk_id": chunk_id,
        "text": "Section 1. Example\n\ncontent",
        "token_count": 5,
        "metadata": {
            "chapter": "CHAPTER I PRELIMINARY",
            "section_number": "1",
            "section_title": "Example",
            "subsection_numbers": ["1", "2"],
            "status": "current",
        },
        "embedding": embedding(dim),
    }


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.conn.statements.append((sql, params))


class FakeConn:
    def __init__(self, existing=None):
        self.existing = existing or set()
        self.statements = []

    def execute(self, sql, params=None):
        self.statements.append((sql, params))
        if sql.strip().startswith("SELECT chunk_id"):
            requested = params[0]
            return FakeResult([(chunk_id,) for chunk_id in requested if chunk_id in self.existing])
        return FakeResult([])

    def cursor(self):
        return FakeCursor(self)


class IngestActChunksTests(unittest.TestCase):
    def test_infers_known_act_name_from_file(self):
        path = "data/acts/integrated_gst_2017_bge_m3_embeddings.jsonl"
        self.assertEqual(infer_act_slug(path), "integrated_gst_2017")
        self.assertEqual(
            infer_act_name(path),
            "Integrated Goods and Services Tax Act, 2017",
        )

    def test_parse_file_args_rejects_one_act_name_for_many_files(self):
        with self.assertRaises(ValueError):
            parse_file_args([Path("a.jsonl"), Path("b.jsonl")], act_name="One Act")

    def test_validate_record_requires_1024_dimension(self):
        with self.assertRaises(ValueError):
            validate_record(raw_record(dim=3), "test:1")

    def test_load_act_records_reads_jsonl_and_attaches_act_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "central_gst_act_2017_bge_m3_embeddings.jsonl"
            path.write_text(json.dumps(raw_record()) + "\n", encoding="utf-8")

            records, failures = load_act_records([(path, infer_act_name(path), infer_act_slug(path))])

        self.assertEqual(failures, [])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["act_name"], "Central Goods and Services Tax Act, 2017")
        self.assertEqual(records[0]["chunk_id"], "central_gst_act_2017:cgst-1-0001")
        self.assertEqual(records[0]["section_number"], "1")
        self.assertEqual(records[0]["subsection_numbers"], ["1", "2"])
        self.assertEqual(len(records[0]["embedding"]), EXPECTED_EMBEDDING_DIMENSION)

    def test_save_act_records_counts_inserted_and_updated(self):
        conn = FakeConn(existing={"existing"})
        records = [
            {**validate_record(raw_record("existing"), "test:1"), "act_name": "Act"},
            {**validate_record(raw_record("new"), "test:2"), "act_name": "Act"},
        ]

        report = save_act_records(conn, records)

        self.assertEqual(report, {"inserted": 1, "updated": 1, "failed": 0})
        self.assertTrue(any("CREATE EXTENSION" in statement[0] for statement in conn.statements))
        self.assertTrue(any("USING hnsw" in statement[0] for statement in conn.statements))


if __name__ == "__main__":
    unittest.main()
