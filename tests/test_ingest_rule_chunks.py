"""Run with: venv/bin/python -m unittest tests.test_ingest_rule_chunks -v."""

import json
import tempfile
import unittest
from pathlib import Path

from ingest_act_chunks import EXPECTED_EMBEDDING_DIMENSION
from ingest_rule_chunks import load_rule_records, save_rule_records, validate_rule_record


def embedding(dim=EXPECTED_EMBEDDING_DIMENSION):
    return [0.0] * dim


def raw_record(chunk_id="cgst-rules-1-0001", dim=EXPECTED_EMBEDDING_DIMENSION):
    return {
        "chunk_id": chunk_id,
        "text": "Rule 1. Example\n\ncontent",
        "token_count": 5,
        "metadata": {
            "rule_number": "1",
            "rule_title": "Example",
            "chapter": "CHAPTER I",
            "chapter_title": "PRELIMINARY",
            "subrule_numbers": ["1"],
            "status": "current",
            "chunk_strategy": "rule",
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


class IngestRuleChunksTests(unittest.TestCase):
    def test_validate_rule_record_preserves_required_metadata(self):
        record = validate_rule_record(raw_record(), "test:1")
        self.assertEqual(record["chunk_id"], "cgst-rules-1-0001")
        self.assertEqual(record["rule_number"], "1")
        self.assertEqual(record["rule_title"], "Example")
        self.assertEqual(record["chapter"], "CHAPTER I")
        self.assertEqual(record["chapter_title"], "PRELIMINARY")
        self.assertEqual(record["subrule_numbers"], ["1"])
        self.assertEqual(record["chunk_strategy"], "rule")
        self.assertEqual(len(record["embedding"]), EXPECTED_EMBEDDING_DIMENSION)

    def test_validate_rule_record_requires_1024_dimension(self):
        with self.assertRaises(ValueError):
            validate_rule_record(raw_record(dim=3), "test:1")

    def test_load_rule_records_reads_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rules.jsonl"
            path.write_text(json.dumps(raw_record()) + "\n", encoding="utf-8")
            records, failures = load_rule_records(path)
        self.assertEqual(failures, [])
        self.assertEqual(len(records), 1)

    def test_save_rule_records_counts_inserted_and_updated(self):
        conn = FakeConn(existing={"existing"})
        records = [validate_rule_record(raw_record("existing"), "test:1"), validate_rule_record(raw_record("new"), "test:2")]
        report = save_rule_records(conn, records)
        self.assertEqual(report, {"inserted": 1, "updated": 1, "failed": 0})
        self.assertTrue(any("CREATE EXTENSION" in statement[0] for statement in conn.statements))
        self.assertTrue(any("USING hnsw" in statement[0] for statement in conn.statements))


if __name__ == "__main__":
    unittest.main()
