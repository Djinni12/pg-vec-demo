"""Run with: venv/bin/python -m unittest tests.test_act_embedder -v."""

import tempfile
import unittest
from pathlib import Path

from act_embedder import build_embedding_records, write_jsonl


class FakeModel:
    def encode(self, texts, batch_size, show_progress_bar, normalize_embeddings):
        self.calls = {
            "texts": texts,
            "batch_size": batch_size,
            "show_progress_bar": show_progress_bar,
            "normalize_embeddings": normalize_embeddings,
        }
        return [[1.0, 0.0, 0.0] for _ in texts]


class BadCountModel:
    def encode(self, texts, batch_size, show_progress_bar, normalize_embeddings):
        return [[1.0, 0.0, 0.0]]


def chunk(chunk_id="c1", text="hello world"):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "token_count": 2,
        "metadata": {"section_number": "1", "status": "current"},
    }


class ActEmbedderTests(unittest.TestCase):
    def test_builds_one_embedding_record_per_chunk_and_preserves_metadata(self):
        model = FakeModel()
        chunks = [chunk("c1", "alpha"), chunk("c2", "beta")]

        records, report = build_embedding_records(chunks, model=model, batch_size=4)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["metadata"], chunks[0]["metadata"])
        self.assertEqual(records[0]["embedding_model"], "BAAI/bge-m3")
        self.assertTrue(records[0]["embedding_normalized"])
        self.assertEqual(records[0]["embedding_dimension"], 3)
        self.assertEqual(report["chunk_count"], 2)
        self.assertEqual(report["embedding_count"], 2)
        self.assertEqual(report["embedding_dimension"], 3)
        self.assertEqual(model.calls["texts"], ["alpha", "beta"])
        self.assertTrue(model.calls["normalize_embeddings"])

    def test_empty_chunk_fails_before_embedding(self):
        with self.assertRaises(ValueError):
            build_embedding_records([chunk("empty", "   ")], model=FakeModel())

    def test_mismatched_embedding_count_fails(self):
        with self.assertRaises(ValueError):
            build_embedding_records([chunk("c1"), chunk("c2")], model=BadCountModel())

    def test_writes_jsonl_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "embeddings.jsonl"
            write_jsonl([{"chunk_id": "c1", "embedding": [1.0]}], path)
            self.assertEqual(path.read_text().count("\n"), 1)
            self.assertIn('"chunk_id": "c1"', path.read_text())


if __name__ == "__main__":
    unittest.main()
