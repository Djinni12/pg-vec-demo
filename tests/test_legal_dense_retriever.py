"""Run with: venv/bin/python -m unittest tests.test_legal_dense_retriever -v."""

import unittest
from unittest.mock import Mock

from legal_dense_retriever import rerank_legal_results


class LegalDenseRetrieverTests(unittest.TestCase):
    def test_rerank_legal_results_orders_by_cross_encoder_score(self):
        reranker = Mock()
        reranker.predict.return_value = [0.2, 0.9]
        first = {
            "document_type": "act",
            "chunk_id": "a1",
            "reference": "Section 1",
            "title": "First",
            "content": "less relevant",
            "source_metadata": {},
            "score": 0.8,
            "distance": 0.2,
        }
        second = {
            "document_type": "rule",
            "chunk_id": "r1",
            "reference": "Rule 1",
            "title": "Second",
            "content": "more relevant",
            "source_metadata": {},
            "score": 0.7,
            "distance": 0.3,
        }

        results = rerank_legal_results("query", [first, second], reranker=reranker)

        reranker.predict.assert_called_once_with([("query", "less relevant"), ("query", "more relevant")])
        self.assertEqual([result["chunk_id"] for result in results], ["r1", "a1"])
        self.assertEqual(results[0]["dense_score"], 0.7)
        self.assertEqual(results[0]["rerank_score"], 0.9)
        self.assertEqual(results[0]["score"], 0.9)

    def test_rerank_legal_results_handles_empty_candidates(self):
        reranker = Mock()
        self.assertEqual(rerank_legal_results("query", [], reranker=reranker), [])
        reranker.predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
