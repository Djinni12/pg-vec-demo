"""Run with: venv/bin/python -m unittest test_rrf -v."""

import unittest
from unittest.mock import patch

from rrf import hybrid_rrf_search, reciprocal_rank_fusion


def row(source_file, source_row, code=None):
    code = code or f"{source_file}-{source_row}"
    return (
        code,
        f"description {code}",
        0.0,
        None,
        None,
        None,
        {"source_file": source_file, "source_row": source_row},
    )


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_document_present_in_both_lists_sums_contributions(self):
        shared = row("Goods.csv", 1, "0901")
        fused = reciprocal_rank_fusion([[shared], [shared]], k=60, top_k=5)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["row"], shared)
        self.assertAlmostEqual(fused[0]["rrf_score"], 2 / 61)

    def test_document_present_only_in_bm25_is_kept(self):
        bm25_only = row("Goods.csv", 2, "1001")
        fused = reciprocal_rank_fusion([[bm25_only], []], k=60, top_k=5)
        self.assertEqual([item["row"] for item in fused], [bm25_only])
        self.assertAlmostEqual(fused[0]["rrf_score"], 1 / 61)

    def test_document_present_only_in_vector_search_is_kept(self):
        vector_only = row("Services.csv", 3, "9965")
        fused = reciprocal_rank_fusion([[], [vector_only]], k=60, top_k=5)
        self.assertEqual([item["row"] for item in fused], [vector_only])
        self.assertAlmostEqual(fused[0]["rrf_score"], 1 / 61)

    def test_duplicate_records_are_merged_once_per_result_list(self):
        duplicate = row("Goods.csv", 4, "2202")
        fused = reciprocal_rank_fusion([[duplicate, duplicate], [duplicate]], k=60, top_k=5)
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0]["rrf_score"], 2 / 61)

    def test_final_ordering_uses_rrf_score_descending(self):
        shared_second = row("Goods.csv", 5, "3301")
        bm25_first = row("Goods.csv", 6, "3302")
        vector_first = row("Services.csv", 7, "9985")
        bm25_rows = [bm25_first, shared_second]
        vector_rows = [vector_first, shared_second]

        fused = reciprocal_rank_fusion([bm25_rows, vector_rows], k=60, top_k=3)

        self.assertEqual([item["row"] for item in fused], [shared_second, bm25_first, vector_first])
        self.assertAlmostEqual(fused[0]["rrf_score"], 2 / 62)
        self.assertAlmostEqual(fused[1]["rrf_score"], 1 / 61)
        self.assertAlmostEqual(fused[2]["rrf_score"], 1 / 61)

    def test_hybrid_search_reuses_bm25_and_vector_retrievers(self):
        conn = object()
        model = object()
        bm25_row = row("Goods.csv", 8, "4401")
        vector_row = row("Services.csv", 9, "9997")
        with patch("rrf.bm25_search", return_value=[bm25_row]) as bm25, patch(
            "rrf.vector_search", return_value=[vector_row]
        ) as vector:
            fused = hybrid_rrf_search(conn, "coffee", retrieve_limit=7, top_k=2, k=60, vector_model=model)

        bm25.assert_called_once_with(conn, "coffee", 7)
        vector.assert_called_once_with(conn, "coffee", 7, model=model)
        self.assertEqual([item["row"] for item in fused], [bm25_row, vector_row])


if __name__ == "__main__":
    unittest.main()
