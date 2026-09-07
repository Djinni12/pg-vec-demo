"""Run with: venv/bin/python -m unittest test_keyword_search -v.

Set TEST_DATABASE_URL to also run PostgreSQL integration tests. They use a
temporary GST table and roll back, leaving persistent GST records intact.
"""

import os
import unittest
from unittest.mock import patch

from keyword_search import exact_lookup, keyword_search


class KeywordRoutingTests(unittest.TestCase):
    def test_bm25_hits_do_not_call_fallback_or_top_up(self):
        rows = [("0901", "Coffee beans", 0.1)]
        with patch("keyword_search.bm25_search", return_value=rows), patch(
            "keyword_search.fuzzy_search"
        ) as fuzzy:
            self.assertEqual(keyword_search(object(), "coffee"), (rows, "bm25"))
            fuzzy.assert_not_called()

    def test_no_bm25_hits_use_same_query_and_limit_for_fallback(self):
        conn = object()
        rows = [("0901", "Coffee beans", 0.4)]
        with patch("keyword_search.bm25_search", return_value=[]) as bm25, patch(
            "keyword_search.fuzzy_search", return_value=rows
        ) as fuzzy:
            self.assertEqual(keyword_search(conn, "cofee", 3), (rows, "trigram"))
            bm25.assert_called_once_with(conn, "cofee", 3)
            fuzzy.assert_called_once_with(conn, "cofee", 3)

    def test_database_error_does_not_trigger_fallback(self):
        with patch("keyword_search.bm25_search", side_effect=RuntimeError("database error")), patch(
            "keyword_search.fuzzy_search"
        ) as fuzzy:
            with self.assertRaises(RuntimeError):
                keyword_search(object(), "coffee")
            fuzzy.assert_not_called()

    def test_blank_query_does_not_access_database(self):
        self.assertEqual(keyword_search(None, "  "), ([], "bm25"))

    def test_nonpositive_limit_is_rejected(self):
        for limit in (0, -1):
            with self.assertRaises(ValueError):
                keyword_search(None, "coffee", limit)


@unittest.skipUnless(os.environ.get("TEST_DATABASE_URL"), "TEST_DATABASE_URL not set")
class PostgreSQLKeywordTests(unittest.TestCase):
    def setUp(self):
        import psycopg

        self.conn = psycopg.connect(os.environ["TEST_DATABASE_URL"])
        self.addCleanup(self.conn.close)
        self.addCleanup(self.conn.rollback)
        self.conn.execute("""
            CREATE TEMP TABLE gst_documents (
                source_file TEXT, source_row INTEGER,
                code TEXT, exact_codes TEXT[], description TEXT, search_text TEXT,
                cgst_rate_pct NUMERIC, sgst_utgst_rate_pct NUMERIC,
                igst_rate_pct NUMERIC, metadata JSONB DEFAULT '{}'
            )
        """)
        self.conn.execute("SET LOCAL pg_trgm.similarity_threshold = 0.3")
        self.conn.execute(
            "CREATE INDEX gst_documents_search_bm25_idx ON gst_documents "
            "USING bm25(search_text) WITH (text_config='english')"
        )
        with self.conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO gst_documents (source_file, source_row, code, exact_codes, "
                "description, search_text) VALUES (%s, %s, %s, %s, %s, %s)",
                [("Goods.csv", 1, "0901", ["0901"], "Coffee", "Coffee 0901"),
                 ("Goods.csv", 2, "0901", ["0901"], "Roasted coffee beans", "Roasted coffee beans 0901"),
                 ("Services.csv", 1, "9965", ["9965"], "Goods transport services",
                  "Goods transport services 9965 refrigeration condition"),
                 ("Goods.csv", 3, "NONE", [], None, None)],
            )

    def test_exact_lookup_preserves_leading_zero_and_returns_multiple_rates(self):
        rows = exact_lookup(self.conn, "0901")
        self.assertEqual(len(rows), 2)
        self.assertEqual(exact_lookup(self.conn, "9010"), [])

    def test_bm25_includes_conditions(self):
        rows, method = keyword_search(self.conn, "refrigeration")
        self.assertEqual(method, "bm25")
        self.assertEqual([r[0] for r in rows], ["9965"])

    def test_bm25_matches_and_applies_limit(self):
        rows, method = keyword_search(self.conn, "coffee", 1)
        self.assertEqual(method, "bm25")
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0][2], 0)

    def test_plain_text_stemming_stopwords_and_multiple_terms(self):
        rows, method = keyword_search(self.conn, "the goods transports")
        self.assertEqual(method, "bm25")
        self.assertEqual([row[0] for row in rows], ["9965"])

    def test_typo_uses_trigram(self):
        rows, method = keyword_search(self.conn, "cofee")
        self.assertEqual(method, "trigram")
        self.assertIn("0901", [row[0] for row in rows])

    def test_bm25_scores_are_positive_and_descending(self):
        rows, method = keyword_search(self.conn, "coffee")
        self.assertEqual(method, "bm25")
        scores = [row[2] for row in rows]
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(score > 0 for score in scores))
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_bm25_retrieves_any_matching_term(self):
        rows, method = keyword_search(self.conn, "coffee transport")
        self.assertEqual(method, "bm25")
        self.assertEqual({row[0] for row in rows}, {"0901", "9965"})

    def test_no_matches_and_stopwords(self):
        for query in ("zzzzzzzzzz", "the and", "' ; --"):
            rows, method = keyword_search(self.conn, query)
            self.assertEqual((rows, method), ([], "trigram"))


if __name__ == "__main__":
    unittest.main()
