"""Run with: venv/bin/python -m unittest test_keyword_search -v.

Set TEST_DATABASE_URL to also run PostgreSQL integration tests. They use a
temporary documents table and roll back, leaving persistent documents intact.
"""

import os
import unittest
from unittest.mock import patch

from keyword_search import keyword_search


class KeywordRoutingTests(unittest.TestCase):
    def test_bm25_hits_do_not_call_fallback_or_top_up(self):
        rows = [("I10", "Essential hypertension", 0.1)]
        with patch("keyword_search.bm25_search", return_value=rows), patch(
            "keyword_search.fuzzy_search"
        ) as fuzzy:
            self.assertEqual(keyword_search(object(), "hypertension"), (rows, "bm25"))
            fuzzy.assert_not_called()

    def test_no_bm25_hits_use_same_query_and_limit_for_fallback(self):
        conn = object()
        rows = [("I10", "Essential hypertension", 0.4)]
        with patch("keyword_search.bm25_search", return_value=[]) as bm25, patch(
            "keyword_search.fuzzy_search", return_value=rows
        ) as fuzzy:
            self.assertEqual(keyword_search(conn, "hypertensoin", 3), (rows, "trigram"))
            bm25.assert_called_once_with(conn, "hypertensoin", 3)
            fuzzy.assert_called_once_with(conn, "hypertensoin", 3)

    def test_database_error_does_not_trigger_fallback(self):
        with patch("keyword_search.bm25_search", side_effect=RuntimeError("database error")), patch(
            "keyword_search.fuzzy_search"
        ) as fuzzy:
            with self.assertRaises(RuntimeError):
                keyword_search(object(), "hypertension")
            fuzzy.assert_not_called()

    def test_blank_query_does_not_access_database(self):
        self.assertEqual(keyword_search(None, "  "), ([], "bm25"))

    def test_nonpositive_limit_is_rejected(self):
        for limit in (0, -1):
            with self.assertRaises(ValueError):
                keyword_search(None, "hypertension", limit)


@unittest.skipUnless(os.environ.get("TEST_DATABASE_URL"), "TEST_DATABASE_URL not set")
class PostgreSQLKeywordTests(unittest.TestCase):
    def setUp(self):
        import psycopg

        self.conn = psycopg.connect(os.environ["TEST_DATABASE_URL"])
        self.addCleanup(self.conn.close)
        self.addCleanup(self.conn.rollback)
        self.conn.execute("CREATE TEMP TABLE documents (code TEXT, description TEXT)")
        self.conn.execute("SET LOCAL pg_trgm.similarity_threshold = 0.3")
        self.conn.execute(
            "CREATE INDEX documents_description_bm25_idx ON documents "
            "USING bm25(description) WITH (text_config='english')"
        )
        with self.conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO documents VALUES (%s, %s)",
                [("I10", "Essential hypertension"),
                 ("I15.0", "Renovascular hypertension"),
                 ("I50.9", "Heart failure, unspecified"),
                 ("NONE", None)],
            )

    def test_bm25_matches_and_applies_limit(self):
        rows, method = keyword_search(self.conn, "hypertension", 1)
        self.assertEqual(method, "bm25")
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0][2], 0)

    def test_plain_text_stemming_stopwords_and_multiple_terms(self):
        rows, method = keyword_search(self.conn, "the heart failures")
        self.assertEqual(method, "bm25")
        self.assertEqual([row[0] for row in rows], ["I50.9"])

    def test_typo_uses_trigram(self):
        rows, method = keyword_search(self.conn, "hypertensoin")
        self.assertEqual(method, "trigram")
        self.assertIn("I10", [row[0] for row in rows])

    def test_bm25_scores_are_positive_and_descending(self):
        rows, method = keyword_search(self.conn, "hypertension")
        self.assertEqual(method, "bm25")
        scores = [row[2] for row in rows]
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(score > 0 for score in scores))
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_bm25_retrieves_any_matching_term(self):
        rows, method = keyword_search(self.conn, "hypertension heart")
        self.assertEqual(method, "bm25")
        self.assertEqual({row[0] for row in rows}, {"I10", "I15.0", "I50.9"})

    def test_no_matches_and_stopwords(self):
        for query in ("zzzzzzzzzz", "the and", "' ; --"):
            rows, method = keyword_search(self.conn, query)
            self.assertEqual((rows, method), ([], "trigram"))


if __name__ == "__main__":
    unittest.main()
