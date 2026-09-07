"""GST parsing tests and optional transactional PostgreSQL ingestion checks."""

import csv
from decimal import Decimal
import os
from pathlib import Path
import tempfile
import unittest

from ingest_gst import FIELDS, exact_codes, rate_percent, read_dataset, save_records


class GSTFixture:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_dir = Path(self.tmp.name)
        for filename, fields in FIELDS.items():
            record_type, code_field, description, cgst, sgst, igst, extra = fields
            headers = ["S. No.", *fields[1:]]
            if record_type == "goods":
                headers.insert(0, "Schedules")
            row = {key: "" for key in headers}
            row.update({"S. No.": "1", code_field: "0901" if record_type == "goods" else "Heading 9965",
                        description: "Coffee\xa0 beans" if record_type == "goods" else "Transport services",
                        cgst: "0.025" if record_type == "goods" else "2.5 or 6",
                        sgst: "0.025" if record_type == "goods" else "Nil",
                        igst: "0.05" if record_type == "goods" else "5 or 12",
                        extra: "12%" if record_type == "goods" else "Only refrigerated goods"})
            if record_type == "goods":
                row["Schedules"] = "I"
            with (self.data_dir / filename).open("w", encoding="cp1252", newline="") as handle:
                writer = csv.DictWriter(handle, headers)
                writer.writeheader()
                writer.writerow(row)
                writer.writerow({**row, description: ""})
                if record_type == "goods":
                    writer.writerow({**row, description: "[Omitted]"})
                else:
                    writer.writerow({**row, "S. No.": "(1)", description: "(3)"})


class GSTParsingTests(GSTFixture, unittest.TestCase):
    def test_rates_have_file_specific_units_and_preserve_unknowns(self):
        self.assertEqual(rate_percent("0.05", "goods"), Decimal("5"))
        self.assertEqual(rate_percent("0.00125", "goods"), Decimal("0.125"))
        self.assertEqual(rate_percent("5", "service"), Decimal("5"))
        self.assertEqual(rate_percent("Nil", "service"), Decimal("0"))
        for value in ("", "5 or 12", "Same rate as like goods", "(4)"):
            self.assertIsNone(rate_percent(value, "service"))

    def test_codes_preserve_zeros_and_parse_explicit_lists_only(self):
        cases = [
            ("0202, 0203", "goods", ["0202", "0203"]),
            ("9405 91 00 or 9405 92 00", "goods", ["94059100", "94059200"]),
            ("Heading 9954 or Heading 9983", "service", ["9954", "9983"]),
            ("Heading 9965 (Goods transport services)", "service", ["9965"]),
            ("Section 5", "service", []),
            ("5004 to 5006", "goods", []),
            ("0507 [Except 050790]", "goods", []),
            ("Any Chapter", "goods", []),
            ("Heading 9983 or any other Heading of Chapter 99", "service", []),
            ("[9003", "goods", []),
        ]
        for text, kind, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(exact_codes(text, kind), expected)

    def test_realistic_csv_cleanup_metadata_and_search_fields(self):
        records, report = read_dataset(self.data_dir)
        self.assertEqual(len(records), 2)
        goods, service = records
        self.assertEqual(goods["description"], "Coffee beans")
        self.assertEqual(goods["metadata"]["raw_row"]["Description of Goods"], "Coffee\xa0 beans")
        self.assertEqual(goods["igst_rate_pct"], Decimal(5))
        self.assertIsNone(service["igst_rate_pct"])
        self.assertEqual(service["metadata"]["rates_raw"]["igst"], "5 or 12")
        self.assertIn("Only refrigerated goods", service["search_text"])
        self.assertNotIn("Only refrigerated goods", service["description"])
        self.assertEqual(report["Goods.csv"]["omitted"], 1)
        self.assertEqual(report["Services.csv"]["column_number_row"], 1)
        self.assertEqual(report["Goods.csv"]["encoding"], "cp1252")
        self.assertEqual([r["source_row"] for r in records], [1, 1])

    def test_unknown_columns_fail_before_import(self):
        (self.data_dir / "Goods.csv").write_text("unexpected\nvalue\n")
        with self.assertRaisesRegex(ValueError, "unexpected CSV columns"):
            read_dataset(self.data_dir)

    def test_embedding_count_mismatch_fails_before_writes(self):
        with self.assertRaisesRegex(ValueError, "one embedding"):
            save_records(None, [{"description": "Coffee"}], [])


@unittest.skipUnless(os.environ.get("TEST_DATABASE_URL"), "TEST_DATABASE_URL not set")
class GSTDatabaseTests(GSTFixture, unittest.TestCase):
    def test_reimport_updates_rates_removes_stale_rows_and_rolls_back(self):
        import numpy as np
        import psycopg
        from pgvector.psycopg import register_vector
        from keyword_search import exact_lookup, fuzzy_search

        records, _ = read_dataset(self.data_dir)
        with psycopg.connect(os.environ["TEST_DATABASE_URL"]) as conn:
            register_vector(conn)
            # Use the real schema's table definition, isolated in the temp schema.
            ddl = (Path(__file__).parent / "schema.sql").read_text()
            table = ddl[ddl.index("CREATE TABLE"):ddl.index("CREATE INDEX")]
            conn.execute(table.replace("CREATE TABLE IF NOT EXISTS", "CREATE TEMP TABLE"))
            embeddings = np.zeros((len(records), 384), dtype=np.float32)
            save_records(conn, records, embeddings)
            self.assertEqual(exact_lookup(conn, "0901")[0][5], Decimal(5))
            self.assertEqual(fuzzy_search(conn, "coffee beans")[0][0], "0901")
            records[0]["igst_rate_pct"] = Decimal(12)
            save_records(conn, records, embeddings)
            self.assertEqual(conn.execute("SELECT count(*) FROM gst_documents").fetchone()[0], 2)
            self.assertEqual(conn.execute("SELECT igst_rate_pct FROM gst_documents WHERE source_file = 'Goods.csv'").fetchone()[0], Decimal(12))
            records[0]["source_row"] = 2
            save_records(conn, records, embeddings)
            self.assertEqual(conn.execute("SELECT source_row FROM gst_documents WHERE source_file = 'Goods.csv'").fetchall(), [(2,)])
            conn.rollback()


if __name__ == "__main__":
    unittest.main()
