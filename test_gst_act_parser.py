"""Run with: venv/bin/python -m unittest test_gst_act_parser -v."""

import unittest
from unittest.mock import patch

from gst_act_parser import parse_sections


class GSTActParserTests(unittest.TestCase):
    def test_parses_section_headings_and_boundaries(self):
        text = """
        Table Of Content
        Section 7
        Scope of supply
        * Section 7. Scope of supply.-
        (1) For the purposes of this Act, the expression supply includes sale.
        Section 8. Tax liability on composite and mixed supplies. -
        Composite supply text.
        """
        with patch("gst_act_parser.extract_pdf_text", return_value=text):
            sections = parse_sections("ignored.pdf")

        self.assertEqual([section["section_number"] for section in sections], ["7", "8"])
        self.assertEqual(sections[0]["section_title"], "Scope of supply")
        self.assertIn("supply includes sale", sections[0]["content"])
        self.assertEqual(sections[1]["section_title"], "Tax liability on composite and mixed supplies")
        self.assertNotIn("Section 7", sections[1]["content"])

    def test_parses_alphanumeric_and_wrapped_headings(self):
        text = """
        1 [ Section 31A. Facility of digital payment to recipient.-
        Payment content.
        * Section 27. Special provisions relating to casual taxable person and non-
        resident taxable person.-
        Registration content.
        Section 17A. Example alphanumeric section . -
        Alphanumeric content.
        """
        with patch("gst_act_parser.extract_pdf_text", return_value=text):
            sections = parse_sections("ignored.pdf")

        self.assertEqual([section["section_number"] for section in sections], ["31A", "27", "17A"])
        self.assertEqual(sections[0]["section_title"], "Facility of digital payment to recipient")
        self.assertEqual(
            sections[1]["section_title"],
            "Special provisions relating to casual taxable person and non-resident taxable person",
        )
        self.assertEqual(sections[2]["section_title"], "Example alphanumeric section")


if __name__ == "__main__":
    unittest.main()
