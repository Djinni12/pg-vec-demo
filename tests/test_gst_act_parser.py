"""Run with: venv/bin/python -m unittest test_gst_act_parser -v."""

import unittest
from unittest.mock import patch

from gst_act_parser import parse_sections, subsection_boundary_warnings


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

    def test_omitted_sections_and_historical_duplicates(self):
        text = """
        1 [Section 38. Communication of details of inward supplies and input tax credit. *
        Current section content.
        1. Substituted by s. 104 for
        Section 38. Furnishing details of inward supplies. -
        Old quoted text.
        * Section 43A. 1 [*** ]
        * Enforced w.e.f. 22nd June, 2017.
        1. Omitted by amendment for
        Section 43A. Old heading. -
        Historical text.
        """
        with patch("gst_act_parser.extract_pdf_text", return_value=text):
            sections = parse_sections("ignored.pdf")

        self.assertEqual([section["section_number"] for section in sections], ["38", "43A"])
        self.assertEqual(sections[0]["section_title"], "Communication of details of inward supplies and input tax credit")
        self.assertEqual(sections[0]["status"], "current")
        self.assertEqual(sections[1]["section_title"], "Omitted")
        self.assertEqual(sections[1]["status"], "omitted")

    def test_chapter_headings_do_not_stay_in_previous_section_content(self):
        text = """
        * Section 6. Authorisation of officers.-
        Section content.
        CHAPTER III LEVY AND COLLECTION OF TAX
        * Section 7. Scope of supply.-
        Supply content under section 15.
        """
        with patch("gst_act_parser.extract_pdf_text", return_value=text):
            sections = parse_sections("ignored.pdf")

        self.assertEqual([section["section_number"] for section in sections], ["6", "7"])
        self.assertNotIn("CHAPTER III", sections[0]["content"])
        self.assertIn("under section 15", sections[1]["content"])

    def test_chapter_headings_with_dots_do_not_stay_in_previous_section_content(self):
        text = """
        Section 2. Definitions.-
        Definition content.
        CHAPTER II. ADMINISTRATION
        Section 3. Appointment of Officers.-
        Officer content.
        CHAPTER III . LEVY AND COLLECTION OF TAX
        Section 4. Levy.-
        Levy content.
        """
        with patch("gst_act_parser.extract_pdf_text", return_value=text):
            sections = parse_sections("ignored.pdf")

        self.assertEqual([section["section_number"] for section in sections], ["2", "3", "4"])
        self.assertNotIn("CHAPTER II", sections[0]["content"])
        self.assertNotIn("CHAPTER III", sections[1]["content"])

    def test_parses_top_level_subsections_and_keeps_clauses_inside_parent(self):
        text = """
        * Section 7. Scope of supply.-
        (1) First subsection text.
        (a) Clause remains inside subsection one.
        (i) Roman clause also remains inside subsection one.
        (2) Second subsection refers to sub-section (1) and section 15.
        """
        with patch("gst_act_parser.extract_pdf_text", return_value=text):
            section = parse_sections("ignored.pdf")[0]

        self.assertIn("(1) First subsection text.", section["content"])
        self.assertEqual([sub["subsection_number"] for sub in section["subsections"]], ["1", "2"])
        self.assertIn("(a) Clause remains inside subsection one.", section["subsections"][0]["text"])
        self.assertIn("(i) Roman clause also remains inside subsection one.", section["subsections"][0]["text"])
        self.assertIn("sub-section (1)", section["subsections"][1]["text"])

    def test_parses_amended_alphanumeric_subsection_marker(self):
        text = """
        * Section 7. Scope of supply.-
        5 [(1A) Where certain activities are treated as supply.
        (2) Subject to sub-section (1A), further text.
        """
        with patch("gst_act_parser.extract_pdf_text", return_value=text):
            section = parse_sections("ignored.pdf")[0]

        self.assertEqual([sub["subsection_number"] for sub in section["subsections"]], ["1A", "2"])
        self.assertTrue(section["subsections"][0]["text"].startswith("(1A) Where certain activities"))

    def test_parses_spaced_numeric_subsection_marker(self):
        text = """
        Section 5. Levy and collection.-
        ( 1 ) First spaced subsection.
        ( 2 ) Second spaced subsection.
        """
        with patch("gst_act_parser.extract_pdf_text", return_value=text):
            section = parse_sections("ignored.pdf")[0]

        self.assertEqual([sub["subsection_number"] for sub in section["subsections"]], ["1", "2"])
        self.assertTrue(section["subsections"][0]["text"].startswith("(1) First spaced subsection"))

    def test_sections_without_subsections_remain_valid(self):
        text = """
        * Section 42. Omitted.-
        Historical omission text without top-level numeric markers.
        """
        with patch("gst_act_parser.extract_pdf_text", return_value=text):
            section = parse_sections("ignored.pdf")[0]

        self.assertEqual(section["subsections"], [])
        self.assertEqual(subsection_boundary_warnings(section), [])


if __name__ == "__main__":
    unittest.main()
