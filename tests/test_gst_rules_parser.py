"""Run with: venv/bin/python -m unittest tests.test_gst_rules_parser -v."""

import unittest
from unittest.mock import patch
from gst_rules_parser import parse_rules


class GSTRulesParserTests(unittest.TestCase):
    def test_parses_chapter_rule_and_subrules(self):
        text = """
        Table Of Content
        Rule 3
        Intimation for composition levy
        Central Goods and Services Tax (CGST) Rules, 2017 Part A (Rules)
        CHAPTER II
        1 [COMPOSITION LEVY]
        Rule 3. Intimation for composition levy. -
        (1) First sub-rule under rule 8.
        (a) clause stays inside.
        (2) Second sub-rule.
        """
        with patch("gst_rules_parser.extract_pdf_text", return_value=text):
            rules = parse_rules("ignored.pdf")

        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["rule_number"], "3")
        self.assertEqual(rules[0]["chapter"], "CHAPTER II")
        self.assertEqual(rules[0]["chapter_title"], "COMPOSITION LEVY")
        self.assertEqual([sub["subrule_number"] for sub in rules[0]["subrules"]], ["1", "2"])
        self.assertIn("(a) clause stays inside", rules[0]["subrules"][0]["text"])
        self.assertIn("under rule 8", rules[0]["subrules"][0]["text"])

    def test_parses_alphanumeric_and_spaced_rule_numbers(self):
        text = """
        Central Goods and Services Tax (CGST) Rules, 2017 Part A (Rules)
        CHAPTER IV. DETERMINATION OF VALUE OF SUPPLY
        1 [Rule 88 C. Manner of dealing with difference. -
        (1) Current text.
        Rule113. Order of Appellate Authority or Appellate Tribunal.-
        (1) Order text.
        """
        with patch("gst_rules_parser.extract_pdf_text", return_value=text):
            rules = parse_rules("ignored.pdf")

        self.assertEqual([rule["rule_number"] for rule in rules], ["88C", "113"])
        self.assertEqual(rules[0]["rule_title"], "Manner of dealing with difference")

    def test_omitted_rule_and_historical_quoted_rule_are_not_duplicated(self):
        text = """
        Central Goods and Services Tax (CGST) Rules, 2017 Part A (Rules)
        CHAPTER X MISCELLANEOUS
        Rule 95A. 1 [***]
        1. Omitted by notification for
        Rule 95A. Refund of taxes to old text.-
        (1) Historical text.
        Rule 96. Refund of integrated tax paid. -
        Current text without sub-rules.
        """
        with patch("gst_rules_parser.extract_pdf_text", return_value=text):
            rules = parse_rules("ignored.pdf")

        self.assertEqual([rule["rule_number"] for rule in rules], ["95A", "96"])
        self.assertEqual(rules[0]["status"], "omitted")
        self.assertEqual(rules[0]["rule_title"], "Omitted")
        self.assertEqual(rules[0]["subrules"], [])

    def test_references_do_not_create_rule_or_subrule_boundaries(self):
        text = """
        Central Goods and Services Tax (CGST) Rules, 2017 Part A (Rules)
        CHAPTER I PRELIMINARY
        Rule 2. Definitions. -
        In these rules, reference under rule 8 and sub-rule (2) should stay text.
        (a) clause text.
        Rule 3. Next rule. -
        (1) Real sub-rule.
        """
        with patch("gst_rules_parser.extract_pdf_text", return_value=text):
            rules = parse_rules("ignored.pdf")

        self.assertEqual([rule["rule_number"] for rule in rules], ["2", "3"])
        self.assertEqual(rules[0]["subrules"], [])
        self.assertIn("under rule 8", rules[0]["content"])

    def test_table_column_markers_are_not_subrules(self):
        text = """
        Central Goods and Services Tax (CGST) Rules, 2017 Part A (Rules)
        CHAPTER II COMPOSITION LEVY
        Rule 7. Rate of tax of the composition levy. -
        Intro text before table.
        TABLE
        Sl. No. Category Rate
        (1) (2) (3)
        1. Manufacturers half percent.
        Rule 8. Application for registration. -
        (1) Real registration sub-rule.
        """
        with patch("gst_rules_parser.extract_pdf_text", return_value=text):
            rules = parse_rules("ignored.pdf")

        self.assertEqual(rules[0]["rule_number"], "7")
        self.assertEqual(rules[0]["subrules"], [])
        self.assertEqual([sub["subrule_number"] for sub in rules[1]["subrules"]], ["1"])

    def test_illustration_markers_are_not_subrules(self):
        text = """
        Central Goods and Services Tax (CGST) Rules, 2017 Part A (Rules)
        CHAPTER IV DETERMINATION OF VALUE OF SUPPLY
        Rule 27. Value of supply. -
        Main valuation text.
        Illustration :
        (1) First example.
        (2) Second example.
        Rule 28. Next rule. -
        (1) Real sub-rule.
        """
        with patch("gst_rules_parser.extract_pdf_text", return_value=text):
            rules = parse_rules("ignored.pdf")

        self.assertEqual(rules[0]["rule_number"], "27")
        self.assertEqual(rules[0]["subrules"], [])
        self.assertIn("(1) First example", rules[0]["content"])

    def test_wrapped_sub_section_reference_is_not_subrule(self):
        text = """
        Central Goods and Services Tax (CGST) Rules, 2017 Part A (Rules)
        CHAPTER XIV TRANSITIONAL PROVISIONS
        Rule 117. Transitional credit. -
        (1) First real sub-rule.
        (2) Second real sub-rule.
        (3) Third real sub-rule.
        (4) Text referring to sub-section
        (3) of section 140 should not split.
        Rule 118. Next rule. -
        Text.
        """
        with patch("gst_rules_parser.extract_pdf_text", return_value=text):
            rules = parse_rules("ignored.pdf")

        self.assertEqual([sub["subrule_number"] for sub in rules[0]["subrules"]], ["1", "2", "3", "4"])
        self.assertIn("sub-section", rules[0]["subrules"][3]["text"])
        self.assertIn("(3) of section 140", rules[0]["subrules"][3]["text"])

    def test_preserves_late_real_subrules_after_table(self):
        text = """
        Central Goods and Services Tax (CGST) Rules, 2017 Part A (Rules)
        CHAPTER XVI E-WAY RULES
        Rule 138. Information to be furnished. -
        (10) Table-bearing real sub-rule.
        Table
        Sl. No Distance Validity
        (1) (2) (3)
        1. Short distance One day
        (11) Real later sub-rule.
        (12) Another real later sub-rule.
        (13) Another real later sub-rule.
        (14) Another real later sub-rule.
        """
        with patch("gst_rules_parser.extract_pdf_text", return_value=text):
            rule = parse_rules("ignored.pdf")[0]

        self.assertEqual(
            [sub["subrule_number"] for sub in rule["subrules"]],
            ["10", "11", "12", "13", "14"],
        )


if __name__ == "__main__":
    unittest.main()
