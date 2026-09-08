"""Run with: venv/bin/python -m unittest tests.test_rules_chunker -v."""

import unittest

from rules_chunker import chunk_rules


class WhitespaceTokenizer:
    def tokenize(self, text):
        return text.split()


def rule(number, title, content, subrules=None, status="current"):
    return {
        "rule_number": number,
        "rule_title": title,
        "chapter": "CHAPTER I",
        "chapter_title": "PRELIMINARY",
        "status": status,
        "content": content,
        "subrules": subrules or [],
    }


class RulesChunkerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = WhitespaceTokenizer()

    def test_small_rule_stays_whole_and_preserves_metadata(self):
        chunks = chunk_rules([rule("1", "Short", "small content")], max_tokens=20, overlap_tokens=2, tokenizer=self.tokenizer)
        self.assertEqual(len(chunks), 1)
        metadata = chunks[0]["metadata"]
        self.assertEqual(metadata["rule_number"], "1")
        self.assertEqual(metadata["rule_title"], "Short")
        self.assertEqual(metadata["chapter"], "CHAPTER I")
        self.assertEqual(metadata["chapter_title"], "PRELIMINARY")
        self.assertEqual(metadata["status"], "current")
        self.assertEqual(metadata["chunk_strategy"], "rule")
        self.assertTrue(chunks[0]["text"].startswith("CHAPTER I PRELIMINARY\nRule 1. Short"))

    def test_large_rule_groups_complete_subrules(self):
        chunks = chunk_rules(
            [
                rule(
                    "2",
                    "Grouped",
                    "(1) alpha beta gamma (2) delta epsilon zeta",
                    [
                        {"subrule_number": "1", "text": "(1) alpha beta gamma"},
                        {"subrule_number": "2", "text": "(2) delta epsilon zeta"},
                    ],
                )
            ],
            max_tokens=12,
            overlap_tokens=2,
            tokenizer=self.tokenizer,
        )
        self.assertEqual([chunk["metadata"]["subrule_numbers"] for chunk in chunks], [["1"], ["2"]])

    def test_oversized_subrule_uses_token_split(self):
        text = "(1) " + " ".join(f"word{i}" for i in range(20))
        chunks = chunk_rules(
            [rule("3", "Big", text, [{"subrule_number": "1", "text": text}])],
            max_tokens=10,
            overlap_tokens=2,
            tokenizer=self.tokenizer,
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["metadata"]["chunk_strategy"] == "subrule_token_split" for chunk in chunks))
        self.assertTrue(all(chunk["metadata"]["subrule_numbers"] == ["1"] for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
