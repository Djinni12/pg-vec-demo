"""Run with: venv/bin/python -m unittest tests.test_act_chunker -v."""

import unittest

from act_chunker import chunk_sections


class WhitespaceTokenizer:
    def tokenize(self, text):
        return text.split()


def section(number, title, content, subsections=None, status="current", chapter="CHAPTER I TEST"):
    return {
        "section_number": number,
        "section_title": title,
        "content": content,
        "subsections": subsections or [],
        "status": status,
        "chapter": chapter,
    }


class ActChunkerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = WhitespaceTokenizer()

    def test_small_section_stays_whole(self):
        chunks = chunk_sections(
            [section("1", "Short", "(1) tiny text", [{"subsection_number": "1", "text": "(1) tiny text"}])],
            max_tokens=20,
            overlap_tokens=2,
            tokenizer=self.tokenizer,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["metadata"]["strategy"], "section")
        self.assertEqual(chunks[0]["metadata"]["subsection_numbers"], [])
        self.assertIn("Section 1. Short", chunks[0]["text"])

    def test_large_section_groups_complete_subsections_without_overlap(self):
        sections = [
            section(
                "2",
                "Grouped",
                "(1) alpha beta gamma\n(2) delta epsilon zeta\n(3) eta theta iota",
                [
                    {"subsection_number": "1", "text": "(1) alpha beta gamma"},
                    {"subsection_number": "2", "text": "(2) delta epsilon zeta"},
                    {"subsection_number": "3", "text": "(3) eta theta iota"},
                ],
            )
        ]

        chunks = chunk_sections(sections, max_tokens=12, overlap_tokens=2, tokenizer=self.tokenizer)

        self.assertEqual([chunk["metadata"]["subsection_numbers"] for chunk in chunks], [["1", "2"], ["3"]])
        combined = "\n".join(chunk["text"] for chunk in chunks)
        self.assertEqual(combined.count("alpha beta gamma"), 1)
        self.assertEqual(combined.count("delta epsilon zeta"), 1)

    def test_oversized_subsection_is_split_with_overlap_inside_that_subsection(self):
        big_text = "(1) " + " ".join(f"word{i}" for i in range(20))
        chunks = chunk_sections(
            [section("3", "Huge subsection", big_text, [{"subsection_number": "1", "text": big_text}])],
            max_tokens=10,
            overlap_tokens=2,
            tokenizer=self.tokenizer,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["metadata"]["strategy"] == "subsection_token_split" for chunk in chunks))
        self.assertTrue(all(chunk["metadata"]["subsection_numbers"] == ["1"] for chunk in chunks))
        self.assertEqual(chunks[0]["text"].split()[-2:], chunks[1]["text"].split()[4:6])

    def test_never_combines_different_sections(self):
        chunks = chunk_sections(
            [
                section("4", "First", "first content"),
                section("5", "Second", "second content"),
            ],
            max_tokens=20,
            overlap_tokens=2,
            tokenizer=self.tokenizer,
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual([chunk["metadata"]["section_number"] for chunk in chunks], ["4", "5"])
        self.assertNotIn("second content", chunks[0]["text"])
        self.assertNotIn("first content", chunks[1]["text"])

    def test_metadata_preserves_section_status_chapter_and_parent_context(self):
        chunks = chunk_sections(
            [section("6", "Meta", "small", status="omitted", chapter="CHAPTER II META")],
            max_tokens=20,
            overlap_tokens=2,
            tokenizer=self.tokenizer,
        )

        metadata = chunks[0]["metadata"]
        self.assertEqual(metadata["section_number"], "6")
        self.assertEqual(metadata["section_title"], "Meta")
        self.assertEqual(metadata["status"], "omitted")
        self.assertEqual(metadata["chapter"], "CHAPTER II META")
        self.assertTrue(chunks[0]["text"].startswith("Section 6. Meta"))


if __name__ == "__main__":
    unittest.main()
