"""Inspect parsed Central GST Act sections before chunking."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from gst_act_parser import DEFAULT_PDF_PATH, parse_sections


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def preview(text, length=300):
    return " ".join(text.split())[:length]


def token_count(tokenizer, text, words_per_batch=250):
    words = text.split()
    total = 0
    for start in range(0, len(words), words_per_batch):
        total += len(tokenizer.tokenize(" ".join(words[start:start + words_per_batch])))
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", nargs="?", default=DEFAULT_PDF_PATH)
    parser.add_argument("--json-output", type=Path, help="Write parsed sections to this JSON file")
    args = parser.parse_args()

    sections = parse_sections(args.pdf_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    inspected_sections = []

    print(f"PDF: {args.pdf_path}")
    print(f"Total sections found: {len(sections)}")
    print()

    for section in sections:
        count = token_count(tokenizer, section["content"])
        section_preview = preview(section["content"])
        inspected_sections.append({**section, "token_count": count, "preview": section_preview})
        print(f"Section {section['section_number']}: {section['section_title']}")
        print(f"Token count: {count}")
        print(f"Preview: {section_preview}")
        print()

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(inspected_sections, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote JSON: {args.json_output}")


if __name__ == "__main__":
    main()
