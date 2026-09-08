"""Generate and inspect structure-aware CGST Act chunks."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import json

from act_chunker import (
    DEFAULT_CHUNKS_JSON,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_SECTIONS_JSON,
    chunk_sections,
    chunk_size_report,
    load_sections,
)


def preview(text, length=360):
    return " ".join(text.split())[:length]


def print_samples(chunks, limit):
    print("SAMPLE CHUNKS")
    for chunk in chunks[:limit]:
        metadata = chunk["metadata"]
        subsections = ", ".join(metadata.get("subsection_numbers") or []) or "whole section"
        print(f"{chunk['chunk_id']} | section {metadata.get('section_number')} | subsections {subsections}")
        print(f"strategy={metadata.get('strategy')} | tokens={chunk['token_count']} | status={metadata.get('status')} | chapter={metadata.get('chapter')}")
        print(preview(chunk["text"]))
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sections_json", nargs="?", default=DEFAULT_SECTIONS_JSON)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_CHUNKS_JSON)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    sections = load_sections(args.sections_json)
    chunks = chunk_sections(sections, max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens)
    report = chunk_size_report(chunks)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Sections loaded: {len(sections)}")
    print(f"Chunks generated: {report['chunk_count']}")
    print(f"Token size min/avg/max: {report['min_tokens']} / {report['avg_tokens']} / {report['max_tokens']}")
    print("Sections producing the most chunks:")
    for section_number, count in report["sections_with_most_chunks"]:
        print(f"- {section_number}: {count}")
    print()
    print_samples(chunks, args.samples)
    print(f"Wrote JSON: {args.json_output}")


if __name__ == "__main__":
    main()
