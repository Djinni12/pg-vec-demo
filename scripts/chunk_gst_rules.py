"""Generate and inspect structure-aware CGST Rules chunks."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import json

from rules_chunker import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_RULE_CHUNKS_JSON,
    DEFAULT_RULES_JSON,
    chunk_rules,
    chunk_size_report,
    load_rules,
)


def preview(text, length=360):
    return " ".join(text.split())[:length]


def print_samples(chunks, limit):
    print("SAMPLE RULE CHUNKS")
    for chunk in chunks[:limit]:
        metadata = chunk["metadata"]
        subrules = ", ".join(metadata.get("subrule_numbers") or []) or "whole rule"
        print(f"{chunk['chunk_id']} | rule {metadata.get('rule_number')} | sub-rules {subrules}")
        print(f"strategy={metadata.get('chunk_strategy')} | tokens={chunk['token_count']} | status={metadata.get('status')}")
        print(preview(chunk["text"]))
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rules_json", nargs="?", default=DEFAULT_RULES_JSON)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_RULE_CHUNKS_JSON)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    rules = load_rules(args.rules_json)
    chunks = chunk_rules(rules, max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens)
    report = chunk_size_report(chunks)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Rules loaded: {len(rules)}")
    print(f"Chunks generated: {report['chunk_count']}")
    print(f"Token size min/avg/max: {report['min_tokens']} / {report['avg_tokens']} / {report['max_tokens']}")
    print("Rules producing the most chunks:")
    for rule_number, count in report["rules_with_most_chunks"]:
        print(f"- {rule_number}: {count}")
    print()
    print_samples(chunks, args.samples)
    print(f"Wrote JSON: {args.json_output}")


if __name__ == "__main__":
    main()
