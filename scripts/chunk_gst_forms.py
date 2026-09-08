"""Generate GST Forms chunks JSON and print an inspection report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from form_chunker import DEFAULT_FORM_CHUNKS_JSON, DEFAULT_FORMS_JSON, build_form_chunks, chunk_report, load_forms, save_chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forms_json", nargs="?", type=Path, default=DEFAULT_FORMS_JSON)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_FORM_CHUNKS_JSON)
    parser.add_argument("--max-tokens", type=int, default=450)
    parser.add_argument("--overlap-tokens", type=int, default=60)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    forms = load_forms(args.forms_json)
    chunks = build_form_chunks(forms, max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens)
    save_chunks(chunks, args.json_output)
    report = chunk_report(chunks)

    print(f"Forms loaded: {len(forms)}")
    print(f"Chunks generated: {report['chunk_count']}")
    print(f"Token size min/avg/max: {report['min_tokens']} / {report['avg_tokens']:.1f} / {report['max_tokens']}")
    print("Forms producing most chunks:")
    for form_number, count in report["forms_by_chunk_count"][:10]:
        print(f"- {form_number}: {count}")
    print(f"Wrote JSON: {args.json_output}")
    for chunk in chunks[: args.samples]:
        preview = " ".join(chunk["text"].split())[:250]
        print(f"\n{chunk['chunk_id']} | tokens={chunk['token_count']} | {chunk['metadata']['chunk_strategy']}")
        print(preview)


if __name__ == "__main__":
    main()
