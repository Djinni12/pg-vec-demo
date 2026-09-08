"""Embed generated CGST Act chunks with BAAI/bge-m3 for inspection."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse

from act_embedder import (
    DEFAULT_CHUNKS_JSON,
    DEFAULT_EMBEDDINGS_JSONL,
    MODEL_NAME,
    build_embedding_records,
    load_chunks,
    write_json,
    write_jsonl,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks_json", nargs="?", default=DEFAULT_CHUNKS_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_EMBEDDINGS_JSONL)
    parser.add_argument("--format", choices=("jsonl", "json"), default="jsonl")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-normalize", action="store_true", help="Disable embedding normalization")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks_json)
    records, report = build_embedding_records(
        chunks,
        model_name=args.model_name,
        batch_size=args.batch_size,
        normalize_embeddings=not args.no_normalize,
    )

    if args.format == "json":
        write_json(records, args.output)
    else:
        write_jsonl(records, args.output)

    print(f"Chunks loaded: {report['chunk_count']}")
    print(f"Embeddings generated: {report['embedding_count']}")
    print(f"Embedding dimension: {report['embedding_dimension']}")
    print(f"Model name: {report['model_name']}")
    print(f"Normalized embeddings: {report['embedding_normalized']}")
    failed = report["failed_or_empty_chunks"]
    print(f"Failed/empty chunks: {len(failed)}")
    if failed:
        for chunk_id in failed:
            print(f"- {chunk_id}")
    print(f"Wrote {args.format.upper()}: {args.output}")


if __name__ == "__main__":
    main()
