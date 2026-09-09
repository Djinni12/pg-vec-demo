"""Generate legal-structure-first chunks for GST Central Tax (Rate) notifications."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.chunkers.notification_chunker import chunk_notifications_batch

NORMALIZED_DIR = Path("data/notifications/normalized")
OUTPUT_CHUNKS_PATH = Path("data/notifications/notification_chunks.json")
OUTPUT_REPORT_PATH = Path("data/notifications/reports/stage3_chunking_report.json")


def main():
    print("=" * 80)
    print("STAGE 3: GST NOTIFICATION LEGAL-STRUCTURE-FIRST CHUNKING")
    print(f"Source Directory: {NORMALIZED_DIR}")
    print("=" * 80)

    chunks, report = chunk_notifications_batch(NORMALIZED_DIR)

    # Save chunks JSON
    OUTPUT_CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump([ch.to_dict() for ch in chunks], f, indent=2, ensure_ascii=False)

    # Save report JSON
    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Total Chunks Generated: {report['total_chunks']}")
    print(f"Token Size min / avg / max: {report['min_tokens']} / {report['avg_tokens']} / {report['max_tokens']}")
    print(f"Oversized Chunks (>450 tokens): {report['oversized_chunks_count']}")
    print(f"Zero-Token Chunks: {report['zero_token_chunks_count']}")
    print(f"Split Units (>450 fallback): {report['split_units_count']}")
    print()

    print(f"{'Filename':25} | Chunks")
    print("-" * 35)
    for fname, count in report["per_file_chunk_counts"].items():
        print(f"{fname:25} | {count}")
    print("-" * 35)

    print()
    print("Chunks by chunk_type:")
    for c_type, count in sorted(report["chunks_by_type"].items()):
        print(f"  {c_type:30}: {count}")

    print()
    print("Chunks by chunk_strategy:")
    for c_strat, count in sorted(report["chunks_by_strategy"].items()):
        print(f"  {c_strat:30}: {count}")

    print("=" * 80)
    print(f"Wrote Chunks JSON: {OUTPUT_CHUNKS_PATH}")
    print(f"Wrote Report JSON: {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    main()
