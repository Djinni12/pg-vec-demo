"""Stage 2 Batch Runner: Structural Parsing & Intermediate Normalization.

Runs structural parsing on all notifications in data/notifications/:
- Normalizes rate schedules, exemptions, annexures, and amendment operations.
- Enforces strict effective-date scope inheritance.
- Saves per-file normalized JSON in data/notifications/normalized/
- Saves master summary report in data/notifications/reports/stage2_normalization_report.json
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parsers.notification_parser import extract_pdf_document
from src.parsers.structural_parser import (
    NormalizedNotification,
    normalize_notification,
)


def run_stage2_normalization(
    pdf_dir: Path = PROJECT_ROOT / "data" / "notifications",
    output_dir: Path = PROJECT_ROOT / "data" / "notifications" / "normalized",
    report_path: Path = PROJECT_ROOT / "data" / "notifications" / "reports" / "stage2_normalization_report.json",
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    print(f"\n{'='*100}")
    print(f"STAGE 2: GST NOTIFICATION STRUCTURAL PARSING & INTERMEDIATE NORMALIZATION")
    print(f"Target Directory: {pdf_dir}")
    print(f"Total PDFs: {len(pdf_files)}")
    print(f"{'='*100}\n")

    results: List[NormalizedNotification] = []
    summary_stats = {
        "total_files": len(pdf_files),
        "successfully_normalized": 0,
        "needs_visual_extraction": 0,
        "total_schedule_entries": 0,
        "total_amendment_operations": 0,
        "total_annexures": 0,
        "per_file_summary": [],
    }

    for p in pdf_files:
        doc = extract_pdf_document(p)
        norm = normalize_notification(doc)
        results.append(norm)

        # Save individual JSON
        stem = p.stem
        file_json_path = output_dir / f"{stem}.json"
        with open(file_json_path, "w", encoding="utf-8") as f:
            json.dump(norm.to_dict(), f, indent=2)

        is_success = norm.normalization_status == "SUCCESS"
        if is_success:
            summary_stats["successfully_normalized"] += 1
        elif norm.normalization_status == "NEEDS_VISUAL_EXTRACTION":
            summary_stats["needs_visual_extraction"] += 1

        sched_count = sum(len(entries) for entries in norm.schedules.values())
        op_count = len(norm.amendment_operations)
        annex_count = len(norm.annexures)

        summary_stats["total_schedule_entries"] += sched_count
        summary_stats["total_amendment_operations"] += op_count
        summary_stats["total_annexures"] += annex_count

        summary_stats["per_file_summary"].append({
            "file": norm.file_name,
            "notif_number": norm.notification_number,
            "parser_type": norm.parser_type,
            "status": norm.normalization_status,
            "sched_entries": sched_count,
            "amend_ops": op_count,
            "annexures": annex_count,
            "warnings_count": len(norm.validation_warnings),
        })

    # Print summary table
    print(f"{'Filename':<24} | {'Parser Type':<24} | {'Status':<18} | {'Sched Entries':<14} | {'Amend Ops':<10} | {'Annexures':<9} | {'Warnings':<8}")
    print("-" * 128)
    for s in summary_stats["per_file_summary"]:
        print(f"{s['file']:<24} | {s['parser_type']:<24} | {s['status']:<18} | {s['sched_entries']:<14} | {s['amend_ops']:<10} | {s['annexures']:<9} | {s['warnings_count']:<8}")
    print("-" * 128)
    print(f"Total Successfully Normalized: {summary_stats['successfully_normalized']}/20 (19 extractable + 1 excluded vector outline)")
    print(f"Total Reconstructed Schedule Entries: {summary_stats['total_schedule_entries']:,}")
    print(f"Total Discrete Amendment Operations:  {summary_stats['total_amendment_operations']}")
    print(f"Total Attached Annexures:             {summary_stats['total_annexures']}")
    print(f"{'='*100}\n")

    # Save summary report
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

    return summary_stats


if __name__ == "__main__":
    run_stage2_normalization()
