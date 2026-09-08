"""Structured GST rate parser for GST rates2025.pdf.

Parses official 2025 CGST goods rates, exemptions, conditional rates, and compensation cess
into normalized, structured records for PostgreSQL storage.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

import pymupdf


DEFAULT_PDF_PATH = Path("data/gst/GST rates2025.pdf")
FALLBACK_PDF_PATH = Path("data/gst/csvs/GST rates2025.pdf")

# Standard Annexure condition text for Notification 02/2022-Central Tax (Rate)
NOTIF_02_2022_COND_1_TEXT = (
    "(a) credit of input tax charged on goods or services used exclusively in supplying such goods "
    "has not been taken; and (b) credit of input tax charged on goods or services used partly for "
    "supplying such goods and partly for effecting other supplies eligible for input tax credits, "
    "is reversed as if supply of such goods is an exempt supply and attracts provisions of "
    "sub-section (2) of section 17 of the Central Goods and Services Tax Act, 2017 (12 of 2017) "
    "and the rules made thereunder."
)

RATE_TO_SCHEDULE_MAP = {
    "2.5%": "Schedule I – 2.5%",
    "9%": "Schedule II – 9%",
    "20%": "Schedule III – 20%",
    "1.5%": "Schedule IV – 1.5%",
    "0.125%": "Schedule V – 0.125%",
    "0.75%": "Schedule VI – 0.75%",
    "14%": "Schedule VII – 14%",
}


def clean_text(val: Any) -> str:
    """Normalize whitespace and strip text."""
    if val is None:
        return ""
    return re.sub(r"\s+", " ", str(val)).strip()


def normalize_digits(code_str: str) -> list[str]:
    """Extract individual normalized HSN codes (2, 4, 6, 8 digits) from classification text."""
    if not code_str:
        return []
    unified = code_str.replace("\n", ", ")
    tokens = [t.strip() for t in re.split(r"[,;]+", unified) if t.strip()]
    results: set[str] = set()
    for tok in tokens:
        digits = re.sub(r"\D", "", tok)
        if len(digits) >= 2:
            results.add(digits)
            if len(digits) in (6, 7, 8):
                results.add(digits[:4])
                results.add(digits[:2])
            elif len(digits) == 4:
                results.add(digits[:2])
    # Also support 'Chapter XX' mentions
    ch_matches = re.findall(r"Chapter\s+(\d+)", code_str, re.I)
    for ch in ch_matches:
        results.add(ch.zfill(2))
    return sorted(results, key=lambda x: (len(x), x))


def resolve_pdf_path(pdf_path: str | Path | None = None) -> Path:
    """Resolve active PDF path with fallback."""
    if pdf_path:
        p = Path(pdf_path)
        if p.exists():
            return p
    if DEFAULT_PDF_PATH.exists():
        return DEFAULT_PDF_PATH
    if FALLBACK_PDF_PATH.exists():
        return FALLBACK_PDF_PATH
    raise FileNotFoundError(f"GST rates2025.pdf not found at {DEFAULT_PDF_PATH} or {FALLBACK_PDF_PATH}")


def parse_pdf_rates(pdf_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Parse complete GST rates2025.pdf into structured row dictionaries."""
    resolved_path = resolve_pdf_path(pdf_path)
    doc = pymupdf.open(str(resolved_path))

    records: list[dict[str, Any]] = []

    current_section = "goods"
    current_notif = "09/2025-Central Tax (Rate)"
    current_notif_date = "17th September, 2025"
    current_effective_date = "22.09.2025"
    current_schedule = "Schedule I – 2.5%"

    for pno in range(len(doc)):
        page = doc[pno]
        page_num = pno + 1
        blocks = page.get_text("blocks")
        tabs = page.find_tables().tables

        # Process each table on the page in vertical order
        for t_idx, tab in enumerate(tabs):
            t_y0 = tab.bbox[1]
            prev_y1 = tabs[t_idx - 1].bbox[3] if t_idx > 0 else 0

            # Scan text blocks between previous table and this table for section/schedule transitions
            for b in blocks:
                b_y0 = b[1]
                b_text = b[4].strip()
                if not b_text:
                    continue

                if prev_y1 <= b_y0 < t_y0:
                    if "1. CGST rates on goods as on 22.09.2025" in b_text:
                        current_section = "goods"
                        current_notif = "09/2025-Central Tax (Rate)"
                        current_notif_date = "17th September, 2025"
                        current_effective_date = "22.09.2025"
                        current_schedule = "Schedule I – 2.5%"
                    elif "2. Exempted Goods as on 22.09.2025" in b_text:
                        current_section = "exempted_goods"
                        current_notif = "10/2025-Central Tax (Rate)"
                        current_notif_date = "17th September, 2025"
                        current_effective_date = "22.09.2025"
                        current_schedule = "Exempted Goods Schedule"
                    elif "3. CGST Rates on Goods as on 22.09.2025" in b_text:
                        current_section = "goods"
                        current_notif = "02/2022-Central Tax (Rate)"
                        current_notif_date = "31st March, 2022"
                        current_effective_date = "01.04.2022"
                        current_schedule = "Notification 02/2022 Table"
                    elif "4. CGST Rates on Goods as on 22.09.2025" in b_text:
                        current_section = "goods"
                        current_notif = "14/2025-Central Tax (Rate)"
                        current_notif_date = "17th September, 2025"
                        current_effective_date = "22.09.2025"
                        current_schedule = "Notification 14/2025 Schedule"
                    elif "5. Effective Compensation Cess as on 22.09.2025" in b_text:
                        current_section = "cess"
                        current_notif = "1/2017-Compensation Cess (Rate)"
                        current_notif_date = "28th June, 2017"
                        current_effective_date = "22.09.2025"
                        current_schedule = "Compensation Cess Schedule"
                    elif "Annexure-I" in b_text and "drugs" in b_text.lower():
                        current_section = "exempted_goods"
                        current_notif = "10/2025-Central Tax (Rate)"
                        current_notif_date = "17th September, 2025"
                        current_effective_date = "22.09.2025"
                        current_schedule = "Notification 10/2025 Annexure-I (Drugs & Medicines)"
                    elif "Annexure-II" in b_text and "musical" in b_text.lower():
                        current_section = "exempted_goods"
                        current_notif = "10/2025-Central Tax (Rate)"
                        current_notif_date = "17th September, 2025"
                        current_effective_date = "22.09.2025"
                        current_schedule = "Notification 10/2025 Annexure-II (Handmade Musical Instruments)"
                    elif "List 1 [See S. No. 478" in b_text:
                        current_section = "goods"
                        current_schedule = "Schedule I – List 1 (Assistive Devices for Disabled)"
                    elif re.match(r"^Schedule\s+[IVXLCDM]+\s*[\-–]", b_text, re.I):
                        m = re.match(r"^(Schedule\s+[IVXLCDM]+\s*[\-–]\s*[^\n]+)", b_text, re.I)
                        if m:
                            current_schedule = m.group(1).strip()

            # Skip disclaimer on page 1
            if page_num == 1 and t_idx == 0:
                continue
            # Skip Annexure condition table header on page 86
            if page_num == 86 and t_idx == 0:
                continue

            raw_rows = tab.extract()
            for r_idx, r in enumerate(raw_rows):
                non_empty = [c for c in r if c is not None and str(c).strip() != ""]
                if not non_empty:
                    continue

                first_val = clean_text(non_empty[0]).lower()
                second_val = clean_text(non_empty[1]).lower() if len(non_empty) > 1 else ""

                # Skip table header rows
                if any(h in first_val for h in ["s. no.", "sl. no.", "s no.", "(1)", "condition no."]):
                    continue
                if any(h in second_val for h in ["chapter", "tariff item", "(2)"]):
                    continue

                # Handle description continuation rows (1 column that isn't a serial number)
                if len(non_empty) == 1 and records and not re.match(r"^\d+[\.\)]?$", non_empty[0].strip()):
                    records[-1]["description"] += " " + clean_text(non_empty[0])
                    continue

                # Standard 4+ column rate rows
                if len(non_empty) >= 3:
                    s_no = clean_text(non_empty[0])
                    hsn = clean_text(non_empty[1])
                    desc = clean_text(non_empty[2])
                    rate_val = clean_text(non_empty[3]) if len(non_empty) >= 4 else ""
                    cond_val = clean_text(non_empty[4]) if len(non_empty) >= 5 else ""

                    # Deduced schedule from rate in Section 1 (Notification 09/2025)
                    sched = current_schedule
                    if current_section == "goods" and current_notif.startswith("09/2025"):
                        for r_key, s_name in RATE_TO_SCHEDULE_MAP.items():
                            if r_key in rate_val:
                                sched = s_name
                                break

                    cond_text = NOTIF_02_2022_COND_1_TEXT if cond_val == "1" else None

                    rate_clean = rate_val.replace("\n", " ").strip()
                    if not rate_clean:
                        m_sch = re.search(r"([\d\.]+)%", sched)
                        if m_sch:
                            rate_clean = f"{m_sch.group(1)}%"
                        elif "schedule i" in sched.lower():
                            rate_clean = "2.5%"
                        elif "schedule ii" in sched.lower():
                            rate_clean = "9%"
                        elif "schedule iii" in sched.lower():
                            rate_clean = "20%"
                        elif "exempted" in sched.lower() or "annexure" in sched.lower():
                            rate_clean = "Nil"

                    cgst_pct = None
                    sgst_pct = None
                    igst_pct = None
                    cess_val = None

                    if current_section == "cess":
                        cess_val = rate_clean
                        formatted_rate = f"{rate_clean} Cess"
                    elif rate_clean.lower() in ("nil", "exempt", "0%"):
                        cgst_pct = 0.0
                        sgst_pct = 0.0
                        igst_pct = 0.0
                        formatted_rate = "Nil / Exempt (0% GST)"
                    elif "%" in rate_clean:
                        m_pct = re.search(r"([\d\.]+)\s*%", rate_clean)
                        if m_pct:
                            cgst_pct = float(m_pct.group(1))
                            sgst_pct = cgst_pct
                            igst_pct = round(cgst_pct * 2, 3)
                            formatted_rate = f"{igst_pct:g}% IGST ({cgst_pct:g}% CGST + {sgst_pct:g}% SGST)"
                        else:
                            formatted_rate = rate_clean
                    else:
                        formatted_rate = rate_clean

                    norm_codes = normalize_digits(hsn)
                    if not norm_codes and any(w in desc.lower() for w in ["heading", "chapter"]):
                        norm_codes = normalize_digits(desc)

                    records.append(
                        {
                            "category": current_section,
                            "notification_number": current_notif,
                            "notification_date": current_notif_date,
                            "effective_date": current_effective_date,
                            "schedule": sched,
                            "serial_number": s_no,
                            "hsn_code": hsn,
                            "normalized_hsn_codes": norm_codes,
                            "description": desc,
                            "rate": rate_clean,
                            "cgst_rate_pct": cgst_pct,
                            "sgst_utgst_rate_pct": sgst_pct,
                            "igst_rate_pct": igst_pct,
                            "formatted_rate": formatted_rate,
                            "compensation_cess": cess_val,
                            "condition_number": cond_val or None,
                            "condition_text": cond_text,
                            "source_page": page_num,
                            "source_file": "GST rates2025.pdf",
                        }
                    )
                elif len(non_empty) == 2:
                    # 2-column entries like Annexure-I drugs or Annexure-II musical instruments
                    s_no = clean_text(non_empty[0])
                    desc = clean_text(non_empty[1])
                    is_drugs = "drugs" in current_schedule.lower()
                    is_music = "musical" in current_schedule.lower()
                    hsn = "3004" if is_drugs else ("92" if is_music else "Any")
                    norm = ["3004", "30"] if is_drugs else (["92"] if is_music else [])

                    records.append(
                        {
                            "category": current_section,
                            "notification_number": current_notif,
                            "notification_date": current_notif_date,
                            "effective_date": current_effective_date,
                            "schedule": current_schedule,
                            "serial_number": s_no,
                            "hsn_code": hsn,
                            "normalized_hsn_codes": norm,
                            "description": desc,
                            "rate": "Nil",
                            "cgst_rate_pct": 0.0,
                            "sgst_utgst_rate_pct": 0.0,
                            "igst_rate_pct": 0.0,
                            "formatted_rate": "Nil / Exempt (0% GST)",
                            "compensation_cess": None,
                            "condition_number": None,
                            "condition_text": None,
                            "source_page": page_num,
                            "source_file": "GST rates2025.pdf",
                        }
                    )

    return records


def validate_extracted_rates(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate extracted GST rate records for completeness, correctness, and required test cases."""
    errors: list[str] = []
    total_count = len(records)

    if total_count < 1400:
        errors.append(f"Row count too low: expected at least 1400 rows, got {total_count}")

    # Required fields check
    missing_desc = 0
    missing_rate = 0
    missing_page = 0
    code_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    schedule_counts: dict[str, int] = {}

    for idx, r in enumerate(records):
        if not r.get("description"):
            missing_desc += 1
        if not r.get("rate"):
            missing_rate += 1
        if not r.get("source_page"):
            missing_page += 1

        cat = r.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

        sch = r.get("schedule", "unknown")
        schedule_counts[sch] = schedule_counts.get(sch, 0) + 1

        for c in r.get("normalized_hsn_codes", []):
            code_counts[c] = code_counts.get(c, 0) + 1

    if missing_desc > 0:
        errors.append(f"{missing_desc} records missing description")
    if missing_rate > 0:
        errors.append(f"{missing_rate} records missing rate")
    if missing_page > 0:
        errors.append(f"{missing_page} records missing source page")

    # Verify required test examples exist in the records
    test_examples_status: dict[str, Any] = {}

    # 1. paneer
    paneer_matches = [
        r for r in records if "paneer" in r["description"].lower()
    ]
    test_examples_status["paneer"] = {
        "found": len(paneer_matches) > 0,
        "count": len(paneer_matches),
        "rates": [m["formatted_rate"] for m in paneer_matches],
        "hsn_codes": [m["hsn_code"] for m in paneer_matches],
    }
    if not paneer_matches:
        errors.append("Test example 'paneer' not found in extracted records")

    # 2. 0406 (multiple entries: 5% cheese vs 0% chena/paneer)
    code_0406_matches = [
        r for r in records if "0406" in r.get("normalized_hsn_codes", []) or "0406" in r.get("hsn_code", "")
    ]
    test_examples_status["0406"] = {
        "found": len(code_0406_matches) >= 2,
        "count": len(code_0406_matches),
        "rates": [m["formatted_rate"] for m in code_0406_matches],
        "schedules": [m["schedule"] for m in code_0406_matches],
    }
    if len(code_0406_matches) < 2:
        errors.append(f"Expected multiple entries for HSN 0406, found {len(code_0406_matches)}")

    # 3. toothpaste
    toothpaste_matches = [
        r for r in records if "toothpaste" in r["description"].lower()
    ]
    test_examples_status["toothpaste"] = {
        "found": len(toothpaste_matches) > 0,
        "count": len(toothpaste_matches),
        "rates": [m["formatted_rate"] for m in toothpaste_matches],
    }
    if not toothpaste_matches:
        errors.append("Test example 'toothpaste' not found in extracted records")

    # 4. medical oxygen
    oxygen_matches = [
        r for r in records if "oxygen" in r["description"].lower() and "medical" in r["description"].lower()
    ]
    test_examples_status["medical_oxygen"] = {
        "found": len(oxygen_matches) > 0,
        "count": len(oxygen_matches),
        "rates": [m["formatted_rate"] for m in oxygen_matches],
        "hsn_codes": [m["hsn_code"] for m in oxygen_matches],
    }
    if not oxygen_matches:
        errors.append("Test example 'medical oxygen' not found in extracted records")

    # 5. fly ash bricks (multiple entries: 3% conditional vs 6% standard)
    fly_ash_matches = [
        r for r in records if "fly ash bricks" in r["description"].lower()
    ]
    test_examples_status["fly_ash_bricks"] = {
        "found": len(fly_ash_matches) >= 2,
        "count": len(fly_ash_matches),
        "rates": [m["formatted_rate"] for m in fly_ash_matches],
        "has_condition": any(m.get("condition_number") == "1" for m in fly_ash_matches),
    }
    if len(fly_ash_matches) < 2:
        errors.append(f"Expected multiple entries for fly ash bricks, found {len(fly_ash_matches)}")

    # Verify conditions
    conditional_entries = [r for r in records if r.get("condition_number") is not None]
    has_full_condition_text = any(
        r.get("condition_text") and "credit of input tax" in r["condition_text"]
        for r in conditional_entries
    )
    if not has_full_condition_text:
        errors.append("Condition text for Notification 02/2022 not properly populated")

    return {
        "is_valid": len(errors) == 0,
        "total_records": total_count,
        "categories": category_counts,
        "schedules": schedule_counts,
        "conditional_records": len(conditional_entries),
        "test_examples": test_examples_status,
        "errors": errors,
    }
