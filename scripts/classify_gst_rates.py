"""Classify GST rates dataset by section heading and derive safe GST rates.

Implements section identification, rate_category classification, and strict rules:
- Ordinary CGST rows: derive SGST = CGST and Total GST = CGST + SGST (Total GST = source_rate * 2)
- EXEMPTION rows: source_rate preserved (Nil), cgst=0%, sgst=0%, total_gst=0%, is_exempt=true
- COMPENSATION_CESS rows: compensation_cess_rate=source_rate, no doubling, cgst/sgst/total=null
- SPECIAL / UNKNOWN rows: preserve source_rate, no calculation, cgst/sgst/total=null
"""

import csv
from pathlib import Path
import re
from typing import Any
import pymupdf

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "data" / "gst" / "GST rates2025.pdf"
CSV_PATH = BASE_DIR / "data" / "gst" / "gst_rates.csv"
BAK_PATH = BASE_DIR / "data" / "gst" / "gst_rates.csv.bak"

FIELDNAMES = [
    "source_file",
    "source_page",
    "section_heading",
    "rate_category",
    "notification_no",
    "notification_date",
    "rate_as_on_date",
    "schedule",
    "serial_no",
    "hsn_code",
    "description",
    "source_rate",
    "cgst_rate",
    "sgst_rate",
    "total_gst_rate",
    "compensation_cess_rate",
    "is_exempt",
    "gst_rate",
    "condition",
    "footnote",
    "amendment_note",
    "raw_text",
]


def load_pdf_page_texts(pdf_path: Path) -> list[str]:
    """Load extracted text for each page from the PDF."""
    doc = pymupdf.open(str(pdf_path))
    return [doc[p].get_text() for p in range(len(doc))]


def get_row_page(
    r: dict[str, Any],
    page_texts: list[str],
    min_p: int,
    max_p: int,
    last_p: int,
) -> int:
    """Accurately identify the 1-indexed source page in the PDF for a rate row."""
    s_no = r["serial_no"].strip().rstrip(".")
    hsn = r["hsn_code"].strip().split(",")[0].strip()
    sched = r["schedule"].strip()
    desc = r["description"].strip()
    words = [
        w
        for w in re.findall(r"[A-Za-z]{4,}", desc)
        if w.lower() not in ("other", "than", "goods", "including", "specified")
    ][:2]

    if s_no.lower() == "explanation" or sched.lower() == "explanation":
        return max_p + 1

    # Search forward from last known page in section
    for p in range(max(min_p, last_p), max_p + 1):
        pt = page_texts[p]
        if re.search(r"\b" + re.escape(s_no) + r"[\.\)]", pt) or (hsn and hsn in pt):
            if not words or any(w.lower() in pt.lower() for w in words):
                return p + 1

    # Fallback to scanning whole section
    for p in range(min_p, max_p + 1):
        pt = page_texts[p]
        if re.search(r"\b" + re.escape(s_no) + r"[\.\)]", pt) and (hsn and hsn in pt):
            return p + 1

    return last_p + 1


def process_dataset(
    input_csv_path: Path = BAK_PATH,
    pdf_path: Path = PDF_PATH,
) -> list[dict[str, Any]]:
    """Classify and derive rates for all rows in the dataset."""
    page_texts = load_pdf_page_texts(pdf_path)

    # Use escapechar='\\' to ensure fields with escaped quotes like "smart cards" parse correctly
    with open(input_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, escapechar="\\")
        rows = list(reader)

    new_rows: list[dict[str, Any]] = []
    last_p = 0

    for r in rows:
        notif = r["notification_no"].strip()
        notif_date = r["notification_date"].strip()
        sched = r["schedule"].strip()
        s_no = r["serial_no"].strip()
        hsn = r["hsn_code"].strip()
        desc = r["description"].strip()
        s_rate = (r.get("source_rate") or r.get("gst_rate") or "").strip()
        cond = r.get("condition", "").strip()
        footnote = r.get("footnote", "").strip()
        amend = r.get("amendment_note", "").strip()
        raw = r.get("raw_text", "").strip()
        rate_as_on = "22.09.2025"

        is_expl = sched.lower() == "explanation" or s_rate in ("-", "")

        if "09/2025" in notif:
            sec_heading = "CGST rates on goods as on 22.09.2025"
            min_p, max_p = 0, 72
            if is_expl:
                rate_cat = "SPECIAL"
                cgst = None
                sgst = None
                total = None
                cess = None
                is_exempt = "false"
            else:
                rate_cat = "CGST"
                m_pct = re.search(r"([\d\.]+)\s*%", s_rate)
                if m_pct:
                    pct_val = float(m_pct.group(1))
                    cgst = f"{pct_val:g}%"
                    sgst = f"{pct_val:g}%"
                    tot_val = round(pct_val * 2, 4)
                    total = f"{tot_val:g}%"
                else:
                    cgst = None
                    sgst = None
                    total = None
                cess = None
                is_exempt = "false"

        elif "10/2025" in notif:
            sec_heading = "Exempted Goods as on 22.09.2025"
            min_p, max_p = 72, 84
            if is_expl:
                rate_cat = "SPECIAL"
                cgst = None
                sgst = None
                total = None
                cess = None
                is_exempt = "false"
            else:
                rate_cat = "EXEMPTION"
                cgst = "0%"
                sgst = "0%"
                total = "0%"
                cess = None
                is_exempt = "true"

        elif "02/2022" in notif:
            sec_heading = "CGST Rates on Goods as on 22.09.2025 [Notification No. 02/2022-Central Tax (Rate)]"
            min_p, max_p = 84, 85
            rate_cat = "SPECIAL"
            cgst = None
            sgst = None
            total = None
            cess = None
            is_exempt = "false"

        elif "14/2025" in notif:
            sec_heading = "CGST Rates on Goods as on 22.09.2025 [Notification No. 14/2025-Central Tax (Rate)]"
            min_p, max_p = 85, 86
            rate_cat = "SPECIAL"
            cgst = None
            sgst = None
            total = None
            cess = None
            is_exempt = "false"

        elif "1/2017" in notif:
            sec_heading = "Effective Compensation Cess as on 22.09.2025"
            min_p, max_p = 86, 93
            if is_expl:
                rate_cat = "SPECIAL"
                cgst = None
                sgst = None
                total = None
                cess = None
                is_exempt = "false"
            else:
                rate_cat = "COMPENSATION_CESS"
                cgst = None
                sgst = None
                total = None
                cess = s_rate
                is_exempt = "false"

        else:
            sec_heading = "Unknown Section"
            min_p, max_p = 0, 93
            rate_cat = "UNKNOWN"
            cgst = None
            sgst = None
            total = None
            cess = None
            is_exempt = "false"

        last_p = max(min_p, last_p)
        p_num = get_row_page(r, page_texts, min_p, max_p, last_p)
        last_p = p_num - 1

        new_rows.append(
            {
                "source_file": "GST rates2025.pdf",
                "source_page": str(p_num),
                "section_heading": sec_heading,
                "rate_category": rate_cat,
                "notification_no": notif,
                "notification_date": notif_date,
                "rate_as_on_date": rate_as_on,
                "schedule": sched,
                "serial_no": s_no,
                "hsn_code": hsn,
                "description": desc,
                "source_rate": s_rate,
                "cgst_rate": cgst or "",
                "sgst_rate": sgst or "",
                "total_gst_rate": total or "",
                "compensation_cess_rate": cess or "",
                "is_exempt": is_exempt,
                "gst_rate": s_rate,
                "condition": cond,
                "footnote": footnote,
                "amendment_note": amend,
                "raw_text": raw,
            }
        )

    return new_rows


def write_classified_csv(output_path: Path = CSV_PATH) -> list[dict[str, Any]]:
    """Process and overwrite output CSV with all classified fields."""
    rows = process_dataset(BAK_PATH if BAK_PATH.exists() else CSV_PATH, PDF_PATH)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    processed = write_classified_csv()
    print(f"Successfully wrote {len(processed)} classified rows to {CSV_PATH}")
