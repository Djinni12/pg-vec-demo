"""Structured CSV-based GST rate retriever.

Loads and searches gst_rates.csv directly without embeddings or pgvector.
Supports exact HSN code lookups and description-based keyword/phrase matching.
Preserves all original legal columns, exact rate strings, conditions, footnotes, and amendment notes.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv()

DEFAULT_CSV_PATH = Path("data/gst/gst_rates.csv")
DEFAULT_APPENDED_CSV_PATH = Path("data/gst/gst_rates_appended.csv")
DEFAULT_DATABASE_URL = "dbname=hybrid_rag user=postgres password=postgres host=localhost port=5432"
EXPECTED_ROW_COUNT = 1663


def database_url() -> str:
    """Retrieve database connection URL."""
    return os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL
REQUIRED_FIELDS = [
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

YEAR_EXCLUSIONS = {str(y) for y in range(1990, 2035)}

# Module-level cache for parsed records and HSN code index
_CACHED_PATH: Path | None = None
_CACHED_RECORDS: list[dict[str, Any]] | None = None
_CACHED_HSN_INDEX: dict[str, list[int]] | None = None


def resolve_rates_csv_path(csv_path: str | Path | None = None) -> Path:
    """Resolve the location of gst_rates.csv with sensible fallbacks."""
    if csv_path:
        p = Path(csv_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified CSV path does not exist: {csv_path}")

    candidates = [
        DEFAULT_CSV_PATH,
        Path("gst_rates.csv"),
        Path(__file__).resolve().parents[2] / "data" / "gst" / "gst_rates.csv",
        Path(__file__).resolve().parents[2] / "gst_rates.csv",
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    raise FileNotFoundError(
        f"gst_rates.csv could not be located. Checked: {[str(c) for c in candidates]}"
    )


def normalize_hsn_codes(code_str: str) -> set[str]:
    """Extract individual normalized HSN codes (2, 4, 6, 8 digits) from classification text.

    Preserves leading zeros and generates prefix keys for hierarchical search.
    """
    if not code_str:
        return set()

    unified = code_str.replace("\n", ", ")
    tokens = [t.strip() for t in re.split(r"[,;/]|\bor\b", unified, flags=re.I) if t.strip()]
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
    for ch in re.findall(r"Chapter\s+(\d+)", code_str, re.I):
        results.add(ch.zfill(2))

    return results


def load_rates_csv(
    csv_path: str | Path | None = None,
    *,
    force_reload: bool = False,
    include_appended: bool = False,
) -> list[dict[str, Any]]:
    """Load and parse gst_rates.csv strictly, validating exact row count and schema.

    Uses escapechar='\\' to correctly parse escaped quotes (e.g. \\\"smart cards\\\").
    Never skips bad lines. Halts with ValueError if the row count is not at least 1,663.
    """
    global _CACHED_PATH, _CACHED_RECORDS, _CACHED_HSN_INDEX

    resolved = resolve_rates_csv_path(csv_path)
    if not force_reload and _CACHED_RECORDS is not None and _CACHED_PATH == resolved:
        return _CACHED_RECORDS

    with open(resolved, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, escapechar="\\")
        fieldnames = reader.fieldnames or []
        missing = [col for col in REQUIRED_FIELDS if col not in fieldnames]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}. Found: {fieldnames}")

        records: list[dict[str, Any]] = []
        for idx, row in enumerate(reader, 1):
            clean_row = {k: (v or "").strip() for k, v in row.items()}
            clean_row["_id"] = idx
            records.append(clean_row)

    if len(records) < EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Expected at least {EXPECTED_ROW_COUNT} rows in {resolved}, "
            f"but found {len(records)} rows. Halting to avoid incomplete rate retrieval."
        )

    # Optionally load appended web records if present
    if include_appended and csv_path is None and DEFAULT_APPENDED_CSV_PATH.exists():
        try:
            with open(DEFAULT_APPENDED_CSV_PATH, "r", encoding="utf-8") as af:
                areader = csv.DictReader(af, escapechar="\\")
                for arow in areader:
                    clean_arow = {k: (v or "").strip() for k, v in arow.items()}
                    clean_arow["_id"] = len(records) + 1
                    records.append(clean_arow)
        except Exception:
            pass

    # Build HSN index mapping normalized digit codes to list of row indices
    hsn_index: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        codes = normalize_hsn_codes(rec["hsn_code"])
        for c in codes:
            hsn_index.setdefault(c, []).append(i)

    _CACHED_PATH = resolved
    _CACHED_RECORDS = records
    _CACHED_HSN_INDEX = hsn_index
    return records


def extract_query_codes(query: str) -> list[str]:
    """Extract candidate 2-8 digit HSN/tariff codes from the user query."""
    if not query:
        return []

    # Clean text and replace spaces inside numbers like '0101 21 00'
    cleaned = re.sub(r"[\r\n\t]+", " ", query)
    cleaned = re.sub(r"\b(\d{4})\s+(\d{2})\s+(\d{2})\b", r"\1\2\3", cleaned)
    cleaned = re.sub(r"\b(\d{4})\s+(\d{2})\b", r"\1\2", cleaned)

    codes: list[str] = []

    # Priority 1: Explicit labels like HSN 8471, Heading 6815, Chapter 04, Code 0402
    for m in re.finditer(
        r"\b(?:hsn|tariff|heading|chapter|code)\s*[:#]?\s*(\d{2,8})\b",
        cleaned,
        re.IGNORECASE,
    ):
        code = m.group(1)
        if code not in codes:
            codes.append(code)

    # Priority 2: Standalone 4, 6, or 8 digit numbers (excluding year exclusions)
    for m in re.finditer(r"\b(\d{4,8})\b", cleaned):
        code = m.group(1)
        if code not in YEAR_EXCLUSIONS and code not in codes:
            codes.append(code)

    return codes


def extract_search_phrase(query: str) -> str:
    """Extract the core product or service search phrase by removing boilerplate stop words."""
    cleaned = query.strip()
    cleaned = re.sub(r"(\d+)(cc|ml|kg|l|gm)\b", r"\1 \2", cleaned, flags=re.IGNORECASE)
    patterns = [
        r"^(?:what\s+(?:is|are)\s+(?:the\s+)?|give\s+me\s+(?:the\s+)?|tell\s+me\s+(?:the\s+)?|find\s+(?:the\s+)?)?(?:applicable\s+)?(?:hsn|sac|tariff)?\s*(?:code|heading|number)?\s*(?:for|of|on)\s*",
        r"^give\s+me\s+(?:full\s+)?(?:gst\s+)?(?:rate\s+)?(?:details\s+)?(?:for|of|on)?\s*",
        r"^what\s+(?:is|are)\s+(?:the\s+)?(?:gst|tax|compensation\s+cess|cess|applicable)?\s*(?:rate|rates|slab|slabs)?\s*(?:of|on|for)?\s*",
        r"^how\s+much\s+(?:is\s+)?(?:gst|tax)\s*(?:on|for)?\s*",
        r"^tell\s+me\s+(?:the\s+)?(?:gst|tax)?\s*(?:rate|rates)?\s*(?:on|for)?\s*",
        r"^gst\s+(?:rate|rates|slab|slabs|percentage)?\s*(?:on|for|of)?\s*",
        r"^tax\s+(?:rate|rates|slab|slabs|percentage)?\s*(?:on|for|of)?\s*",
        r"^compensation\s+cess\s+(?:rate|rates|slab|slabs)?\s*(?:of|on|for)?\s*",
        r"^cess\s+(?:rate|rates|slab|slabs)?\s*(?:of|on|for)?\s*",
        r"^rate\s+(?:of|on|for)?\s*",
        r"\b(?:under\s+)?(?:hsn|heading|code)\s*[:#]?\s*\d+\b",
        r"\b(?:hsn|sac)\s*(?:code)?\s*(?:for|of|on)?\b",
        r"\bcode\s+(?:for|of|on)\b",
        r"\b(?:in\s+india|under\s+gst|please|can\s+you\s+tell|applicable)\b",
        r"[?.,!]",
    ]
    for pat in patterns:
        cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE)

    tokens = [t for t in cleaned.split() if len(t) > 1]
    return " ".join(tokens).strip()


def _stem(word: str) -> str:
    """Basic English stemming for plurals and inflections without external libraries."""
    w = word.lower()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 3:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 2:
        return w[:-1]
    return w


def _format_record(
    row: dict[str, Any], match_type: str = "exact_code", score: float = 1.0
) -> dict[str, Any]:
    """Format a raw CSV row into the standard structured rate response dict."""
    notif = row.get("notification_no", "")
    sched = row.get("schedule", "")
    rate_cat = row.get("rate_category", "")
    sec_heading = row.get("section_heading", "")
    source_rate = row.get("source_rate") or row.get("gst_rate", "")
    cgst_rate = row.get("cgst_rate") or None
    sgst_rate = row.get("sgst_rate") or None
    total_gst_rate = row.get("total_gst_rate") or None
    cess_rate = row.get("compensation_cess_rate") or None
    is_exempt = str(row.get("is_exempt", "")).lower() == "true"

    is_cess = rate_cat == "COMPENSATION_CESS" or "cess" in notif.lower() or "cess" in sched.lower()

    if rate_cat == "CGST":
        rate_type = "Central Tax (CGST)"
        if total_gst_rate and cgst_rate and sgst_rate:
            formatted_rate = f"Total GST: {total_gst_rate} (CGST: {cgst_rate}, SGST: {sgst_rate})"
        else:
            formatted_rate = f"Central Tax (CGST): {source_rate}"
    elif rate_cat == "EXEMPTION":
        rate_type = "Exempt"
        formatted_rate = "Nil / Exempt (0% GST)"
    elif rate_cat == "COMPENSATION_CESS":
        rate_type = "Compensation Cess"
        formatted_rate = f"{cess_rate or source_rate} (Compensation Cess)"
    elif rate_cat == "SPECIAL":
        rate_type = "Special / Conditional"
        formatted_rate = f"{source_rate} (Special Notification)"
    else:
        if "central tax" in notif.lower():
            rate_type = "Central Tax (CGST)"
        elif "integrated tax" in notif.lower():
            rate_type = "Integrated Tax (IGST)"
        elif is_cess:
            rate_type = "Compensation Cess"
        else:
            rate_type = "GST"
        formatted_rate = source_rate

    # Parse numeric percentages if available
    cgst_pct = None
    sgst_pct = None
    igst_pct = None
    if cgst_rate:
        m = re.search(r"([\d\.]+)", cgst_rate)
        if m:
            cgst_pct = float(m.group(1))
    if sgst_rate:
        m = re.search(r"([\d\.]+)", sgst_rate)
        if m:
            sgst_pct = float(m.group(1))
    if total_gst_rate:
        m = re.search(r"([\d\.]+)", total_gst_rate)
        if m:
            igst_pct = float(m.group(1))

    return {
        "id": row.get("_id"),
        "item_type": "cess" if is_cess else "goods",
        "category": "cess" if is_cess else "goods",
        "code": row.get("hsn_code", ""),
        "hsn_code": row.get("hsn_code", ""),
        "description": row.get("description", ""),
        "section_heading": sec_heading,
        "rate_category": rate_cat,
        "source_rate": source_rate,
        "cgst_rate": cgst_rate,
        "sgst_rate": sgst_rate,
        "total_gst_rate": total_gst_rate,
        "compensation_cess_rate": cess_rate,
        "is_exempt": is_exempt,
        "gst_rate": source_rate,
        "rate": source_rate,
        "formatted_rate": formatted_rate,
        "rate_type": rate_type,
        "schedule": sched,
        "serial_no": row.get("serial_no", ""),
        "serial_number": row.get("serial_no", ""),
        "notification_no": notif,
        "notification_number": notif,
        "notification_date": row.get("notification_date", ""),
        "rate_as_on_date": row.get("rate_as_on_date", "22.09.2025"),
        "effective_date": row.get("effective_date") or None,
        "condition": row.get("condition", ""),
        "footnote": row.get("footnote", ""),
        "amendment_note": row.get("amendment_note", ""),
        "raw_text": row.get("raw_text", ""),
        "source_page": row.get("source_page"),
        "source_file": row.get("source_file", "GST rates2025.pdf"),
        "source_reference": f"{row.get('source_file', 'GST rates2025.pdf')} ({notif})" if notif else row.get("source_file", "GST rates2025.pdf"),
        "cgst_rate_pct": cgst_pct,
        "sgst_utgst_rate_pct": sgst_pct,
        "igst_rate_pct": igst_pct,
        "compensation_cess": cess_rate if is_cess else None,
        "condition_number": row.get("condition_number") or (
            re.search(r"\bCondition\s+(?:No\.?\s*)?(\d+)\b", f"{row.get('amendment_note', '')} {row.get('raw_text', '')}", re.I).group(1)
            if re.search(r"\bCondition\s+(?:No\.?\s*)?(\d+)\b", f"{row.get('amendment_note', '')} {row.get('raw_text', '')}", re.I)
            else None
        ),
        "condition_text": row.get("condition", ""),
        "match_type": match_type,
        "score": score,
    }


def _format_db_record(r: tuple, match_type: str = "database", score: float = 1.0) -> dict[str, Any]:
    """Format a row from gst_rates_2025 table into standard rate dict."""
    rid = r[0]
    category = r[1] or "goods"
    notif = r[2] or ""
    notif_date = r[3] or ""
    eff_date = r[4] or None
    sched = r[5] or ""
    serial_no = r[6] or ""
    hsn_code = r[7] or ""
    desc = r[9] or ""
    rate_str = r[10] or ""
    cgst_pct = float(r[11]) if r[11] is not None else None
    sgst_pct = float(r[12]) if r[12] is not None else None
    igst_pct = float(r[13]) if r[13] is not None else None
    formatted_rate = r[14] or rate_str
    comp_cess = r[15]
    cond_num = r[16]
    cond_text = r[17] or ""
    page = r[18]
    source_file = r[19] or "gst_rates_2025"

    is_cess = "cess" in category.lower() or bool(comp_cess)
    is_exempt = "nil" in rate_str.lower() or "exempt" in rate_str.lower()

    cgst_rate = f"{cgst_pct}%" if cgst_pct is not None else ""
    sgst_rate = f"{sgst_pct}%" if sgst_pct is not None else ""
    total_rate = f"{igst_pct}%" if igst_pct is not None else rate_str

    return {
        "id": rid,
        "item_type": "cess" if is_cess else category.lower(),
        "category": "cess" if is_cess else category.lower(),
        "code": hsn_code,
        "hsn_code": hsn_code,
        "description": desc,
        "section_heading": f"GST rates - {category}",
        "rate_category": category,
        "source_rate": rate_str,
        "cgst_rate": cgst_rate,
        "sgst_rate": sgst_rate,
        "total_gst_rate": total_rate,
        "compensation_cess_rate": comp_cess or "",
        "is_exempt": str(is_exempt).lower(),
        "gst_rate": rate_str,
        "rate": rate_str,
        "formatted_rate": formatted_rate,
        "rate_type": "Compensation Cess" if is_cess else "GST",
        "schedule": sched,
        "serial_no": serial_no,
        "serial_number": serial_no,
        "notification_no": notif,
        "notification_number": notif,
        "notification_date": notif_date,
        "rate_as_on_date": "22.09.2025",
        "effective_date": eff_date,
        "condition": cond_text,
        "footnote": "",
        "amendment_note": "",
        "raw_text": f"{serial_no} | {hsn_code} | {desc} | {rate_str}",
        "source_page": page,
        "source_file": source_file,
        "source_reference": f"{source_file} ({notif})" if notif else source_file,
        "cgst_rate_pct": cgst_pct,
        "sgst_utgst_rate_pct": sgst_pct,
        "igst_rate_pct": igst_pct,
        "compensation_cess": comp_cess,
        "condition_number": cond_num,
        "condition_text": cond_text,
        "match_type": match_type,
        "score": score,
    }


def search_rates_database(
    query: str,
    db_url: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Search gst_rates_2025 in PostgreSQL directly."""
    try:
        import psycopg
    except ImportError:
        return []

    url = db_url or database_url()
    codes = extract_query_codes(query)
    phrase = extract_search_phrase(query)

    results: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    try:
        with psycopg.connect(url, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                # 1. Exact / normalized HSN code matches
                for code in codes:
                    cur.execute(
                        """
                        SELECT id, category, notification_number, notification_date, effective_date,
                               schedule, serial_number, hsn_code, normalized_hsn_codes, description,
                               rate, cgst_rate_pct, sgst_utgst_rate_pct, igst_rate_pct, formatted_rate,
                               compensation_cess, condition_number, condition_text, source_page, source_file
                        FROM gst_rates_2025
                        WHERE %s = ANY(normalized_hsn_codes) OR hsn_code ILIKE %s
                        LIMIT %s;
                        """,
                        (code, f"%{code}%", limit),
                    )
                    for row in cur.fetchall():
                        rid = row[0]
                        if rid not in seen_ids:
                            seen_ids.add(rid)
                            results.append(_format_db_record(row, match_type="exact_code", score=1.0))

                # 2. Text / description matches
                if len(results) < limit and phrase:
                    cur.execute(
                        """
                        SELECT id, category, notification_number, notification_date, effective_date,
                               schedule, serial_number, hsn_code, normalized_hsn_codes, description,
                               rate, cgst_rate_pct, sgst_utgst_rate_pct, igst_rate_pct, formatted_rate,
                               compensation_cess, condition_number, condition_text, source_page, source_file
                        FROM gst_rates_2025
                        WHERE description ILIKE %s
                        LIMIT %s;
                        """,
                        (f"%{phrase}%", limit - len(results)),
                    )
                    for row in cur.fetchall():
                        rid = row[0]
                        if rid not in seen_ids:
                            seen_ids.add(rid)
                            results.append(_format_db_record(row, match_type="description_search", score=0.85))
    except Exception:
        pass

    return results


def append_rate_record_to_csv_and_cache(
    record: dict[str, Any],
    csv_path: str | Path | None = None,
) -> bool:
    """Appends a new structured rate record to the in-memory cache and CSV on disk.

    Deduplicates against existing cache by matching (hsn_code, rate, description).
    Returns True if appended, False if duplicate.
    """
    global _CACHED_RECORDS, _CACHED_HSN_INDEX, _CACHED_PATH

    # Ensure cache is loaded
    rows = load_rates_csv(csv_path, include_appended=True)

    hsn = str(record.get("hsn_code") or record.get("code") or "").strip()
    rate = str(record.get("gst_rate") or record.get("rate") or record.get("source_rate") or "").strip()
    desc = str(record.get("description") or "").strip()

    # Deduplication check: same HSN and rate means record is already stored
    for r in rows:
        r_hsn = (r.get("hsn_code") or "").strip()
        r_rate = (r.get("gst_rate") or r.get("source_rate") or "").strip()
        r_desc = (r.get("description") or "").strip().lower()
        if r_hsn == hsn and r_rate == rate:
            return False
        if r_desc and desc and r_desc == desc.lower() and r_rate == rate:
            return False

    # Format row conforming to REQUIRED_FIELDS
    new_idx = len(rows) + 1
    clean_row = {field: str(record.get(field, "") or "").strip() for field in REQUIRED_FIELDS}
    clean_row["_id"] = new_idx
    if not clean_row["hsn_code"]:
        clean_row["hsn_code"] = hsn
    if not clean_row["gst_rate"]:
        clean_row["gst_rate"] = rate
    if not clean_row["source_rate"]:
        clean_row["source_rate"] = rate
    if not clean_row["description"]:
        clean_row["description"] = desc
    if not clean_row["source_file"]:
        clean_row["source_file"] = str(record.get("source_file", "Web Ingestion"))
    if not clean_row["source_page"]:
        clean_row["source_page"] = "1"
    if not clean_row["rate_as_on_date"]:
        clean_row["rate_as_on_date"] = "22.09.2025"

    # 1. Update in-memory cache immediately
    rows.append(clean_row)
    if _CACHED_HSN_INDEX is not None:
        for c in normalize_hsn_codes(clean_row["hsn_code"]):
            _CACHED_HSN_INDEX.setdefault(c, []).append(len(rows) - 1)

    # 2. Append to CSV on disk safely (uses separate appended file to preserve baseline CSV)
    try:
        target_path = Path(csv_path) if csv_path else DEFAULT_APPENDED_CSV_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = target_path.exists() and target_path.stat().st_size > 0

        with open(target_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar="\\")
            if not file_exists:
                writer.writerow(REQUIRED_FIELDS)
            row_vals = [clean_row.get(col, "") for col in REQUIRED_FIELDS]
            writer.writerow(row_vals)
    except Exception:
        pass

    return True


def exact_code_lookup(
    code: str,
    limit: int = 10,
    rows: list[dict[str, Any]] | None = None,
    csv_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Look up exact HSN code, returning matching candidate rows preserving leading zeros."""
    if rows is None:
        rows = load_rates_csv(csv_path)

    clean_code = re.sub(r"\D", "", code.strip())
    if not clean_code:
        return []

    matched_indices: list[int] = []
    if _CACHED_HSN_INDEX and clean_code in _CACHED_HSN_INDEX:
        matched_indices = _CACHED_HSN_INDEX[clean_code]
    else:
        for idx, r in enumerate(rows):
            norm = normalize_hsn_codes(r["hsn_code"])
            if clean_code in norm or clean_code == r["hsn_code"].replace(" ", ""):
                matched_indices.append(idx)

    results: list[dict[str, Any]] = []
    for idx in matched_indices[:limit]:
        results.append(_format_record(rows[idx], match_type="exact_code", score=1.0))

    return results


def text_rate_search(
    phrase: str,
    limit: int = 5,
    rows: list[dict[str, Any]] | None = None,
    csv_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Search goods descriptions using word matching, phrase scoring, and legal exclusion awareness."""
    if rows is None:
        rows = load_rates_csv(csv_path)

    clean_phrase = phrase.strip().lower()
    if not clean_phrase:
        return []

    words = [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", clean_phrase) if len(w) > 1]
    if not words:
        return []

    stems = [_stem(w) for w in words]
    phrase_text = " ".join(words)
    container_words = {"can", "cans", "tank", "tanks", "box", "boxes", "drum", "drums", "container", "containers", "machinery", "machine"}
    user_asked_container = any(w in container_words or _stem(w) in container_words for w in words)

    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        desc = row["description"].lower()
        desc_words = re.findall(r"[a-zA-Z0-9]+", desc)
        if not desc_words:
            continue
        desc_stems = [_stem(w) for w in desc_words]

        # Find matching word positions
        matched_indices = []
        for i, (dw, ds) in enumerate(zip(desc_words, desc_stems)):
            if dw in words or ds in stems or any(dw.startswith(s) for s in stems):
                matched_indices.append(i)

        if not matched_indices:
            continue

        matched_stems = {desc_stems[i] for i in matched_indices}
        ratio = len([s for s in stems if s in matched_stems]) / len(stems)

        exact_phrase = phrase_text in desc
        if len(words) <= 2 and ratio < 1.0 and not exact_phrase:
            continue
        if len(words) > 2 and ratio < 0.6 and not exact_phrase:
            continue

        score = 0.0
        # Boost for exact phrase match
        if exact_phrase:
            score += 5.0

        # Boost for match starting at or near beginning of description
        first_pos = matched_indices[0]
        if first_pos == 0:
            score += 4.0
        elif first_pos < 4:
            score += 2.0

        # Penalty if match only occurs in 'other than' clause
        other_than_pos = desc.find("other than")
        if other_than_pos != -1:
            all_in_other_than = all(
                desc.find(desc_words[i], other_than_pos) != -1 for i in matched_indices
            )
            if all_in_other_than:
                score -= 3.0

        # Penalty for container/packaging items when user did not ask for containers
        if not user_asked_container and any(desc.find(cw) != -1 for cw in container_words):
            score -= 2.0

        # Match coverage and frequency
        score += ratio * 3.0
        score += min(len(matched_indices) * 0.5, 2.0)

        # Slight length penalty to favor specific items over long catch-alls
        score -= min(len(desc_words) * 0.01, 1.0)

        scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, row in scored[:limit]:
        results.append(_format_record(row, match_type="description_search", score=round(max(score, 0.1), 4)))

    return results


def retrieve_rates(
    query: str,
    *,
    csv_path: str | Path | None = None,
    db_url: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve structured GST rate records from PostgreSQL gst_rates_2025 and gst_rates.csv.

    First extracts HSN/tariff codes and checks database + exact CSV code lookup.
    If limit is not reached, runs natural-language text search on description.
    """
    if not query or not query.strip():
        return []

    results: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()

    # 1. Local CSV & in-memory cache (primary search with tuned keyword/phrase scoring)
    rows = load_rates_csv(csv_path, include_appended=True)
    codes = extract_query_codes(query)
    phrase = extract_search_phrase(query)

    # 1a. Exact code lookup for all extracted codes
    for code in codes:
        code_hits = exact_code_lookup(code, limit=limit, rows=rows)
        for hit in code_hits:
            key = (
                str(hit.get("hsn_code") or "").strip(),
                str(hit.get("rate") or "").strip(),
                str(hit.get("description") or "")[:30].strip().lower(),
            )
            if key not in seen_keys:
                seen_keys.add(key)
                results.append(hit)

    # 1b. Text description matching
    if len(results) < limit and phrase:
        text_hits = text_rate_search(phrase, limit=limit - len(results) + 5, rows=rows)
        for hit in text_hits:
            key = (
                str(hit.get("hsn_code") or "").strip(),
                str(hit.get("rate") or "").strip(),
                str(hit.get("description") or "")[:30].strip().lower(),
            )
            if key not in seen_keys:
                seen_keys.add(key)
                results.append(hit)
            if len(results) >= limit:
                break

    # 2. Query PostgreSQL gst_rates_2025 to supplement if limit not reached
    target_db_url = db_url or (database_url() if len(results) < limit else None)
    if len(results) < limit and target_db_url:
        db_hits = search_rates_database(query, db_url=target_db_url, limit=limit - len(results))
        for hit in db_hits:
            key = (
                str(hit.get("hsn_code") or "").strip(),
                str(hit.get("rate") or "").strip(),
                str(hit.get("description") or "")[:30].strip().lower(),
            )
            if key not in seen_keys:
                seen_keys.add(key)
                results.append(hit)
            if len(results) >= limit:
                break

    return results[:limit]
