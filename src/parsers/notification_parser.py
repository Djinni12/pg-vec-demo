"""Parser for GST Central Tax (Rate) notification PDFs.

Extracts text, page-by-page metrics, document-level metadata,
statutory authority clauses, explicit document-wide effective dates,
clause-scoped effective dates, and provenance notes from official Gazette notifications.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import pymupdf


@dataclass
class NotificationPage:
    """Represents an extracted page from a notification PDF."""

    page_num: int
    char_count: int
    word_count: int
    text: str


@dataclass
class EffectiveDateScope:
    """Represents a specific effective date scope within a notification."""

    group_number: int
    raw_clause: str
    raw_date: str
    parsed_date: Optional[str]  # ISO YYYY-MM-DD
    scope_type: str = "CLAUSE_SCOPED"  # NUMBERED_GROUP, CLAUSE_SCOPED
    context_snippet: str = ""


@dataclass
class NotificationDocument:
    """Represents full extraction and document-level metadata for a notification."""

    file_path: str
    file_name: str
    file_size_bytes: int
    page_count: int
    total_char_count: int
    total_word_count: int
    avg_chars_per_page: float
    pages_with_zero_text: List[int] = field(default_factory=list)
    vector_drawings_count: int = 0
    raster_images_count: int = 0
    extraction_status: str = "SUCCESS"  # SUCCESS, FAILED_ZERO_TEXT, SUSPECT_LOW_TEXT
    error_message: Optional[str] = None

    notification_number: Optional[str] = None  # Canonical format: NN/YYYY-Central Tax (Rate)
    notification_number_raw: Optional[str] = None
    notification_date: Optional[str] = None  # ISO YYYY-MM-DD
    notification_date_raw: Optional[str] = None

    effective_date: Optional[str] = None  # ISO YYYY-MM-DD (ONLY if explicit document-wide commencement)
    effective_date_raw: Optional[str] = None
    effective_date_scopes: List[Dict[str, Any]] = field(default_factory=list)

    document_type: str = "UNKNOWN"  # RATE_SCHEDULE, EXEMPTION, AMENDMENT, SPECIAL, UNKNOWN
    legal_authority_raw: Optional[str] = None
    sections_invoked: List[str] = field(default_factory=list)
    target_notification: Optional[str] = None
    superseded_notification: Optional[str] = None
    file_number: Optional[str] = None
    signatory_raw: Optional[str] = None
    provenance_note: Optional[str] = None

    full_text: str = ""
    pages: List[NotificationPage] = field(default_factory=list)

    def to_dict(self, include_pages: bool = False, include_full_text: bool = False) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        data = asdict(self)
        if not include_pages:
            data.pop("pages", None)
        if not include_full_text:
            data.pop("full_text", None)
        return data


def parse_indian_gazette_date(d_str: Optional[str]) -> Optional[str]:
    """Parse Indian Gazette date formats into ISO YYYY-MM-DD.

    Examples:
        '17th September, 2025' -> '2025-09-17'
        '1st day of April, 2025' -> '2025-04-01'
        '16 January, 2025' -> '2025-01-16'
        '1st May, 2026' -> '2026-05-01'
    """
    if not d_str:
        return None
    cleaned = d_str.strip().rstrip(".")
    cleaned = re.sub(r"\bday\s+of\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(\d+)(?:st|nd|rd|th)\b", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().replace(",", "")
    for fmt in ("%d %B %Y", "%d %b %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def canonicalize_notification_number(notif_str: Optional[str]) -> Optional[str]:
    """Standardize notification numbers to NN/YYYY-Central Tax (Rate).

    Examples:
        'Notification No. 9/2025-Central Tax (Rate)' -> '09/2025-Central Tax (Rate)'
        'Notification No. 01/2026-Central Tax (Rate)' -> '01/2026-Central Tax (Rate)'
        '1/2017- Central Tax (Rate)' -> '01/2017-Central Tax (Rate)'
    """
    if not notif_str:
        return None
    m = re.search(r"([0-9]+)\s*/\s*([0-9]{4})\s*-\s*Central\s+Tax\s*\(\s*Rate\s*\)", notif_str, re.IGNORECASE)
    if m:
        num = int(m.group(1))
        year = m.group(2)
        return f"{num:02d}/{year}-Central Tax (Rate)"
    return notif_str.strip()


def extract_notification_number(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract canonical and raw notification number from the document preamble."""
    header = text[:3000]
    m = re.search(
        r"(?:Notification|NOTIFICATION)\s*(?:No\.?|NO\.?)\s*([0-9]+\s*/\s*[0-9]{4}\s*-\s*Central\s+Tax\s*\(\s*Rate\s*\))",
        header,
        re.IGNORECASE,
    )
    if m:
        raw = m.group(0).strip()
        canonical = canonicalize_notification_number(m.group(1))
        return canonical, raw
    return None, None


def extract_notification_date(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract notification issue date from the 'New Delhi, the ...' clause."""
    header = text[:3500]
    m = re.search(
        r"New\s+Delhi,\s*(?:dated\s+)?the\s+([0-9]{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+,?\s+[0-9]{4})",
        header,
        re.IGNORECASE,
    )
    if m:
        raw = m.group(1).strip().rstrip(".")
        iso = parse_indian_gazette_date(raw)
        return iso, raw
    return None, None


def extract_effective_dates(
    text: str, notification_date_iso: Optional[str]
) -> Tuple[Optional[str], Optional[str], List[Dict[str, Any]]]:
    """Extract explicit effective dates.

    RULE: document.effective_date should ONLY be populated when the notification
    explicitly provides a document-wide commencement date (e.g. 'This notification shall come into force...').
    If dates apply only to specific amendment items/clauses, document.effective_date = None,
    and the dates are preserved in effective_date_scopes for Stage 2 chunk-level association.

    Returns:
        (doc_effective_date_iso, doc_effective_date_raw, effective_date_scopes)
    """
    # 1. Document-wide commencement clause
    gen_match = re.search(
        r"(?:This\s+notification\s+shall\s+)?come\s+into\s+force\s+(?:with\s+effect\s+from|on|from)\s+(?:the\s+)?([0-9]{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?[A-Za-z]+,?\s+[0-9]{4})",
        text,
        re.IGNORECASE,
    )
    imm_match = re.search(r"come\s+into\s+force\s+with\s+immediate\s+effect", text, re.IGNORECASE)

    if gen_match:
        raw_d = gen_match.group(1).strip().rstrip(".")
        iso_d = parse_indian_gazette_date(raw_d)
        return iso_d, raw_d, []

    if imm_match:
        return notification_date_iso, "immediate effect", []

    # 2. Scoped effective dates (when no document-wide commencement clause exists)
    scopes: List[Dict[str, Any]] = []

    # A. Numbered scopes (e.g. 15/2025: '(1) with effect from...', '(2) with effect from...')
    scoped_numbered = list(
        re.finditer(
            r"\(([0-9]+)\)\s+with\s+effect\s+from\s+(?:the\s+)?([0-9]{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?[A-Za-z]+,?\s+[0-9]{4})",
            text,
            re.IGNORECASE,
        )
    )
    for sm in scoped_numbered:
        g_num = int(sm.group(1))
        raw_d = sm.group(2).strip().rstrip(".")
        snip_start = max(0, sm.start() - 20)
        snip_end = min(len(text), sm.end() + 100)
        raw_context = " ".join(text[snip_start:snip_end].split())
        scopes.append(
            {
                "scope_type": "NUMBERED_GROUP",
                "group_number": g_num,
                "raw_clause": sm.group(0).strip(),
                "raw_date": raw_d,
                "parsed_date": parse_indian_gazette_date(raw_d),
                "context_snippet": raw_context,
            }
        )

    # B. Clause-level inline dates (e.g. 05/2025, 06/2025)
    inline_wef = list(
        re.finditer(
            r"with\s+effect\s+from\s+(?:the\s+)?([0-9]{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?[A-Za-z]+,?\s+[0-9]{4})",
            text,
            re.IGNORECASE,
        )
    )
    for im in inline_wef:
        if any(abs(im.start() - sm.start()) < 20 for sm in scoped_numbered):
            continue
        raw_d = im.group(1).strip().rstrip(".")
        snip_start = max(0, im.start() - 50)
        snip_end = min(len(text), im.end() + 80)
        raw_context = " ".join(text[snip_start:snip_end].split())
        scopes.append(
            {
                "scope_type": "CLAUSE_SCOPED",
                "group_number": len(scopes) + 1,
                "raw_clause": im.group(0).strip(),
                "raw_date": raw_d,
                "parsed_date": parse_indian_gazette_date(raw_d),
                "context_snippet": raw_context,
            }
        )

    doc_raw = None
    if scopes:
        parts = [f"Scope {s['group_number']}: {s['raw_date']}" for s in scopes]
        doc_raw = "SCOPED_DATES: " + "; ".join(parts)

    return None, doc_raw, scopes


def extract_statutory_authority(text: str) -> Tuple[Optional[str], List[str]]:
    """Extract the statutory authority sentence and parsed sections invoked."""
    norm_text = " ".join(text[:4000].split())
    auth_m = re.search(
        r"In\s+exercise\s+of\s+the\s+powers\s+conferred\s+(?:by\s+)?(?:and\s+under\s+)?(.*?Central\s+Goods\s+and\s+Services\s*(?:Tax)?\s*Act,\s*2017\s*(?:\([0-9]+\s+of\s+[0-9]+\))?)",
        norm_text,
        re.IGNORECASE,
    )
    if not auth_m:
        return None, []

    raw_clause = auth_m.group(0).strip()
    sections: List[str] = []

    # Detect section 9 subsections
    if re.search(r"section\s+9\b", raw_clause, re.I):
        sub_matches = re.findall(r"sub-sections?\s+([0-9,\s(and)]+)\s+of\s+section\s+9", raw_clause, re.I)
        if sub_matches:
            digits = re.findall(r"\b([1-5])\b", sub_matches[0])
            for d in digits:
                sec = f"Section 9({d})"
                if sec not in sections:
                    sections.append(sec)
        else:
            if "Section 9" not in sections:
                sections.append("Section 9")

    # Detect section 11 subsections
    if re.search(r"section\s+11\b", raw_clause, re.I):
        sub_matches = re.findall(r"sub-sections?\s+([0-9,\s(and)]+)\s+of\s+section\s+11", raw_clause, re.I)
        if sub_matches:
            digits = re.findall(r"\b([1-4])\b", sub_matches[0])
            for d in digits:
                sec = f"Section 11({d})"
                if sec not in sections:
                    sections.append(sec)
        else:
            if "Section 11" not in sections:
                sections.append("Section 11")

    # Detect section 15 subsections
    if re.search(r"section\s+15\b", raw_clause, re.I):
        sub_matches = re.findall(r"sub-sections?\s+([0-9,\s(and)]+)\s+of\s+section\s+15", raw_clause, re.I)
        if sub_matches:
            digits = re.findall(r"\b([1-5])\b", sub_matches[0])
            for d in digits:
                sec = f"Section 15({d})"
                if sec not in sections:
                    sections.append(sec)
        else:
            if "Section 15" not in sections:
                sections.append("Section 15")

    # Detect section 16 subsections
    if re.search(r"section\s+16\b", raw_clause, re.I):
        sub_matches = re.findall(r"sub-sections?\s+([0-9,\s(and)]+)\s+of\s+section\s+16", raw_clause, re.I)
        if sub_matches:
            digits = re.findall(r"\b([1-4])\b", sub_matches[0])
            for d in digits:
                sec = f"Section 16({d})"
                if sec not in sections:
                    sections.append(sec)
        else:
            if "Section 16" not in sections:
                sections.append("Section 16")

    # Detect section 148
    if re.search(r"section\s+148\b", raw_clause, re.I):
        if "Section 148" not in sections:
            sections.append("Section 148")

    return raw_clause, sections


def extract_target_and_superseded_notifications(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract any amended target notification or superseded notification."""
    norm_text = " ".join(text[:4000].split())
    target_notif = None
    superseded_notif = None

    # Superseded
    sup_m = re.search(
        r"in\s+supersession\s+of[^\.]*?(?:No\.?|number)\s*([0-9]+/[0-9]{4}\s*-\s*Central\s+Tax\s*\(\s*Rate\s*\))",
        norm_text,
        re.IGNORECASE,
    )
    if sup_m:
        superseded_notif = canonicalize_notification_number(sup_m.group(1))

    # Target (amendment)
    amend_m = re.search(
        r"(?:amendments?\s+in\s+the\s+notification|amend\s+the\s+notification)[^\.]*?(?:No\.?|number)\s*([0-9]+/[0-9]{4}\s*-\s*Central\s+Tax\s*\(\s*Rate\s*\))",
        norm_text,
        re.IGNORECASE,
    )
    if amend_m:
        target_notif = canonicalize_notification_number(amend_m.group(1))

    return target_notif, superseded_notif


def extract_file_number_and_signatory(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract the CBIC file reference number and signatory officer."""
    norm_text = " ".join(text.split())

    # File number
    fno_m = re.search(r"\[\s*(F\.\s*No\.?[^\]]+)\]", norm_text, re.IGNORECASE)
    fno = fno_m.group(1).strip() if fno_m else None

    # Signatory name and designation
    sig = None
    sig_m = re.search(
        r"\[\s*F\.\s*No\.?[^\]]+\]\s*(?:\(([^)]+)\)|([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+))\s*([^\.\n]{0,80})",
        norm_text,
        re.IGNORECASE,
    )
    if sig_m:
        name = sig_m.group(1) or sig_m.group(2)
        designation = sig_m.group(3).strip() if sig_m.group(3) else ""
        if name:
            designation = re.sub(r"Note\s*:.*$", "", designation, flags=re.I).strip()
            sig = f"{name.strip()}, {designation}".rstrip(", ")

    return fno, sig


def extract_provenance_note(text: str) -> Optional[str]:
    """Extract trailing provenance / historical publication note."""
    norm_text = " ".join(text[-3000:].split())
    note_m = re.search(
        r"Note\s*:\s*[-–—]?\s*(The\s+principal\s+notification[^\.]*\.(?:\s*and\s+(?:was\s+)?last\s+amended[^\.]*\.)?)",
        norm_text,
        re.IGNORECASE,
    )
    if note_m:
        return note_m.group(1).strip()
    return None


def classify_document_type(
    notif_num: Optional[str],
    text: str,
    target_notif: Optional[str],
    superseded_notif: Optional[str],
    total_chars: int,
) -> str:
    """Classify the notification document into functional tax types."""
    if total_chars == 0:
        return "UNKNOWN"

    norm_head = " ".join(text[:2500].split()).lower()

    if superseded_notif and ("01/2017" in superseded_notif or "1/2017" in superseded_notif):
        return "RATE_SCHEDULE"

    if superseded_notif and ("02/2017" in superseded_notif or "2/2017" in superseded_notif):
        return "EXEMPTION"

    if "rate of the central tax of" in norm_head and "schedule appended" in norm_head:
        return "RATE_SCHEDULE"

    if target_notif or "hereby makes the following" in norm_head and "amendment" in norm_head:
        return "AMENDMENT"

    if "exempts intra-state supplies" in norm_head or "exempted goods" in norm_head:
        return "EXEMPTION"

    return "SPECIAL"


def extract_pdf_document(pdf_path: Union[str, Path]) -> NotificationDocument:
    """Extract text, page metrics, and document metadata from a notification PDF."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    file_size = path.stat().st_size
    file_name = path.name

    doc = pymupdf.open(path)
    page_count = len(doc)

    pages: List[NotificationPage] = []
    total_chars = 0
    total_words = 0
    zero_text_pages: List[int] = []
    total_drawings = 0
    total_raster_images = 0

    full_text_parts: List[str] = []

    for i, page in enumerate(doc, start=1):
        p_text = page.get_text()
        p_text_clean = p_text.replace("\u200b", "").replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
        chars = len(p_text_clean.strip())
        words = len(p_text_clean.split())
        total_chars += chars
        total_words += words

        if chars == 0:
            zero_text_pages.append(i)

        total_drawings += len(page.get_drawings())
        total_raster_images += len(page.get_images())

        pages.append(NotificationPage(page_num=i, char_count=chars, word_count=words, text=p_text_clean))
        full_text_parts.append(p_text_clean)

    full_text = "\n\n".join(full_text_parts)
    avg_chars = round(total_chars / page_count, 2) if page_count > 0 else 0.0

    # Determine extraction status
    if total_chars == 0:
        extraction_status = "FAILED_ZERO_TEXT"
        error_msg = (
            f"0 characters extracted across {page_count} page(s). "
            f"Found {total_drawings} vector drawings and {total_raster_images} raster images. "
            "File contains vector outline glyphs without font text stream (NEEDS_VISUAL_EXTRACTION)."
        )
    elif avg_chars < 100:
        extraction_status = "SUSPECT_LOW_TEXT"
        error_msg = f"Low average characters per page ({avg_chars} chars/page)."
    else:
        extraction_status = "SUCCESS"
        error_msg = None

    # Metadata extraction
    notif_num, notif_num_raw = extract_notification_number(full_text)
    notif_date_iso, notif_date_raw = extract_notification_date(full_text)
    eff_date_iso, eff_date_raw, eff_scopes = extract_effective_dates(full_text, notif_date_iso)
    auth_raw, sections = extract_statutory_authority(full_text)
    target_notif, superseded_notif = extract_target_and_superseded_notifications(full_text)
    file_num, sig_raw = extract_file_number_and_signatory(full_text)
    prov_note = extract_provenance_note(full_text)
    doc_type = classify_document_type(notif_num, full_text, target_notif, superseded_notif, total_chars)

    return NotificationDocument(
        file_path=str(path.resolve()),
        file_name=file_name,
        file_size_bytes=file_size,
        page_count=page_count,
        total_char_count=total_chars,
        total_word_count=total_words,
        avg_chars_per_page=avg_chars,
        pages_with_zero_text=zero_text_pages,
        vector_drawings_count=total_drawings,
        raster_images_count=total_raster_images,
        extraction_status=extraction_status,
        error_message=error_msg,
        notification_number=notif_num,
        notification_number_raw=notif_num_raw,
        notification_date=notif_date_iso,
        notification_date_raw=notif_date_raw,
        effective_date=eff_date_iso,
        effective_date_raw=eff_date_raw,
        effective_date_scopes=eff_scopes,
        document_type=doc_type,
        legal_authority_raw=auth_raw,
        sections_invoked=sections,
        target_notification=target_notif,
        superseded_notification=superseded_notif,
        file_number=file_num,
        signatory_raw=sig_raw,
        provenance_note=prov_note,
        full_text=full_text,
        pages=pages,
    )
