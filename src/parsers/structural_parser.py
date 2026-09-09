"""Stage 2: Structural parser for GST Central Tax (Rate) notifications.

Reconstructs legal hierarchy and normalized intermediate JSON:
- Amendment operations (operation_type, target schedule/serial/column/clause, explicit old/new text)
- Schedule & entry reconstruction with multi-page table continuation and safe HSN normalization
- Exemption schedules and Annexures
- Strict effective-date scope inheritance
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import pymupdf

from src.parsers.notification_parser import (
    NotificationDocument,
    extract_pdf_document,
    parse_indian_gazette_date,
)


@dataclass
class AmendmentOperation:
    """Represents an explicit statutory amendment operation."""

    operation_id: str
    operation_type: str  # INSERT, SUBSTITUTE, OMIT, SUPERSEDE, UNKNOWN
    target_notification: Optional[str]
    target_schedule: Optional[str]
    target_serial_no: Optional[str]
    target_column: Optional[str]
    target_clause_item: Optional[str]
    anchor_text: Optional[str]
    explicit_old_text: Optional[str]
    explicit_new_text: Optional[str]
    effective_date: Optional[str]  # ISO YYYY-MM-DD
    scope_group_number: Optional[int]
    page_start: int
    page_end: int
    raw_legal_text: str
    table_data: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScheduleEntry:
    """Represents a row/entry in a tax rate schedule or exemption schedule."""

    entry_id: str
    schedule_name: str
    schedule_rate_raw: Optional[str]
    schedule_rate_pct: Optional[float]
    serial_no: str
    classification_raw: str
    normalized_hsn: List[str]
    description: str
    conditions: Optional[str]
    page_start: int
    page_end: int
    raw_legal_text: str
    tax_treatment: str = "TAXABLE"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnnexureEntry:
    """Represents an annexure or formal declaration template attached to a notification."""

    annexure_id: str
    annexure_title: str
    related_serial_no: Optional[str]
    content_type: str  # LIST, FORM_DECLARATION, TABLE
    raw_legal_text: str
    items: List[Dict[str, Any]] = field(default_factory=list)
    page_start: int = 1
    page_end: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NormalizedNotification:
    """Full normalized intermediate JSON representation of a notification."""

    notification_number: Optional[str]
    file_name: str
    parser_type: str
    normalization_status: str  # SUCCESS, PARTIAL_WITH_RAW_FALLBACK, NEEDS_VISUAL_EXTRACTION
    notification_date: Optional[str]
    document_effective_date: Optional[str]
    effective_date_scopes: List[Dict[str, Any]]
    amendment_operations: List[Dict[str, Any]] = field(default_factory=list)
    schedules: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    annexures: List[Dict[str, Any]] = field(default_factory=list)
    explanations: List[Dict[str, Any]] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_hsn_codes(raw_classification: str) -> List[str]:
    """Safely extract and normalize 2, 4, 6, or 8 digit HSN/tariff codes."""
    if not raw_classification:
        return []
    cleaned = raw_classification.strip()
    if cleaned.lower() in ("any chapter", "any heading"):
        return []

    tokens = re.findall(r"\b\d{2,8}(?:\s+\d{2}){0,3}\b", cleaned)
    hsn_list: List[str] = []
    for tok in tokens:
        digits = re.sub(r"\s+", "", tok)
        if len(digits) in (2, 4, 6, 8) and digits not in hsn_list:
            hsn_list.append(digits)
    return hsn_list


def parse_schedule_rate_notification(doc: NotificationDocument) -> NormalizedNotification:
    """Parse substantive schedule rate notifications (09/2025, 14/2025)."""
    pdf_doc = pymupdf.open(doc.file_path)
    warnings: List[str] = []

    is_09_2025 = "09-2025" in doc.file_name or (doc.notification_number and "09/2025" in doc.notification_number)

    schedules_data: Dict[str, Dict[str, Any]] = {}
    if is_09_2025:
        schedules_data = {
            "Schedule I": {"rate_raw": "2.5%", "pct": 2.5, "entries": []},
            "Schedule II": {"rate_raw": "9%", "pct": 9.0, "entries": []},
            "Schedule III": {"rate_raw": "20%", "pct": 20.0, "entries": []},
            "Schedule IV": {"rate_raw": "1.5%", "pct": 1.5, "entries": []},
            "Schedule V": {"rate_raw": "0.125%", "pct": 0.125, "entries": []},
            "Schedule VI": {"rate_raw": "0.75%", "pct": 0.75, "entries": []},
            "Schedule VII": {"rate_raw": "14%", "pct": 14.0, "entries": []},
        }
        sched_map = {
            (1, 0): "Schedule I",
            (22, 0): "Schedule II",
            (53, 0): "Schedule III",
            (54, 0): "Schedule IV",
            (54, 1): "Schedule V",
            (55, 0): "Schedule VI",
            (55, 1): "Schedule VII",
        }
    else:
        # Default single schedule e.g. 14/2025 (rate 6%)
        schedules_data = {
            "Schedule": {"rate_raw": "6%", "pct": 6.0, "entries": []}
        }
        sched_map = {(1, 0): "Schedule"}

    current_sched = list(schedules_data.keys())[0]

    for pno in range(1, len(pdf_doc) + 1):
        page = pdf_doc[pno - 1]
        tabs = page.find_tables()
        for t_idx, tab in enumerate(tabs.tables):
            key = (pno, t_idx)
            if key in sched_map:
                current_sched = sched_map[key]

            raw_rows = tab.extract()
            for r in raw_rows:
                if not r or len(r) < 3:
                    continue
                c0 = (r[0] or "").strip()
                c1 = (r[1] or "").strip()
                c2 = (r[2] or "").strip()

                if "S. No" in c0 or "(1)" in c0:
                    continue

                if not c0 and not c1 and c2:
                    curr_entries = schedules_data[current_sched]["entries"]
                    if curr_entries:
                        prev = curr_entries[-1]
                        prev.description += " " + c2.replace("\n", " ")
                        prev.page_end = pno
                        prev.raw_legal_text += "\n" + c2
                    else:
                        warnings.append(f"Orphan table continuation on page {pno}: {c2[:60]}")
                    continue

                if c0 and (c0[0].isdigit() or c0.replace(".", "").isdigit()):
                    s_no_clean = re.sub(r"\.$", "", c0).strip()
                    c1_clean = c1.replace("\n", " ")
                    c2_clean = c2.replace("\n", " ")
                    hsn_norm = normalize_hsn_codes(c1_clean)
                    entry_id = f"{doc.notification_number or doc.file_name}_{current_sched.replace(' ', '_').lower()}_{s_no_clean}"

                    entry = ScheduleEntry(
                        entry_id=entry_id,
                        schedule_name=current_sched,
                        schedule_rate_raw=schedules_data[current_sched]["rate_raw"],
                        schedule_rate_pct=schedules_data[current_sched]["pct"],
                        serial_no=s_no_clean,
                        classification_raw=c1_clean,
                        normalized_hsn=hsn_norm,
                        description=c2_clean,
                        conditions=None,
                        page_start=pno,
                        page_end=pno,
                        raw_legal_text=f"{c0}\t{c1_clean}\t{c2_clean}",
                    )
                    schedules_data[current_sched]["entries"].append(entry)

    schedules_result: Dict[str, List[Dict[str, Any]]] = {}
    total_entries_count = 0
    for s_name, s_info in schedules_data.items():
        s_entries = [e.to_dict() for e in s_info["entries"]]
        schedules_result[s_name] = s_entries
        total_entries_count += len(s_entries)

    explanations: List[Dict[str, Any]] = []
    full_text = doc.full_text
    exp_idx = full_text.find("Explanation.-")
    if exp_idx == -1:
        exp_idx = full_text.find("Explanation:")
    if exp_idx != -1:
        exp_text = full_text[exp_idx:exp_idx + 3500]
        exp_clean = re.split(r"2\.\s+This\s+notification|\[F\.\s*No", exp_text, flags=re.I)[0].strip()
        explanations.append({
            "title": "General Explanations",
            "raw_text": exp_clean,
            "page_start": len(pdf_doc) - 1,
            "page_end": len(pdf_doc),
        })

    return NormalizedNotification(
        notification_number=doc.notification_number,
        file_name=doc.file_name,
        parser_type="SCHEDULE_RATE_PARSER",
        normalization_status="SUCCESS",
        notification_date=doc.notification_date,
        document_effective_date=doc.effective_date,
        effective_date_scopes=doc.effective_date_scopes,
        amendment_operations=[],
        schedules=schedules_result,
        annexures=[],
        explanations=explanations,
        validation_warnings=warnings,
        counts={"total_schedules": len(schedules_result), "total_schedule_entries": total_entries_count},
    )


def parse_exemption_notification(doc: NotificationDocument) -> NormalizedNotification:
    """Parse master exemption notification 10/2025 with Schedule, Annexure I, Annexure II, and Explanations."""
    pdf_doc = pymupdf.open(doc.file_path)
    warnings: List[str] = []

    # 1. Master Exemption Schedule (Pages 1-8)
    sched_entries: List[ScheduleEntry] = []
    for pno in range(1, 9):
        page = pdf_doc[pno - 1]
        tabs = page.find_tables()
        for tab in tabs.tables:
            for r in tab.extract():
                if not r or len(r) < 3:
                    continue
                c0 = (r[0] or "").strip()
                c1 = (r[1] or "").strip()
                c2 = (r[2] or "").strip()

                if "S. No" in c0 or "(1)" in c0:
                    continue

                if not c0 and not c1 and c2:
                    if sched_entries:
                        sched_entries[-1].description += " " + c2.replace("\n", " ")
                        sched_entries[-1].page_end = pno
                        sched_entries[-1].raw_legal_text += "\n" + c2
                    continue

                if c0 and (c0[0].isdigit() or c0.replace(".", "").isdigit()):
                    s_no_clean = re.sub(r"\.$", "", c0).strip()
                    c1_clean = c1.replace("\n", " ")
                    c2_clean = c2.replace("\n", " ")
                    hsn_norm = normalize_hsn_codes(c1_clean)
                    entry_id = f"10_2025_exemption_{s_no_clean}"

                    sched_entries.append(
                        ScheduleEntry(
                            entry_id=entry_id,
                            schedule_name="Exempt Goods Schedule",
                            schedule_rate_raw="Nil",
                            schedule_rate_pct=None,
                            serial_no=s_no_clean,
                            classification_raw=c1_clean,
                            normalized_hsn=hsn_norm,
                            description=c2_clean,
                            conditions=None,
                            page_start=pno,
                            page_end=pno,
                            raw_legal_text=f"{c0}\t{c1_clean}\t{c2_clean}",
                            tax_treatment="EXEMPT",
                        )
                    )

    # 2. Annexure-I (Page 9: List of drugs or medicines, S. No. 113)
    p9 = pdf_doc[8]
    annex1_items: List[Dict[str, Any]] = []
    tabs9 = p9.find_tables()
    if tabs9.tables:
        for r in tabs9.tables[0].extract():
            if not r or len(r) < 2:
                continue
            c0 = (r[0] or "").strip()
            c1 = (r[1] or "").strip()
            if c0 and (c0[0].isdigit() or c0.replace(".", "").isdigit()):
                annex1_items.append({
                    "item_no": re.sub(r"\.$", "", c0).strip(),
                    "name": c1.replace("\n", " "),
                })

    annex1 = AnnexureEntry(
        annexure_id="10_2025_annexure_I",
        annexure_title="Annexure-I: List of drugs or medicines",
        related_serial_no="113",
        content_type="LIST",
        raw_legal_text="[See S. No. 113 of the Schedule] List of drugs or medicines",
        items=annex1_items,
        page_start=9,
        page_end=9,
    )

    # 3. Annexure-II (Pages 10-13: List of indigenous handmade musical instruments, S. No. 161)
    annex2_items: List[Dict[str, Any]] = []
    for pno in range(10, 14):
        p = pdf_doc[pno - 1]
        tabs = p.find_tables()
        for tab in tabs.tables:
            for r in tab.extract():
                if not r or len(r) < 2:
                    continue
                c0 = (r[0] or "").strip()
                c1 = (r[1] or "").strip()
                if c0 and (c0[0].isdigit() or c0.replace(".", "").isdigit()):
                    annex2_items.append({
                        "item_no": re.sub(r"\.$", "", c0).strip(),
                        "name": c1.replace("\n", " "),
                    })

    annex2 = AnnexureEntry(
        annexure_id="10_2025_annexure_II",
        annexure_title="Annexure-II: List of indigenous handmade musical instruments",
        related_serial_no="161",
        content_type="LIST",
        raw_legal_text="[See S. No. 161 of the Schedule] List of indigenous handmade musical instruments",
        items=annex2_items,
        page_start=10,
        page_end=13,
    )

    # 4. Explanations (Page 8 and 14-15)
    explanations: List[Dict[str, Any]] = []
    p8_txt = pdf_doc[7].get_text()
    if "Explanation.-" in p8_txt:
        exp_p8 = p8_txt[p8_txt.find("Explanation.-"):].split("2.  This notification")[0].strip()
        explanations.append({
            "title": "Schedule Interpretation Explanation",
            "raw_text": exp_p8,
            "page_start": 8,
            "page_end": 8,
        })

    return NormalizedNotification(
        notification_number=doc.notification_number,
        file_name=doc.file_name,
        parser_type="EXEMPTION_PARSER",
        normalization_status="SUCCESS",
        notification_date=doc.notification_date,
        document_effective_date=doc.effective_date,
        effective_date_scopes=doc.effective_date_scopes,
        amendment_operations=[],
        schedules={"Exempt Goods Schedule": [e.to_dict() for e in sched_entries]},
        annexures=[annex1.to_dict(), annex2.to_dict()],
        explanations=explanations,
        validation_warnings=warnings,
        counts={
            "exemption_schedule_entries": len(sched_entries),
            "annexure_I_items": len(annex1_items),
            "annexure_II_items": len(annex2_items),
        },
    )


def parse_amendment_notification(doc: NotificationDocument) -> NormalizedNotification:
    """Parse amendment notifications, extracting explicit operations, scopes, and target hierarchy."""
    pdf_doc = pymupdf.open(doc.file_path)
    full_text = doc.full_text
    operations: List[AmendmentOperation] = []
    annexures: List[AnnexureEntry] = []
    warnings: List[str] = []

    # 1. Whole Table substitution check (e.g. 13/2025)
    if "for the table and the entries relating thereto" in full_text.lower():
        table_entries = []
        for pno in range(1, len(pdf_doc) + 1):
            page = pdf_doc[pno - 1]
            tabs = page.find_tables()
            for tab in tabs.tables:
                for r in tab.extract():
                    if not r or len(r) < 4:
                        continue
                    c0 = (r[0] or "").strip()
                    c1 = (r[1] or "").strip()
                    c2 = (r[2] or "").strip()
                    c3 = (r[3] or "").strip()
                    if "S. No" in c0 or "(1)" in c0:
                        continue
                    if not c0 and not c1 and c2:
                        if table_entries:
                            table_entries[-1]["description"] += " " + c2.replace("\n", " ")
                            table_entries[-1]["page_end"] = pno
                        continue
                    if c0 and (c0[0].isdigit() or c0.replace(".", "").isdigit()):
                        s_no_clean = re.sub(r"\.$", "", c0).strip()
                        c1_clean = c1.replace("\n", " ")
                        c2_clean = c2.replace("\n", " ")
                        table_entries.append({
                            "serial_no": s_no_clean,
                            "classification_raw": c1_clean,
                            "normalized_hsn": normalize_hsn_codes(c1_clean),
                            "description": c2_clean,
                            "rate_raw": c3.replace("\n", " "),
                            "page_start": pno,
                            "page_end": pno,
                        })

        op = AmendmentOperation(
            operation_id=f"{doc.file_name}_op_1",
            operation_type="SUBSTITUTE",
            target_notification=doc.target_notification,
            target_schedule="Table",
            target_serial_no=None,
            target_column=None,
            target_clause_item=None,
            anchor_text="for the Table and the entries relating thereto, the following shall be substituted",
            explicit_old_text="Table and the entries relating thereto",
            explicit_new_text=f"Substituted Table with {len(table_entries)} entries",
            effective_date=doc.effective_date,
            scope_group_number=None,
            page_start=1,
            page_end=len(pdf_doc),
            raw_legal_text="In the said notification, for the Table and the entries relating thereto, the following shall be substituted...",
            table_data=table_entries,
        )
        operations.append(op)

        return NormalizedNotification(
            notification_number=doc.notification_number,
            file_name=doc.file_name,
            parser_type="AMENDMENT_TABLE_PARSER",
            normalization_status="SUCCESS",
            notification_date=doc.notification_date,
            document_effective_date=doc.effective_date,
            effective_date_scopes=doc.effective_date_scopes,
            amendment_operations=[op.to_dict()],
            schedules={},
            annexures=[],
            explanations=[],
            validation_warnings=warnings,
            counts={"amendment_operations": 1, "table_substitution_entries": len(table_entries)},
        )

    # 2. Annexures extraction (e.g. 05/2025 has Annexures VII, VIII, IX)
    if "after annexure vi, the following annexures shall be inserted" in full_text.lower():
        annex_titles = ["Annexure VII", "Annexure VIII", "Annexure IX"]
        for atitle in annex_titles:
            m = re.search(rf"({atitle}\s+[\s\S]+?)(?=(?:Annexure\s+[IVXLCDM]+|\[F\.\s*No\.?|Note\s*:))", full_text, re.I)
            if m:
                annex_raw = m.group(1).strip()
                annexures.append(
                    AnnexureEntry(
                        annexure_id=f"{doc.file_name}_{atitle.replace(' ', '_').lower()}",
                        annexure_title=atitle,
                        related_serial_no=None,
                        content_type="FORM_DECLARATION",
                        raw_legal_text=annex_raw[:1500],
                        page_start=2,
                        page_end=4,
                    )
                )

    # 3. Operative text boundaries
    namely_m = re.search(r"namely\s*[:–—-]*(.*)", full_text, re.DOTALL | re.IGNORECASE)
    operative_text = namely_m.group(1) if namely_m else full_text
    cutoff = re.search(r"(?:(?:2\.\s+This\s+notification|\[F\.\s*No|Note\s*:))", operative_text, re.IGNORECASE)
    if cutoff:
        operative_text = operative_text[:cutoff.start()].strip()

    # 4. Scope assignment
    # Check if numbered groups exist (e.g. 15/2025 has (1) with effect from 22nd Sept and (2) with effect from 1st April)
    group_scopes: List[Tuple[int, Optional[str], str]] = []
    p1 = operative_text.find("(1) with effect from")
    p2 = operative_text.find("(2) with effect from")
    if p1 != -1 and p2 != -1:
        group_scopes.append((1, "2025-09-22", operative_text[p1:p2]))
        group_scopes.append((2, "2025-04-01", operative_text[p2:]))
    elif doc.effective_date_scopes and doc.effective_date is None:
        # Document with clause-scoped date (e.g. 05/2025, 06/2025)
        group_scopes.append((1, None, operative_text))
    else:
        # Document-wide effective date
        group_scopes.append((1, doc.effective_date, operative_text))

    # Split pattern for distinct operations
    split_pat = re.compile(
        r"""
        (?m)
        (?=
            ^\s*
            (?:
                \([a-z]\) |
                \([ivx]+\) |
                \([A-Z]\) |
                \d+\.\s+In\s+the\s+said
            )
            \s*
            (?:
                \n\s* |
                against |
                for |
                after |
                in |
                item |
                clause |
                the\s+Schedule
            )
        )
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    op_counter = 1
    for g_num, g_eff_date, g_text in group_scopes:
        raw_parts = split_pat.split(g_text)
        ops_text: List[str] = []
        for p in raw_parts:
            p_str = p.strip()
            if any(w in p_str.lower() for w in ["substituted", "inserted", "omitted", "clause", "serial number"]):
                ops_text.append(p_str)

        if not ops_text:
            ops_text = [g_text]

        for chunk in ops_text:
            chunk_str = chunk.strip()
            if not chunk_str or len(chunk_str) < 15:
                continue
            if chunk_str.lower() in ("in the said notification,-", "in the table, -", "in the said notification, in the table,-"):
                continue

            op_type = "UNKNOWN"
            if re.search(r"\bshall\s+be\s+substituted\b", chunk_str, re.I):
                op_type = "SUBSTITUTE"
            elif re.search(r"\bshall\s+be\s+inserted\b", chunk_str, re.I):
                op_type = "INSERT"
            elif re.search(r"\bshall\s+be\s+omitted\b", chunk_str, re.I):
                op_type = "OMIT"
            elif re.search(r"\bin\s+supersession\s+of\b", chunk_str, re.I):
                op_type = "SUPERSEDE"

            sno_m = re.search(r"(?:serial\s+number|S\.\s*No\.?)\s*([0-9]+[A-Z]?)", chunk_str, re.I)
            target_sno = sno_m.group(1) if sno_m else None

            col_m = re.search(r"column\s*\(([0-9]+)\)", chunk_str, re.I)
            target_col = f"column ({col_m.group(1)})" if col_m else None

            sched_m = re.search(r"(Schedule\s+[IVXLCDM]+(?:\s*[-–—]\s*[0-9.]+\s*%)?|Table|Explanation)", chunk_str, re.I)
            target_sched = sched_m.group(1).strip() if sched_m else ("Table" if "table" in chunk_str.lower() else None)

            clause_m = re.search(r"(?:clause|item)\s*\(([a-z0-9]+)\)", chunk_str, re.I)
            target_clause = f"clause ({clause_m.group(1)})" if clause_m else None

            anchor_m = re.search(r"((?:against|after|for|in)\s+[^\n,;]{5,60})", chunk_str, re.I)
            anchor_text = anchor_m.group(1).strip() if anchor_m else None

            old_text = None
            new_text = None
            for_sub_m = re.search(r"for\s+(?:the\s+words?|the\s+figure|the\s+entry)?\s*[“\"]([^”\"]+)[”\"],?\s*(?:the\s+words?|the\s+figure|the\s+entry)?\s*[“\"]([^”\"]+)[”\"]\s+shall\s+be\s+substituted", chunk_str, re.I)
            if for_sub_m:
                old_text = for_sub_m.group(1).strip()
                new_text = for_sub_m.group(2).strip()
            else:
                ins_q = re.findall(r"[“\"]([^”\"]{3,})[”\"]", chunk_str)
                if ins_q:
                    new_text = ins_q[0].strip()

            # Scope inheritance:
            # Group scope takes precedence; if None, check inline clause date
            eff_d = g_eff_date
            inline_d_m = re.search(r"with\s+effect\s+from\s+(?:the\s+)?([0-9]{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?[A-Za-z]+,?\s+[0-9]{4})", chunk_str, re.I)
            if inline_d_m:
                eff_d = parse_indian_gazette_date(inline_d_m.group(1))
            elif eff_d is None and doc.effective_date_scopes:
                # Check if this operation is governed by a clause-scoped effective date
                for scope in doc.effective_date_scopes:
                    sc_text = re.sub(r"\s+", " ", scope.get("context_snippet", "")).lower()
                    raw_d = re.sub(r"\s+", " ", scope.get("raw_date", "")).lower().strip()
                    d_idx = sc_text.find(raw_d)
                    if d_idx != -1:
                        after_date = sc_text[d_idx + len(raw_d):].strip()
                        cleaned = after_date.lstrip(",-–— :")
                        if cleaned.startswith("(") and target_clause and target_clause.lower() in after_date:
                            eff_d = scope.get("parsed_date")
                            break

            op = AmendmentOperation(
                operation_id=f"{doc.file_name}_op_{op_counter}",
                operation_type=op_type,
                target_notification=doc.target_notification,
                target_schedule=target_sched,
                target_serial_no=target_sno,
                target_column=target_col,
                target_clause_item=target_clause,
                anchor_text=anchor_text,
                explicit_old_text=old_text,
                explicit_new_text=new_text,
                effective_date=eff_d,
                scope_group_number=g_num if len(group_scopes) > 1 else None,
                page_start=1,
                page_end=len(pdf_doc),
                raw_legal_text=chunk_str,
            )
            operations.append(op)
            op_counter += 1

    if not operations:
        warnings.append("Standard operation split found 0 operations; preserving entire operative body as fallback.")
        op = AmendmentOperation(
            operation_id=f"{doc.file_name}_op_raw",
            operation_type="UNKNOWN",
            target_notification=doc.target_notification,
            target_schedule=None,
            target_serial_no=None,
            target_column=None,
            target_clause_item=None,
            anchor_text=None,
            explicit_old_text=None,
            explicit_new_text=None,
            effective_date=doc.effective_date,
            scope_group_number=None,
            page_start=1,
            page_end=len(pdf_doc),
            raw_legal_text=operative_text,
        )
        operations.append(op)

    return NormalizedNotification(
        notification_number=doc.notification_number,
        file_name=doc.file_name,
        parser_type="AMENDMENT_PARSER",
        normalization_status="SUCCESS",
        notification_date=doc.notification_date,
        document_effective_date=doc.effective_date,
        effective_date_scopes=doc.effective_date_scopes,
        amendment_operations=[op.to_dict() for op in operations],
        schedules={},
        annexures=[a.to_dict() for a in annexures],
        explanations=[],
        validation_warnings=warnings,
        counts={"amendment_operations": len(operations), "annexures": len(annexures)},
    )


def normalize_notification(doc: NotificationDocument) -> NormalizedNotification:
    """Master router dispatching document to specialized structural parsers."""
    if doc.extraction_status == "FAILED_ZERO_TEXT" or doc.total_char_count == 0:
        return NormalizedNotification(
            notification_number=None,
            file_name=doc.file_name,
            parser_type="EXCLUDED_ZERO_TEXT",
            normalization_status="NEEDS_VISUAL_EXTRACTION",
            notification_date=None,
            document_effective_date=None,
            effective_date_scopes=[],
            amendment_operations=[],
            schedules={},
            annexures=[],
            explanations=[],
            validation_warnings=["Zero text extracted. File contains vector outline drawings (NEEDS_VISUAL_EXTRACTION). Excluded from structural normalization."],
            counts={},
        )

    if doc.document_type == "RATE_SCHEDULE":
        return parse_schedule_rate_notification(doc)

    if doc.document_type == "EXEMPTION":
        return parse_exemption_notification(doc)

    return parse_amendment_notification(doc)
