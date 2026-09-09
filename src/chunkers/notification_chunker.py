"""Legal-structure-first chunker for GST Central Tax (Rate) notifications.

Transforms Stage 2 normalized notification JSON structures into structure-aware,
retrieval-ready chunks obeying the ~450 token maximum:
- Atomic statutory units (schedule entries, amendment operations, table rows, annexure forms).
- Short adjacent entries grouped only within the same schedule, rate, and scope (<=450 tokens).
- Oversized operations (>450 tokens) split using token-aware word windows and ~60-token overlap.
- Generated inherited context header prepended to verbatim legal text.
- Verbatim raw statutory text preserved separately in metadata.
- Exemption semantics strictly maintained (tax_treatment="EXEMPT", schedule_rate_raw="Nil", schedule_rate_pct=None).
- Strict propagation of target_notification and effective-date scopes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer

from src.chunkers.act_chunker import count_tokens, split_large_text

DEFAULT_MAX_TOKENS = 450
DEFAULT_OVERLAP_TOKENS = 60
DEFAULT_TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class NotificationChunk:
    """Represents a single retrieval-ready statutory chunk."""

    chunk_id: str
    text: str  # Generated inherited context header + \n\n + raw_legal_text
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NotificationChunker:
    """Generates structure-aware chunks from Stage 2 normalized notifications."""

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        tokenizer: Optional[Any] = None,
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_MODEL)
        else:
            self.tokenizer = tokenizer

    def chunk_notification(self, norm_data: Dict[str, Any]) -> List[NotificationChunk]:
        """Dispatch a normalized notification dictionary to its specialized chunker."""
        parser_type = norm_data.get("parser_type", "")
        status = norm_data.get("normalization_status", "")

        # Excluded documents (e.g. 18/2025 vector outline)
        if parser_type == "EXCLUDED_ZERO_TEXT" or status == "NEEDS_VISUAL_EXTRACTION":
            return []

        chunks: List[NotificationChunk] = []
        clean_num = self._clean_id(norm_data.get("notification_number") or norm_data.get("file_name", "notif"))

        if parser_type == "SCHEDULE_RATE_PARSER":
            chunks = self._chunk_schedule_notification(norm_data, clean_num)
        elif parser_type == "EXEMPTION_PARSER":
            chunks = self._chunk_exemption_notification(norm_data, clean_num)
        elif parser_type == "AMENDMENT_TABLE_PARSER":
            chunks = self._chunk_table_substitution_notification(norm_data, clean_num)
        else:
            # Standard amendment notification (AMENDMENT_PARSER)
            chunks = self._chunk_amendment_notification(norm_data, clean_num)

        return chunks

    def _clean_id(self, notif_str: str) -> str:
        """Convert notification number to a clean slug for chunk IDs."""
        return notif_str.replace("/", "-").replace(" ", "_").replace(".", "-").replace("(", "").replace(")", "").lower()

    # -------------------------------------------------------------------------
    # 1. Schedule Notifications (e.g. 09/2025, 14/2025)
    # -------------------------------------------------------------------------
    def _chunk_schedule_notification(self, norm: Dict[str, Any], clean_id: str) -> List[NotificationChunk]:
        chunks: List[NotificationChunk] = []
        notif_no = norm.get("notification_number", "")
        notif_date = norm.get("notification_date")
        eff_date = norm.get("document_effective_date")
        schedules = norm.get("schedules", {})

        chunk_counter = 1

        for sched_name, entries in schedules.items():
            if not entries:
                continue

            for entry in entries:
                entry_chunks = self._emit_single_schedule_entry_chunks(
                    clean_id, chunk_counter, notif_no, notif_date, eff_date,
                    sched_name, entry, "RATE_SCHEDULE", "TAXABLE"
                )
                chunks.extend(entry_chunks)
                chunk_counter += len(entry_chunks)

        # Explanations if present
        for exp in norm.get("explanations", []):
            exp_chunks = self._chunk_explanation(clean_id, chunk_counter, notif_no, notif_date, eff_date, exp)
            chunks.extend(exp_chunks)
            chunk_counter += len(exp_chunks)

        return chunks

    def _emit_single_schedule_entry_chunks(
        self, clean_id: str, start_index: int, notif_no: str, notif_date: Optional[str],
        eff_date: Optional[str], sched_name: str, entry: Dict[str, Any],
        doc_type: str, tax_treatment: str
    ) -> List[NotificationChunk]:
        sno = entry.get("serial_no", "")
        hsn_raw = entry.get("classification_raw", "")
        rate_raw = entry.get("schedule_rate_raw", "")
        if tax_treatment == "EXEMPT" and not rate_raw:
            rate_raw = "Nil"

        sno_str = f"S. No. {sno}" if sno else "S. No. -"
        hsn_str = f"HSN {hsn_raw}" if hsn_raw else "HSN -"
        rate_clause = f"Rate: {rate_raw}, EXEMPT" if tax_treatment == "EXEMPT" else f"Rate: {rate_raw}"
        header = f"[Notification No. {notif_no} | {sched_name} ({rate_clause}) | {sno_str} | {hsn_str}]"
        raw_body = entry.get("raw_legal_text", "")
        full_text = f"{header}\n\n{raw_body}"
        tokens = count_tokens(self.tokenizer, full_text)

        all_hsn = entry.get("normalized_hsn", [])
        p_start = entry.get("page_start", 1)
        p_end = entry.get("page_end", 1)

        base_metadata = {
            "notification_number": notif_no,
            "notification_date": notif_date,
            "document_type": doc_type,
            "chunk_type": "SCHEDULE_ENTRY",
            "chunk_strategy": "schedule_entry",
            "effective_date": eff_date,
            "target_notification": None,
            "schedule": sched_name,
            "schedule_rate_raw": rate_raw,
            "tax_treatment": tax_treatment,
            "serial_numbers": [sno] if sno else [],
            "classification_raw": hsn_raw,
            "normalized_hsn": all_hsn,
            "operation_type": None,
            "source_page_start": p_start,
            "source_page_end": p_end,
            "raw_legal_text": raw_body,
            "inherited_context_header": header,
        }

        if tokens <= self.max_tokens:
            return [
                NotificationChunk(
                    chunk_id=f"chunk-{clean_id}-{start_index:04d}",
                    text=full_text,
                    token_count=tokens,
                    metadata=base_metadata,
                )
            ]

        # Fallback split if single entry > 450 tokens
        h_tokens = count_tokens(self.tokenizer, header)
        body_limit = max(1, self.max_tokens - h_tokens)
        parts = split_large_text(raw_body, body_limit, self.overlap_tokens, self.tokenizer)
        split_chunks = []
        for p_idx, part in enumerate(parts, 1):
            p_header = f"[Notification No. {notif_no} | {sched_name} ({rate_clause}) | S. No. {sno} (Part {p_idx}/{len(parts)}) | {hsn_str}]"
            p_text = f"{p_header}\n\n{part}"
            s_meta = {
                **base_metadata,
                "chunk_type": "SCHEDULE_ENTRY_SPLIT",
                "chunk_strategy": "schedule_token_split",
                "raw_legal_text": part,
                "inherited_context_header": p_header,
                "split_index": p_idx,
                "split_count": len(parts),
            }
            split_chunks.append(
                NotificationChunk(
                    chunk_id=f"chunk-{clean_id}-{start_index + p_idx - 1:04d}",
                    text=p_text,
                    token_count=count_tokens(self.tokenizer, p_text),
                    metadata=s_meta,
                )
            )
        return split_chunks

    # -------------------------------------------------------------------------
    # 2. Exemption Notifications (e.g. 10/2025)
    # -------------------------------------------------------------------------
    def _chunk_exemption_notification(self, norm: Dict[str, Any], clean_id: str) -> List[NotificationChunk]:
        chunks: List[NotificationChunk] = []
        notif_no = norm.get("notification_number", "")
        notif_date = norm.get("notification_date")
        eff_date = norm.get("document_effective_date")
        schedules = norm.get("schedules", {})

        chunk_counter = 1

        # A. Exemption Schedule (each entry is atomic)
        for sched_name, entries in schedules.items():
            if not entries:
                continue

            for entry in entries:
                entry_chunks = self._emit_single_schedule_entry_chunks(
                    clean_id, chunk_counter, notif_no, notif_date, eff_date,
                    sched_name, entry, "EXEMPTION", "EXEMPT"
                )
                chunks.extend(entry_chunks)
                chunk_counter += len(entry_chunks)

        # B. Annexures (Annexure I and Annexure II: grouped conservatively)
        for annex in norm.get("annexures", []):
            annex_chunks = self._chunk_annexure(clean_id, chunk_counter, notif_no, notif_date, eff_date, annex)
            chunks.extend(annex_chunks)
            chunk_counter += len(annex_chunks)

        # C. Explanations
        for exp in norm.get("explanations", []):
            exp_chunks = self._chunk_explanation(clean_id, chunk_counter, notif_no, notif_date, eff_date, exp)
            chunks.extend(exp_chunks)
            chunk_counter += len(exp_chunks)

        return chunks

    def _chunk_annexure(
        self, clean_id: str, start_index: int, notif_no: str, notif_date: Optional[str],
        eff_date: Optional[str], annex: Dict[str, Any]
    ) -> List[NotificationChunk]:
        chunks: List[NotificationChunk] = []
        title = annex.get("annexure_title", "")
        rel_sno = annex.get("related_serial_no")
        items = annex.get("items", [])
        raw_full = annex.get("raw_legal_text", "")
        p_start = annex.get("page_start", 1)
        p_end = annex.get("page_end", 1)

        rel_desc = f" (Related to S. No. {rel_sno})" if rel_sno else ""

        if not items:
            header = f"[Notification No. {notif_no} | {title}{rel_desc} | Format: {annex.get('content_type', 'DECLARATION')}]"
            full_text = f"{header}\n\n{raw_full}"
            tokens = count_tokens(self.tokenizer, full_text)

            if tokens <= self.max_tokens:
                metadata = {
                    "notification_number": notif_no,
                    "notification_date": notif_date,
                    "document_type": "EXEMPTION",
                    "chunk_type": "ANNEXURE_FORM",
                    "chunk_strategy": "annexure_form",
                    "effective_date": eff_date,
                    "target_notification": None,
                    "schedule": None,
                    "schedule_rate_raw": "Nil",
                    "tax_treatment": "EXEMPT",
                    "serial_numbers": [rel_sno] if rel_sno else [],
                    "classification_raw": None,
                    "normalized_hsn": [],
                    "operation_type": None,
                    "source_page_start": p_start,
                    "source_page_end": p_end,
                    "raw_legal_text": raw_full,
                    "inherited_context_header": header,
                }
                chunks.append(NotificationChunk(f"chunk-{clean_id}-{start_index:04d}", full_text, tokens, metadata))
            else:
                h_tokens = count_tokens(self.tokenizer, header)
                body_limit = max(1, self.max_tokens - h_tokens)
                parts = split_large_text(raw_full, body_limit, self.overlap_tokens, self.tokenizer)
                for part_idx, part in enumerate(parts, 1):
                    p_text = f"{header}\n\n{part}"
                    metadata = {
                        "notification_number": notif_no,
                        "notification_date": notif_date,
                        "document_type": "EXEMPTION",
                        "chunk_type": "ANNEXURE_FORM_SPLIT",
                        "chunk_strategy": "annexure_token_split",
                        "effective_date": eff_date,
                        "target_notification": None,
                        "schedule": None,
                        "schedule_rate_raw": "Nil",
                        "tax_treatment": "EXEMPT",
                        "serial_numbers": [rel_sno] if rel_sno else [],
                        "classification_raw": None,
                        "normalized_hsn": [],
                        "operation_type": None,
                        "source_page_start": p_start,
                        "source_page_end": p_end,
                        "raw_legal_text": part,
                        "inherited_context_header": header,
                        "split_index": part_idx,
                        "split_count": len(parts),
                    }
                    chunks.append(NotificationChunk(f"chunk-{clean_id}-{start_index + part_idx - 1:04d}", p_text, count_tokens(self.tokenizer, p_text), metadata))
            return chunks

        buffer: List[Dict[str, Any]] = []
        c_idx = start_index

        for item in items:
            candidate = [*buffer, item]
            min_i = candidate[0].get("item_no", "")
            max_i = candidate[-1].get("item_no", "")
            range_str = f"Item No. {min_i}" if min_i == max_i else f"Item Nos. {min_i}–{max_i}"
            h = f"[Notification No. {notif_no} | {title}{rel_desc} | {range_str}]"
            b = "\n".join(f"{it.get('item_no', '')}. {it.get('name', '')}" for it in candidate)
            cand_text = f"{h}\n\n{b}"

            if buffer and count_tokens(self.tokenizer, cand_text) > self.max_tokens:
                min_b = buffer[0].get("item_no", "")
                max_b = buffer[-1].get("item_no", "")
                r_str = f"Item No. {min_b}" if min_b == max_b else f"Item Nos. {min_b}–{max_b}"
                h_b = f"[Notification No. {notif_no} | {title}{rel_desc} | {r_str}]"
                b_text = "\n".join(f"{it.get('item_no', '')}. {it.get('name', '')}" for it in buffer)
                f_text = f"{h_b}\n\n{b_text}"

                metadata = {
                    "notification_number": notif_no,
                    "notification_date": notif_date,
                    "document_type": "EXEMPTION",
                    "chunk_type": "ANNEXURE_ITEM_GROUP",
                    "chunk_strategy": "annexure_group",
                    "effective_date": eff_date,
                    "target_notification": None,
                    "schedule": None,
                    "schedule_rate_raw": "Nil",
                    "tax_treatment": "EXEMPT",
                    "serial_numbers": [rel_sno] if rel_sno else [],
                    "classification_raw": None,
                    "normalized_hsn": [],
                    "operation_type": None,
                    "source_page_start": p_start,
                    "source_page_end": p_end,
                    "raw_legal_text": b_text,
                    "inherited_context_header": h_b,
                }
                chunks.append(NotificationChunk(f"chunk-{clean_id}-{c_idx:04d}", f_text, count_tokens(self.tokenizer, f_text), metadata))
                c_idx += 1
                buffer = [item]
            else:
                buffer = candidate

        if buffer:
            min_b = buffer[0].get("item_no", "")
            max_b = buffer[-1].get("item_no", "")
            r_str = f"Item No. {min_b}" if min_b == max_b else f"Item Nos. {min_b}–{max_b}"
            h_b = f"[Notification No. {notif_no} | {title}{rel_desc} | {r_str}]"
            b_text = "\n".join(f"{it.get('item_no', '')}. {it.get('name', '')}" for it in buffer)
            f_text = f"{h_b}\n\n{b_text}"

            metadata = {
                "notification_number": notif_no,
                "notification_date": notif_date,
                "document_type": "EXEMPTION",
                "chunk_type": "ANNEXURE_ITEM_GROUP",
                "chunk_strategy": "annexure_group",
                "effective_date": eff_date,
                "target_notification": None,
                "schedule": None,
                "schedule_rate_raw": "Nil",
                "tax_treatment": "EXEMPT",
                "serial_numbers": [rel_sno] if rel_sno else [],
                "classification_raw": None,
                "normalized_hsn": [],
                "operation_type": None,
                "source_page_start": p_start,
                "source_page_end": p_end,
                "raw_legal_text": b_text,
                "inherited_context_header": h_b,
            }
            chunks.append(NotificationChunk(f"chunk-{clean_id}-{c_idx:04d}", f_text, count_tokens(self.tokenizer, f_text), metadata))

        return chunks

    # -------------------------------------------------------------------------
    # 3. Whole Table Substitution Notifications (e.g. 13/2025)
    # -------------------------------------------------------------------------
    def _chunk_table_substitution_notification(self, norm: Dict[str, Any], clean_id: str) -> List[NotificationChunk]:
        chunks: List[NotificationChunk] = []
        notif_no = norm.get("notification_number", "")
        notif_date = norm.get("notification_date")
        eff_date = norm.get("document_effective_date")

        ops = norm.get("amendment_operations", [])
        if not ops:
            return []

        op = ops[0]
        target_notif = op.get("target_notification")
        table_data = op.get("table_data", [])
        p_start = op.get("page_start", 1)
        p_end = op.get("page_end", 1)

        parent_context = f"In notification No. {target_notif}, for the Table and the entries relating thereto, the following shall be substituted:"

        chunk_counter = 1
        for row in table_data:
            r_start = row.get("page_start", p_start)
            r_end = row.get("page_end", p_end)
            row_chunks = self._emit_single_table_row_chunks(
                clean_id, chunk_counter, notif_no, notif_date, eff_date,
                target_notif, parent_context, row, r_start, r_end
            )
            chunks.extend(row_chunks)
            chunk_counter += len(row_chunks)

        return chunks

    def _emit_single_table_row_chunks(
        self, clean_id: str, start_index: int, notif_no: str, notif_date: Optional[str],
        eff_date: Optional[str], target_notif: Optional[str], parent_context: str,
        row: Dict[str, Any], p_start: int, p_end: int
    ) -> List[NotificationChunk]:
        sno = row.get("serial_no", "")
        hsn_raw = row.get("classification_raw", "")
        rate_raw = row.get("rate_raw", "2.5 %")
        desc = row.get("description", "")
        line = f"{sno}.\t{hsn_raw}\t{desc}\t{rate_raw}"

        sno_str = f"S. No. {sno}" if sno else "S. No. -"
        hsn_str = f"HSN {hsn_raw}" if hsn_raw else "HSN -"
        header = f"[Notification No. {notif_no} | Amending Notification No. {target_notif} | Table Substitution (Rate: {rate_raw}) | {sno_str} | {hsn_str}]"
        full_text = f"{header}\n\n{parent_context}\n\n{line}"
        tokens = count_tokens(self.tokenizer, full_text)

        metadata = {
            "notification_number": notif_no,
            "notification_date": notif_date,
            "document_type": "AMENDMENT",
            "chunk_type": "AMENDMENT_TABLE_ROW",
            "chunk_strategy": "table_row",
            "effective_date": eff_date,
            "target_notification": target_notif,
            "schedule": "Table",
            "schedule_rate_raw": rate_raw,
            "tax_treatment": "CONCESSIONAL",
            "serial_numbers": [sno] if sno else [],
            "classification_raw": hsn_raw,
            "normalized_hsn": row.get("normalized_hsn", []),
            "operation_type": "SUBSTITUTE",
            "source_page_start": p_start,
            "source_page_end": p_end,
            "raw_legal_text": line,
            "inherited_context_header": header,
        }

        if tokens <= self.max_tokens:
            return [
                NotificationChunk(
                    chunk_id=f"chunk-{clean_id}-{start_index:04d}",
                    text=full_text,
                    token_count=tokens,
                    metadata=metadata,
                )
            ]

        # Fallback split (rare)
        h_tokens = count_tokens(self.tokenizer, f"{header}\n\n{parent_context}")
        body_limit = max(1, self.max_tokens - h_tokens)
        parts = split_large_text(line, body_limit, self.overlap_tokens, self.tokenizer)
        split_chunks = []
        for p_idx, part in enumerate(parts, 1):
            p_header = f"[Notification No. {notif_no} | Amending Notification No. {target_notif} | Table Substitution (Rate: {rate_raw}) | S. No. {sno} (Part {p_idx}/{len(parts)}) | {hsn_str}]"
            p_text = f"{p_header}\n\n{parent_context}\n\n{part}"
            s_meta = {
                **metadata,
                "chunk_type": "AMENDMENT_TABLE_ROW_SPLIT",
                "chunk_strategy": "table_row_split",
                "raw_legal_text": part,
                "inherited_context_header": p_header,
                "split_index": p_idx,
                "split_count": len(parts),
            }
            split_chunks.append(
                NotificationChunk(
                    chunk_id=f"chunk-{clean_id}-{start_index + p_idx - 1:04d}",
                    text=p_text,
                    token_count=count_tokens(self.tokenizer, p_text),
                    metadata=s_meta,
                )
            )
        return split_chunks

    # -------------------------------------------------------------------------
    # 4. Standard Amendment Notifications
    # -------------------------------------------------------------------------
    def _chunk_amendment_notification(self, norm: Dict[str, Any], clean_id: str) -> List[NotificationChunk]:
        chunks: List[NotificationChunk] = []
        notif_no = norm.get("notification_number", "")
        notif_date = norm.get("notification_date")
        doc_eff_date = norm.get("document_effective_date")
        ops = norm.get("amendment_operations", [])

        chunk_counter = 1

        for op in ops:
            op_chunks = self._chunk_single_amendment_op(
                clean_id, chunk_counter, notif_no, notif_date, doc_eff_date, op
            )
            chunks.extend(op_chunks)
            chunk_counter += len(op_chunks)

        # Attached Annexures (e.g. Annexures VII, VIII, IX in 05/2025)
        for annex in norm.get("annexures", []):
            title = annex.get("annexure_title", "")
            raw_full = annex.get("raw_legal_text", "")
            header = f"[Notification No. {notif_no} | Form Declaration: {title} | Status: Attached Form]"
            full_text = f"{header}\n\n{raw_full}"
            tokens = count_tokens(self.tokenizer, full_text)

            if tokens <= self.max_tokens:
                metadata = {
                    "notification_number": notif_no,
                    "notification_date": notif_date,
                    "document_type": "AMENDMENT",
                    "chunk_type": "ANNEXURE_FORM",
                    "chunk_strategy": "annexure_form",
                    "effective_date": None,  # Neutral narrative
                    "target_notification": norm.get("amendment_operations", [{}])[0].get("target_notification"),
                    "schedule": None,
                    "schedule_rate_raw": None,
                    "tax_treatment": None,
                    "serial_numbers": [],
                    "classification_raw": None,
                    "normalized_hsn": [],
                    "operation_type": "INSERT",
                    "source_page_start": annex.get("page_start", 1),
                    "source_page_end": annex.get("page_end", 1),
                    "raw_legal_text": raw_full,
                    "inherited_context_header": header,
                }
                chunks.append(NotificationChunk(f"chunk-{clean_id}-{chunk_counter:04d}", full_text, tokens, metadata))
                chunk_counter += 1
            else:
                h_tokens = count_tokens(self.tokenizer, header)
                body_limit = max(1, self.max_tokens - h_tokens)
                parts = split_large_text(raw_full, body_limit, self.overlap_tokens, self.tokenizer)
                for p_idx, part in enumerate(parts, 1):
                    p_text = f"{header}\n\n{part}"
                    metadata = {
                        "notification_number": notif_no,
                        "notification_date": notif_date,
                        "document_type": "AMENDMENT",
                        "chunk_type": "ANNEXURE_FORM_SPLIT",
                        "chunk_strategy": "annexure_token_split",
                        "effective_date": None,
                        "target_notification": norm.get("amendment_operations", [{}])[0].get("target_notification"),
                        "schedule": None,
                        "schedule_rate_raw": None,
                        "tax_treatment": None,
                        "serial_numbers": [],
                        "classification_raw": None,
                        "normalized_hsn": [],
                        "operation_type": "INSERT",
                        "source_page_start": annex.get("page_start", 1),
                        "source_page_end": annex.get("page_end", 1),
                        "raw_legal_text": part,
                        "inherited_context_header": header,
                        "split_index": p_idx,
                        "split_count": len(parts),
                    }
                    chunks.append(NotificationChunk(f"chunk-{clean_id}-{chunk_counter:04d}", p_text, count_tokens(self.tokenizer, p_text), metadata))
                    chunk_counter += 1

        return chunks

    def _chunk_single_amendment_op(
        self, clean_id: str, index: int, notif_no: str, notif_date: Optional[str],
        doc_eff_date: Optional[str], op: Dict[str, Any]
    ) -> List[NotificationChunk]:
        raw_text = op.get("raw_legal_text", "").strip()
        op_type = op.get("operation_type", "AMENDMENT")
        target_notif = op.get("target_notification")
        eff_date = op.get("effective_date") or doc_eff_date
        sno = op.get("target_serial_no")
        col = op.get("target_column")
        sched = op.get("target_schedule")
        clause = op.get("target_clause_item")
        scope_grp = op.get("scope_group_number")

        target_parts = []
        if sched:
            target_parts.append(sched)
        if sno:
            target_parts.append(f"S. No. {sno}")
        if col:
            target_parts.append(col)
        if clause:
            target_parts.append(clause)
        target_desc = ", ".join(target_parts) if target_parts else "General Provision"

        eff_str = f"Effective: {eff_date}" if eff_date else "Effective: Unspecified"
        header = f"[Notification No. {notif_no} | Amending Notification No. {target_notif} | Operation: {op_type} | Target: {target_desc} | {eff_str}]"

        full_text = f"{header}\n\n{raw_text}"
        tokens = count_tokens(self.tokenizer, full_text)

        metadata_base = {
            "notification_number": notif_no,
            "notification_date": notif_date,
            "document_type": "AMENDMENT",
            "effective_date": eff_date,
            "target_notification": target_notif,
            "schedule": sched,
            "schedule_rate_raw": None,
            "tax_treatment": "AMENDMENT",
            "serial_numbers": [sno] if sno else [],
            "classification_raw": None,
            "normalized_hsn": [],
            "operation_type": op_type,
            "source_page_start": op.get("page_start", 1),
            "source_page_end": op.get("page_end", 1),
            "scope_group_number": scope_grp,
            "inherited_context_header": header,
        }

        if tokens <= self.max_tokens:
            meta = {
                **metadata_base,
                "chunk_type": "AMENDMENT_OPERATION",
                "chunk_strategy": "amendment_operation",
                "raw_legal_text": raw_text,
            }
            return [NotificationChunk(f"chunk-{clean_id}-{index:04d}", full_text, tokens, meta)]

        h_tokens = count_tokens(self.tokenizer, header)
        body_limit = max(1, self.max_tokens - h_tokens)
        parts = split_large_text(raw_text, body_limit, self.overlap_tokens, self.tokenizer)

        split_chunks = []
        for p_idx, part in enumerate(parts, 1):
            p_text = f"{header}\n\n{part}"
            p_tokens = count_tokens(self.tokenizer, p_text)
            meta = {
                **metadata_base,
                "chunk_type": "AMENDMENT_OPERATION_SPLIT",
                "chunk_strategy": "amendment_token_split",
                "raw_legal_text": part,
                "split_index": p_idx,
                "split_count": len(parts),
            }
            split_chunks.append(
                NotificationChunk(f"chunk-{clean_id}-{index + p_idx - 1:04d}", p_text, p_tokens, meta)
            )

        return split_chunks

    # -------------------------------------------------------------------------
    # 5. Explanations Chunking
    # -------------------------------------------------------------------------
    def _chunk_explanation(
        self, clean_id: str, start_index: int, notif_no: str, notif_date: Optional[str],
        eff_date: Optional[str], exp: Dict[str, Any]
    ) -> List[NotificationChunk]:
        raw_text = exp.get("raw_text", "").strip()
        title = exp.get("title", "General Explanation")
        p_start = exp.get("page_start", 1)
        p_end = exp.get("page_end", 1)

        header = f"[Notification No. {notif_no} | {title}]"
        full_text = f"{header}\n\n{raw_text}"
        tokens = count_tokens(self.tokenizer, full_text)

        metadata_base = {
            "notification_number": notif_no,
            "notification_date": notif_date,
            "document_type": "EXPLANATION",
            "effective_date": eff_date,
            "target_notification": None,
            "schedule": None,
            "schedule_rate_raw": None,
            "tax_treatment": None,
            "serial_numbers": [],
            "classification_raw": None,
            "normalized_hsn": [],
            "operation_type": None,
            "source_page_start": p_start,
            "source_page_end": p_end,
            "inherited_context_header": header,
        }

        if tokens <= self.max_tokens:
            meta = {
                **metadata_base,
                "chunk_type": "EXPLANATION",
                "chunk_strategy": "explanation",
                "raw_legal_text": raw_text,
            }
            return [NotificationChunk(f"chunk-{clean_id}-{start_index:04d}", full_text, tokens, meta)]

        h_tokens = count_tokens(self.tokenizer, header)
        body_limit = max(1, self.max_tokens - h_tokens)
        parts = split_large_text(raw_text, body_limit, self.overlap_tokens, self.tokenizer)

        split_chunks = []
        for p_idx, part in enumerate(parts, 1):
            p_text = f"{header}\n\n{part}"
            meta = {
                **metadata_base,
                "chunk_type": "EXPLANATION_SPLIT",
                "chunk_strategy": "explanation_token_split",
                "raw_legal_text": part,
                "split_index": p_idx,
                "split_count": len(parts),
            }
            split_chunks.append(
                NotificationChunk(
                    f"chunk-{clean_id}-{start_index + p_idx - 1:04d}",
                    p_text,
                    count_tokens(self.tokenizer, p_text),
                    meta,
                )
            )
        return split_chunks


def chunk_notifications_batch(
    normalized_dir: Path = Path("data/notifications/normalized"),
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    tokenizer: Optional[Any] = None,
) -> Tuple[List[NotificationChunk], Dict[str, Any]]:
    """Batch process all normalized JSON files and generate chunks with summary metrics."""
    chunker = NotificationChunker(max_tokens=max_tokens, overlap_tokens=overlap_tokens, tokenizer=tokenizer)
    all_chunks: List[NotificationChunk] = []

    files = sorted(normalized_dir.glob("*.json"))
    per_file_counts: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    by_strategy: Dict[str, int] = {}
    split_units: List[Dict[str, Any]] = []

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            norm_data = json.load(f)

        chunks = chunker.chunk_notification(norm_data)
        all_chunks.extend(chunks)
        per_file_counts[file_path.name] = len(chunks)

        for ch in chunks:
            c_type = ch.metadata.get("chunk_type", "UNKNOWN")
            c_strat = ch.metadata.get("chunk_strategy", "UNKNOWN")
            by_type[c_type] = by_type.get(c_type, 0) + 1
            by_strategy[c_strat] = by_strategy.get(c_strat, 0) + 1

            if "split" in c_strat:
                split_units.append({
                    "chunk_id": ch.chunk_id,
                    "file": file_path.name,
                    "strategy": c_strat,
                    "tokens": ch.token_count,
                    "split_index": ch.metadata.get("split_index"),
                    "split_count": ch.metadata.get("split_count"),
                })

    token_counts = [ch.token_count for ch in all_chunks]
    oversized = [ch for ch in all_chunks if ch.token_count > max_tokens]
    zero_tokens = [ch for ch in all_chunks if ch.token_count == 0]

    report = {
        "total_chunks": len(all_chunks),
        "min_tokens": min(token_counts) if token_counts else 0,
        "avg_tokens": round(mean(token_counts), 2) if token_counts else 0.0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "oversized_chunks_count": len(oversized),
        "zero_token_chunks_count": len(zero_tokens),
        "per_file_chunk_counts": per_file_counts,
        "chunks_by_type": by_type,
        "chunks_by_strategy": by_strategy,
        "split_units_count": len(split_units),
        "split_units": split_units,
    }

    return all_chunks, report
