"""Background knowledge-ingestion pipeline for successful fallback web search results.

Asynchronously processes, validates, classifies, embeds, and persists verified web evidence
into the existing GST knowledge tables (gst_rates_2025, notification_chunks, act_chunks, rule_chunks, form_chunks)
and in-memory rate stores without blocking or delaying the user query response.
"""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse

import psycopg
from psycopg.types.json import Jsonb

from src.observability.trace_store import trace_store
from src.retrievers.rate_retriever import (
    append_rate_record_to_csv_and_cache,
    database_url,
    normalize_hsn_codes,
)
from src.retrievers.legal_dense_retriever import embed_query
from src.ingestors.ingest_notification_chunks import save_notification_records
from src.ingestors.ingest_act_chunks import save_act_records
from src.ingestors.ingest_rule_chunks import save_rule_records

logger = logging.getLogger(__name__)

# Persistent background daemon executor: 2 workers, non-blocking
_INGESTION_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="bg_web_ingest_",
)

# Trust gate: Authoritative domains allowed for ingestion
TRUSTED_DOMAIN_PATTERNS = [
    r"(^|\.)gst\.gov\.in$",
    r"(^|\.)cbic-gst\.gov\.in$",
    r"(^|\.)cbic\.gov\.in$",
    r"(^|\.)gstcouncil\.gov\.in$",
    r"(^|\.)gov\.in$",
    r"(^|\.)taxguru\.in$",
    r"(^|\.)cleartax\.in$",
    r"(^|\.)taxmann\.com$",
    r"(^|\.)indiafilings\.com$",
    r"(^|\.)pocketgst\.com$",
    r"(^|\.)mastersindia\.co$",
    r"(^|\.)saginfotech\.com$",
    r"(^|\.)caclubindia\.com$",
    r"(^|\.)taxscan\.in$",
    r"(^|\.)taxclue\.in$",
    r"(^|\.)taxgarden\.in$",
    r"(^|\.)vakilsearch\.com$",
]

UNTRUSTED_DOMAINS = [
    "tripadvisor", "zomato", "swiggy", "magicpin", "eatsure",
    "quora.com", "reddit.com", "facebook.com", "twitter.com", "x.com",
    "wikipedia.org", "justdial.com", "dineout.co.in", "mouthshut.com",
]


def is_trusted_domain(url_or_domain: str) -> bool:
    """Check if the given URL or domain is an authoritative source for statutory ingestion."""
    if not url_or_domain:
        return False

    raw = url_or_domain.lower().strip()
    if "://" in raw:
        try:
            domain = urllib.parse.urlparse(raw).netloc.lower()
        except Exception:
            domain = raw
    else:
        domain = raw.split("/")[0].strip()

    if ":" in domain:
        domain = domain.split(":")[0]

    for bad in UNTRUSTED_DOMAINS:
        if bad in domain:
            return False

    for pattern in TRUSTED_DOMAIN_PATTERNS:
        if re.search(pattern, domain):
            return True

    return False


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove common web noise."""
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    return t


def classify_and_extract_evidence(
    web_chunk: dict[str, Any],
    user_query: str,
    discovered_hsn: Optional[str] = None,
) -> Tuple[Optional[str], Optional[dict[str, Any]], str]:
    """Classify web search evidence and extract standardized record fields.

    Returns:
        (category, record_dict, status_or_reason)
        category can be 'rate', 'notification', 'act', 'rule', or None if skipped.
    """
    url = str(web_chunk.get("url") or web_chunk.get("reference") or "").strip()
    domain = str(web_chunk.get("domain") or "").strip()
    if not domain and url:
        try:
            domain = urllib.parse.urlparse(url).netloc
        except Exception:
            pass

    # 1. Trust gate
    if not is_trusted_domain(domain or url):
        return None, None, f"untrusted_domain: {domain or url}"

    title = _clean_text(str(web_chunk.get("title") or ""))
    snippet = _clean_text(str(web_chunk.get("snippet") or ""))
    combined_text = f"{title} {snippet}"

    # 2. Relevance check: Must contain statutory or tax concepts
    keywords = ["gst", "tax", "rate", "hsn", "sac", "notification", "section", "rule", "circular", "exempt", "schedule", "%"]
    if not any(kw in combined_text.lower() for kw in keywords):
        return None, None, "irrelevant_content: missing tax keywords"

    # Category A: Structured Tariff / Rate Record
    # Check for HSN/SAC code and rate percentage
    hsn_code = discovered_hsn
    if not hsn_code:
        # Match SAC code (99xxxx) or standard HSN (4-8 digits)
        m_sac = re.search(r"\b(?:sac|heading)?\s*[:#]?\s*(99\d{2,6})\b", combined_text, re.I)
        if m_sac:
            hsn_code = m_sac.group(1)
        else:
            m_hsn = re.search(r"\b(?:hsn|tariff|heading|code)?\s*[:#]?\s*(\d{4,8})\b", combined_text, re.I)
            if m_hsn:
                hsn_code = m_hsn.group(1)

    # Check for rate percentage (e.g. 5%, 12%, 18%, 28%, Nil, Exempt)
    rate_str = None
    m_rate = re.search(r"\b(\d+(?:\.\d+)?)\s*%\s*(?:gst|igst|tax)?\b", combined_text, re.I)
    if m_rate:
        rate_val = float(m_rate.group(1))
        # Sanity check valid GST rates
        if rate_val in (0.0, 0.1, 0.25, 1.5, 3.0, 5.0, 6.0, 9.0, 12.0, 14.0, 18.0, 28.0):
            rate_str = f"{rate_val:g}%"
    elif re.search(r"\b(nil|exempt|exempted)\b", combined_text, re.I):
        rate_str = "Nil"

    is_rate_query = bool(
        "rate" in user_query.lower()
        or "gst on" in user_query.lower()
        or "gst for" in user_query.lower()
        or "tariff" in user_query.lower()
    )

    if hsn_code and rate_str and is_rate_query:
        # Extract description
        desc = title
        # Strip trailing domain brand like "| PocketGST" or "- ClearTax"
        desc = re.sub(r"\s*[|\-–—]\s*([A-Za-z0-9\s.]+)$", "", desc).strip()
        # Clean generic search prefixes but keep essential nouns
        desc = re.sub(r"(?i)^(what is the|gst rate for|gst rate on|gst on|gst for|rate of|sac code for|hsn code for)\s*", "", desc).strip(" :-|")
        if not desc or len(desc) < 3:
            desc = snippet[:120].strip()
        # Ensure description is informative and contains the query phrase if applicable
        if "restaurant" in user_query.lower() and "restaurant" not in desc.lower():
            desc = f"Restaurant services - {desc}"

        # Parse percentage breakdown
        igst_pct = None
        cgst_pct = None
        sgst_pct = None
        if rate_str == "Nil":
            igst_pct = 0.0
            cgst_pct = 0.0
            sgst_pct = 0.0
        elif "%" in rate_str:
            val = float(rate_str.replace("%", ""))
            igst_pct = val
            cgst_pct = round(val / 2.0, 3)
            sgst_pct = round(val / 2.0, 3)

        condition_text = ""
        if re.search(r"\bwithout\s+itc\b", combined_text, re.I):
            condition_text = "Without ITC"
        elif re.search(r"\bwith\s+itc\b", combined_text, re.I):
            condition_text = "With ITC"

        category = "services" if (str(hsn_code).startswith("99") or "service" in combined_text.lower()) else "goods"
        norm_codes = sorted(list(normalize_hsn_codes(str(hsn_code))))

        # Notification reference if mentioned
        notif_match = re.search(r"\b(\d{1,2}/\d{4})\b", combined_text)
        notif_no = f"{notif_match.group(1)}-Central Tax (Rate)" if notif_match else ""

        rate_record = {
            "source_file": f"{domain} (Web Ingestion)" if domain else "Web Ingestion",
            "source_page": 1,
            "section_heading": f"CGST rates on {category} as on 22.09.2025",
            "rate_category": "CGST",
            "notification_no": notif_no,
            "notification_number": notif_no,
            "notification_date": "",
            "rate_as_on_date": "22.09.2025",
            "schedule": f"Schedule – {rate_str}",
            "serial_no": "web_ext",
            "serial_number": "web_ext",
            "hsn_code": str(hsn_code),
            "normalized_hsn_codes": norm_codes,
            "description": desc,
            "source_rate": rate_str,
            "rate": rate_str,
            "gst_rate": rate_str,
            "cgst_rate": f"{cgst_pct}%" if cgst_pct is not None else "",
            "sgst_rate": f"{sgst_pct}%" if sgst_pct is not None else "",
            "total_gst_rate": rate_str,
            "formatted_rate": f"{rate_str} (CGST {cgst_pct}% + SGST {sgst_pct}%)" if cgst_pct else rate_str,
            "compensation_cess_rate": "",
            "compensation_cess": None,
            "is_exempt": "true" if rate_str == "Nil" else "false",
            "condition": condition_text,
            "condition_number": None,
            "condition_text": condition_text,
            "footnote": "",
            "amendment_note": "",
            "raw_text": f"{hsn_code} | {desc} | {rate_str}",
            "cgst_rate_pct": cgst_pct,
            "sgst_utgst_rate_pct": sgst_pct,
            "igst_rate_pct": igst_pct,
            "category": category,
        }
        return "rate", rate_record, "valid_rate"

    # Category B: Statutory Notification or Circular
    m_notif = re.search(r"\b(?:notification\s*(?:no\.?)?\s*)?(\d{1,2}/\d{4})\b", combined_text, re.I)
    if m_notif and ("central tax" in combined_text.lower() or "notification" in combined_text.lower() or "circular" in combined_text.lower()):
        notif_num = m_notif.group(1)
        clean_num = f"{notif_num.split('/')[0].zfill(2)}/{notif_num.split('/')[1]}"
        content_hash = hashlib.sha256(snippet.encode("utf-8")).hexdigest()[:10]
        chunk_id = f"web_notif_{clean_num.replace('/', '_')}_{content_hash}"

        header = f"Notification No. {clean_num}-Central Tax (Rate), issued under GST Law\nSource: {domain}\n\n"
        full_content = header + (snippet if len(snippet) > 40 else combined_text)

        notif_record = {
            "chunk_id": chunk_id,
            "notification_number": f"{clean_num}-Central Tax (Rate)",
            "notification_date": None,
            "document_type": "notification",
            "chunk_type": "statutory_clause",
            "chunk_strategy": "web_fallback_ingestion",
            "effective_date": None,
            "target_notification": None,
            "schedule": None,
            "schedule_rate_raw": rate_str,
            "tax_treatment": "TAXABLE" if rate_str and rate_str != "Nil" else "EXEMPT",
            "serial_numbers": [],
            "classification_raw": hsn_code or "",
            "normalized_hsn": [hsn_code] if hsn_code else [],
            "operation_type": "web_ingested_clause",
            "source_page_start": 1,
            "source_page_end": 1,
            "content": full_content,
            "token_count": len(full_content.split()),
            "metadata": {
                "source": "web_fallback",
                "url": url,
                "domain": domain,
                "title": title,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        return "notification", notif_record, "valid_notification"

    # Category C: Legal Act / Section
    m_sec = re.search(r"\bsection\s+(\d+[A-Za-z]?)\b", combined_text, re.I)
    if m_sec and ("act" in combined_text.lower() or "cgst" in combined_text.lower() or "igst" in combined_text.lower()):
        sec_num = m_sec.group(1)
        content_hash = hashlib.sha256(snippet.encode("utf-8")).hexdigest()[:10]
        chunk_id = f"web_act_sec_{sec_num}_{content_hash}"
        full_content = f"Section {sec_num}. {title}\n\n{snippet}"

        act_record = {
            "chunk_id": chunk_id,
            "act_name": "Central Goods and Services Tax Act, 2017",
            "chapter": "Legal Provisions",
            "section_number": sec_num,
            "section_title": title,
            "subsection_numbers": [],
            "status": "Active",
            "content": full_content,
            "token_count": len(full_content.split()),
        }
        return "act", act_record, "valid_act"

    # Category D: Legal Rule
    m_rule = re.search(r"\brule\s+(\d+[A-Za-z]?)\b", combined_text, re.I)
    if m_rule and ("rule" in combined_text.lower() or "cgst" in combined_text.lower()):
        rule_num = m_rule.group(1)
        content_hash = hashlib.sha256(snippet.encode("utf-8")).hexdigest()[:10]
        chunk_id = f"web_rule_{rule_num}_{content_hash}"
        full_content = f"Rule {rule_num}. {title}\n\n{snippet}"

        rule_record = {
            "chunk_id": chunk_id,
            "rule_number": rule_num,
            "rule_title": title,
            "chapter": "Rules",
            "chapter_title": "Central Goods and Services Tax Rules, 2017",
            "subrule_numbers": [],
            "status": "Active",
            "content": full_content,
            "token_count": len(full_content.split()),
            "chunk_strategy": "web_fallback_ingestion",
        }
        return "rule", rule_record, "valid_rule"

    return None, None, "unclassified_web_content"


def _insert_rate_to_postgres(conn: psycopg.Connection, rate_record: dict[str, Any]) -> bool:
    """Idempotently insert a single rate record into PostgreSQL gst_rates_2025."""
    with conn.cursor() as cur:
        # Check if already exists
        hsn = rate_record["hsn_code"]
        norm_codes = rate_record["normalized_hsn_codes"]
        rate = rate_record["rate"]
        desc = rate_record["description"]

        cur.execute(
            """
            SELECT id FROM gst_rates_2025
            WHERE (%s = ANY(normalized_hsn_codes) OR hsn_code = %s)
              AND rate = %s
            LIMIT 1;
            """,
            (hsn, hsn, rate),
        )
        if cur.fetchone():
            return False  # Already exists

        insert_sql = """
            INSERT INTO gst_rates_2025 (
                category, notification_number, notification_date, effective_date,
                schedule, serial_number, hsn_code, normalized_hsn_codes,
                description, rate, cgst_rate_pct, sgst_utgst_rate_pct, igst_rate_pct,
                formatted_rate, compensation_cess, condition_number, condition_text,
                source_page, source_file
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            );
        """
        cur.execute(
            insert_sql,
            (
                rate_record["category"],
                rate_record.get("notification_number"),
                rate_record.get("notification_date"),
                rate_record.get("effective_date"),
                rate_record.get("schedule"),
                rate_record.get("serial_number"),
                rate_record.get("hsn_code"),
                norm_codes,
                rate_record["description"],
                rate_record["rate"],
                rate_record.get("cgst_rate_pct"),
                rate_record.get("sgst_utgst_rate_pct"),
                rate_record.get("igst_rate_pct"),
                rate_record["formatted_rate"],
                rate_record.get("compensation_cess"),
                rate_record.get("condition_number"),
                rate_record.get("condition_text"),
                rate_record.get("source_page", 1),
                rate_record.get("source_file", "Web Ingestion"),
            ),
        )
        conn.commit()
        return True


def execute_web_ingestion_job(
    execution_id: str,
    user_query: str,
    web_results: list[dict[str, Any]],
    discovered_hsn: Optional[str] = None,
    missing_info_context: Optional[str] = None,
    db_url: Optional[str] = None,
) -> dict[str, Any]:
    """Execute classification, validation, embedding, and storage of web results.

    Always run inside background thread; isolated from main request lifecycle.
    """
    t0 = time.perf_counter()
    if not web_results:
        return {"status": "skipped", "reason": "empty_web_results"}

    if execution_id:
        trace_store.record_background_ingestion(execution_id, status="queued")

    stored_count = 0
    skipped_count = 0
    details: list[dict[str, Any]] = []
    target_category = None
    target_table = None

    url = db_url or database_url()

    for idx, chunk in enumerate(web_results):
        try:
            category, record, reason = classify_and_extract_evidence(
                chunk,
                user_query=user_query,
                discovered_hsn=discovered_hsn,
            )

            if not category or not record:
                skipped_count += 1
                details.append({
                    "index": idx,
                    "status": "skipped",
                    "reason": reason,
                    "title": chunk.get("title", "")[:50],
                })
                continue

            target_category = category

            # 1. Rate Ingestion Path
            if category == "rate":
                target_table = "gst_rates_2025"
                # A. Append to CSV & in-memory cache
                appended_csv = append_rate_record_to_csv_and_cache(record)

                # B. Insert into PostgreSQL gst_rates_2025
                inserted_pg = False
                try:
                    with psycopg.connect(url, connect_timeout=5) as conn:
                        inserted_pg = _insert_rate_to_postgres(conn, record)
                except Exception as db_err:
                    logger.warning(f"Database rate insert failed (CSV still active): {db_err}")

                if appended_csv or inserted_pg:
                    stored_count += 1
                    details.append({
                        "index": idx,
                        "status": "stored",
                        "table": "gst_rates_2025",
                        "hsn_code": record.get("hsn_code"),
                        "rate": record.get("rate"),
                        "csv_appended": appended_csv,
                        "pg_inserted": inserted_pg,
                    })
                else:
                    skipped_count += 1
                    details.append({
                        "index": idx,
                        "status": "skipped",
                        "reason": "already_exists",
                        "hsn_code": record.get("hsn_code"),
                    })

            # 2. Statutory Notification Path
            elif category == "notification":
                target_table = "notification_chunks"
                # Compute 1024-dim BGE-M3 embedding
                emb = embed_query(record["content"])
                record["embedding"] = emb
                record_for_db = dict(record)
                record_for_db["metadata"] = Jsonb(record_for_db["metadata"])

                try:
                    with psycopg.connect(url, connect_timeout=5) as conn:
                        save_notification_records(conn, [record_for_db])
                        conn.commit()
                    stored_count += 1
                    details.append({
                        "index": idx,
                        "status": "stored",
                        "table": "notification_chunks",
                        "chunk_id": record.get("chunk_id"),
                        "notification_number": record.get("notification_number"),
                    })
                except Exception as db_err:
                    skipped_count += 1
                    details.append({
                        "index": idx,
                        "status": "failed",
                        "reason": str(db_err),
                        "table": "notification_chunks",
                    })

            # 3. Legal Act Path
            elif category == "act":
                target_table = "act_chunks"
                emb = embed_query(record["content"])
                record["embedding"] = emb

                try:
                    with psycopg.connect(url, connect_timeout=5) as conn:
                        save_act_records(conn, [record])
                        conn.commit()
                    stored_count += 1
                    details.append({
                        "index": idx,
                        "status": "stored",
                        "table": "act_chunks",
                        "chunk_id": record.get("chunk_id"),
                        "section_number": record.get("section_number"),
                    })
                except Exception as db_err:
                    skipped_count += 1
                    details.append({
                        "index": idx,
                        "status": "failed",
                        "reason": str(db_err),
                        "table": "act_chunks",
                    })

            # 4. Legal Rule Path
            elif category == "rule":
                target_table = "rule_chunks"
                emb = embed_query(record["content"])
                record["embedding"] = emb

                try:
                    with psycopg.connect(url, connect_timeout=5) as conn:
                        save_rule_records(conn, [record])
                        conn.commit()
                    stored_count += 1
                    details.append({
                        "index": idx,
                        "status": "stored",
                        "table": "rule_chunks",
                        "chunk_id": record.get("chunk_id"),
                        "rule_number": record.get("rule_number"),
                    })
                except Exception as db_err:
                    skipped_count += 1
                    details.append({
                        "index": idx,
                        "status": "failed",
                        "reason": str(db_err),
                        "table": "rule_chunks",
                    })

        except Exception as exc:
            logger.error(f"Error ingesting chunk {idx}: {exc}", exc_info=True)
            skipped_count += 1
            details.append({
                "index": idx,
                "status": "failed",
                "reason": str(exc),
            })

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    final_status = "stored" if stored_count > 0 else ("skipped" if skipped_count > 0 else "empty")

    if execution_id:
        trace_store.record_background_ingestion(
            execution_id,
            status=final_status,
            target_category=target_category,
            target_table=target_table,
            records_ingested=stored_count,
            records_skipped=skipped_count,
            details=details,
            timing_ms=elapsed_ms,
        )

    return {
        "status": final_status,
        "records_ingested": stored_count,
        "records_skipped": skipped_count,
        "details": details,
        "timing_ms": elapsed_ms,
    }


def schedule_background_web_ingestion(
    execution_id: str,
    user_query: str,
    web_results: list[dict[str, Any]],
    discovered_hsn: Optional[str] = None,
    missing_info_context: Optional[str] = None,
    db_url: Optional[str] = None,
) -> concurrent.futures.Future:
    """Non-blocking entrypoint: submits ingestion to daemon worker and returns immediately.

    Never blocks, awaits, or raises to the caller.
    """
    logger.info(
        f"Scheduling background web ingestion for query='{user_query}' "
        f"({len(web_results)} web results, execution_id={execution_id})"
    )
    return _INGESTION_EXECUTOR.submit(
        execute_web_ingestion_job,
        execution_id=execution_id,
        user_query=user_query,
        web_results=web_results,
        discovered_hsn=discovered_hsn,
        missing_info_context=missing_info_context,
        db_url=db_url,
    )
