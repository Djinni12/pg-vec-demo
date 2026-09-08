"""Structured GST rate retriever querying official 2025 PDF rates from PostgreSQL.

Retrieves rates strictly from gst_rates_2025 without CSV fallback or LLM rate guessing.
Supports exact HSN code lookups and description-based fuzzy search.
"""

from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv
import psycopg

from src.parsers.pdf_rate_parser import clean_text, normalize_digits

load_dotenv()

DEFAULT_DATABASE_URL = "dbname=hybrid_rag user=postgres password=postgres host=localhost port=5432"


def database_url() -> str:
    """Retrieve database connection URL."""
    return os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL


def extract_query_codes(query: str) -> list[str]:
    """Extract candidate 2-8 digit HSN/tariff codes from the user query."""
    if not query:
        return []

    # Clean text and replace spaces inside numbers like '0101 21 00'
    cleaned = re.sub(r"[\r\n\t]+", " ", query)
    cleaned = re.sub(r"\b(\d{4})\s+(\d{2})\s+(\d{2})\b", r"\1\2\3", cleaned)
    cleaned = re.sub(r"\b(\d{4})\s+(\d{2})\b", r"\1\2", cleaned)

    codes: list[str] = []

    # Priority 1: Explicit labels like HSN 8471, Heading 6815, Chapter 04
    for m in re.finditer(r"\b(?:hsn|tariff|heading|chapter|code)\s*[:#]?\s*(\d{2,8})\b", cleaned, re.IGNORECASE):
        code = m.group(1)
        if code not in codes:
            codes.append(code)

    # Priority 2: Standalone 4, 6, or 8 digit numbers
    for m in re.finditer(r"\b(\d{4,8})\b", cleaned):
        code = m.group(1)
        if code not in codes:
            codes.append(code)

    return codes


def extract_search_phrase(query: str) -> str:
    """Extract the core product or service search phrase by removing boilerplate stop words."""
    cleaned = query.strip()
    # Strip question boilerplate
    patterns = [
        r"^what\s+(?:is|are)\s+(?:the\s+)?(?:gst|tax|applicable)?\s*(?:rate|rates|slab)?\s*(?:of|on|for)?\s*",
        r"^how\s+much\s+(?:is\s+)?(?:gst|tax)\s*(?:on|for)?\s*",
        r"^tell\s+me\s+(?:the\s+)?(?:gst|tax)?\s*(?:rate|rates)?\s*(?:on|for)?\s*",
        r"^gst\s+(?:rate|rates|slab|percentage)\s*(?:on|for|of)?\s*",
        r"^rate\s+(?:of|on|for)?\s*",
        r"\b(?:under\s+)?(?:hsn|heading|code)\s*[:#]?\s*\d+\b",
        r"\b(?:in\s+india|under\s+gst|please|can\s+you\s+tell|applicable)\b",
        r"[?.,!]",
    ]
    for pat in patterns:
        cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE)

    tokens = [t for t in cleaned.split() if len(t) > 1]
    return " ".join(tokens).strip()


def exact_code_lookup(
    conn: psycopg.Connection, code: str, limit: int = 10
) -> list[dict[str, Any]]:
    """Look up exact HSN code in gst_rates_2025, returning all applicable candidates."""
    results: list[dict[str, Any]] = []
    clean_code = code.strip().replace(" ", "")

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, category, notification_number, notification_date, effective_date,
                   schedule, serial_number, hsn_code, normalized_hsn_codes,
                   description, rate, cgst_rate_pct, sgst_utgst_rate_pct, igst_rate_pct,
                   formatted_rate, compensation_cess, condition_number, condition_text,
                   source_page, source_file
            FROM gst_rates_2025
            WHERE %s = ANY(normalized_hsn_codes) OR hsn_code ILIKE %s
            ORDER BY
                CASE WHEN %s = ANY(normalized_hsn_codes) THEN 0 ELSE 1 END,
                id ASC
            LIMIT %s
            """,
            (clean_code, f"%{clean_code}%", clean_code, limit),
        )
        for row in cur.fetchall():
            cgst = float(row[11]) if row[11] is not None else None
            sgst = float(row[12]) if row[12] is not None else None
            igst = float(row[13]) if row[13] is not None else None
            results.append(
                {
                    "id": row[0],
                    "item_type": "goods" if row[1] != "cess" else "cess",
                    "category": row[1],
                    "notification_number": row[2],
                    "notification_date": row[3],
                    "effective_date": row[4],
                    "schedule": row[5],
                    "serial_no": row[6],
                    "code": row[7],
                    "hsn_code": row[7],
                    "normalized_codes": row[8],
                    "description": row[9],
                    "rate": row[10],
                    "cgst_rate_pct": cgst,
                    "sgst_utgst_rate_pct": sgst,
                    "igst_rate_pct": igst,
                    "formatted_rate": row[14],
                    "compensation_cess": row[15],
                    "condition_number": row[16],
                    "condition": row[17],
                    "source_page": row[18],
                    "source_reference": f"{row[19]} (Page {row[18]})",
                    "match_type": "exact_code",
                    "score": 1.0,
                }
            )
    return results


def text_rate_search(
    conn: psycopg.Connection, phrase: str, limit: int = 5
) -> list[dict[str, Any]]:
    """Search goods descriptions using word matching and pg_trgm similarity."""
    results: list[dict[str, Any]] = []
    phrase_clean = phrase.strip()
    if not phrase_clean:
        return results

    tokens = [t for t in phrase_clean.split() if len(t) > 1]

    with conn.cursor() as cur:
        named_params = {
            "phrase": phrase_clean,
            "phrase_like": f"%{phrase_clean}%",
            "limit": limit,
        }
        word_clauses = []
        for i, tok in enumerate(tokens):
            p_name = f"tok_{i}"
            named_params[p_name] = f"%{tok}%"
            word_clauses.append(f"description ILIKE %({p_name})s")

        word_cond = " AND ".join(word_clauses) if word_clauses else "TRUE"

        sql = f"""
            SELECT id, category, notification_number, notification_date, effective_date,
                   schedule, serial_number, hsn_code, normalized_hsn_codes,
                   description, rate, cgst_rate_pct, sgst_utgst_rate_pct, igst_rate_pct,
                   formatted_rate, compensation_cess, condition_number, condition_text,
                   source_page, source_file,
                   similarity(description, %(phrase)s) AS sim,
                   CASE WHEN ({word_cond}) THEN 1.0 ELSE 0.0 END AS word_match
            FROM gst_rates_2025
            WHERE ({word_cond})
               OR description ILIKE %(phrase_like)s
               OR similarity(description, %(phrase)s) > 0.15
            ORDER BY word_match DESC, sim DESC, id ASC
            LIMIT %(limit)s
        """
        cur.execute(sql, named_params)

        for row in cur.fetchall():
            cgst = float(row[11]) if row[11] is not None else None
            sgst = float(row[12]) if row[12] is not None else None
            igst = float(row[13]) if row[13] is not None else None
            sim = float(row[20]) if row[20] is not None else 0.0
            word_match = float(row[21]) if row[21] is not None else 0.0
            score = round(max(sim, 0.7 if word_match > 0 else sim), 4)

            results.append(
                {
                    "id": row[0],
                    "item_type": "goods" if row[1] != "cess" else "cess",
                    "category": row[1],
                    "notification_number": row[2],
                    "notification_date": row[3],
                    "effective_date": row[4],
                    "schedule": row[5],
                    "serial_no": row[6],
                    "code": row[7],
                    "hsn_code": row[7],
                    "normalized_codes": row[8],
                    "description": row[9],
                    "rate": row[10],
                    "cgst_rate_pct": cgst,
                    "sgst_utgst_rate_pct": sgst,
                    "igst_rate_pct": igst,
                    "formatted_rate": row[14],
                    "compensation_cess": row[15],
                    "condition_number": row[16],
                    "condition": row[17],
                    "source_page": row[18],
                    "source_reference": f"{row[19]} (Page {row[18]})",
                    "match_type": "description_search",
                    "score": score,
                }
            )
    return results


def retrieve_rates(
    query: str,
    *,
    db_url: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve structured GST rate records strictly from gst_rates_2025."""
    if not query or not query.strip():
        return []

    url = db_url or database_url()
    codes = extract_query_codes(query)
    phrase = extract_search_phrase(query)

    results: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    with psycopg.connect(url) as conn:
        # 1. Exact code lookup for all extracted codes
        for code in codes:
            code_hits = exact_code_lookup(conn, code, limit=limit)
            for hit in code_hits:
                if hit["id"] not in seen_ids:
                    seen_ids.add(hit["id"])
                    results.append(hit)

        # 2. Text description fuzzy / word matching
        if len(results) < limit and phrase:
            text_hits = text_rate_search(conn, phrase, limit=limit - len(results) + 5)
            for hit in text_hits:
                if hit["id"] not in seen_ids:
                    seen_ids.add(hit["id"])
                    results.append(hit)
                if len(results) >= limit:
                    break

    return results[:limit]
