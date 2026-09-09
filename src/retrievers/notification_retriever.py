"""Notification retrieval module for GST rate and amendment notifications.

Performs vector search and metadata filtering over the 1,495 notification_chunks
ingested in PostgreSQL (BAAI/bge-m3 1024-dim embeddings).
"""

from __future__ import annotations

import os
import re
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

from src.embedders.act_embedder import MODEL_NAME as EMBEDDING_MODEL_NAME
from src.ingestors.ingest_gst import DATABASE_URL as DEFAULT_DATABASE_URL


def database_url() -> str:
    """Return configured database URL."""
    return os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL


def _snippet(content: str, max_length: int = 280) -> str:
    """Generate a readable snippet from content text."""
    if len(content) <= max_length:
        return content
    truncate_at = content.rfind(" ", 0, max_length)
    if truncate_at == -1:
        truncate_at = max_length
    return content[:truncate_at] + "..."


def extract_notification_numbers(query: str) -> list[str]:
    """Extract notification numbers such as 09/2025, 15/2025, 11/2017 from query text."""
    matches = re.findall(r"\b(\d{1,2}/\d{4})\b", query)
    canonical = []
    for m in matches:
        parts = m.split("/")
        c = f"{parts[0].zfill(2)}/{parts[1]}"
        if c not in canonical:
            canonical.append(c)
    return canonical


def retrieve_notifications(
    query: str,
    top_k: int = 5,
    *,
    db_url: str | None = None,
    model=None,
) -> list[dict[str, Any]]:
    """Retrieve top-k relevant notification chunks using vector search and metadata matching."""
    if not query or not query.strip():
        return []

    target_notifs = extract_notification_numbers(query)
    target_pattern = f"%{target_notifs[0]}%" if target_notifs else None

    # Embed query if model available
    query_embedding = None
    if model is not None:
        if hasattr(model, "encode"):
            query_embedding = model.encode(query, normalize_embeddings=True)
        else:
            from src.retrievers.legal_dense_retriever import embed_query
            query_embedding = embed_query(query, model=model)
    else:
        from src.retrievers.legal_dense_retriever import embed_query
        query_embedding = embed_query(query)

    conn_url = db_url or database_url()
    results: list[dict[str, Any]] = []

    with psycopg.connect(conn_url) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            rows = []
            if target_pattern:
                sql = """
                    SELECT
                        chunk_id,
                        notification_number,
                        notification_date,
                        effective_date,
                        target_notification,
                        schedule,
                        schedule_rate_raw,
                        tax_treatment,
                        serial_numbers,
                        classification_raw,
                        operation_type,
                        source_page_start,
                        source_page_end,
                        content,
                        token_count,
                        metadata,
                        embedding <=> %s AS distance
                    FROM notification_chunks
                    WHERE notification_number ILIKE %s OR target_notification ILIKE %s
                    ORDER BY distance ASC
                    LIMIT %s
                """
                cur.execute(sql, (query_embedding, target_pattern, target_pattern, top_k))
                rows = cur.fetchall()

            if len(rows) < top_k:
                remaining = top_k - len(rows)
                exclude_ids = [r[0] for r in rows] if rows else [""]
                sql_vec = """
                    SELECT
                        chunk_id,
                        notification_number,
                        notification_date,
                        effective_date,
                        target_notification,
                        schedule,
                        schedule_rate_raw,
                        tax_treatment,
                        serial_numbers,
                        classification_raw,
                        operation_type,
                        source_page_start,
                        source_page_end,
                        content,
                        token_count,
                        metadata,
                        embedding <=> %s AS distance
                    FROM notification_chunks
                    WHERE chunk_id != ALL(%s)
                    ORDER BY distance ASC
                    LIMIT %s
                """
                cur.execute(sql_vec, (query_embedding, exclude_ids, remaining))
                rows.extend(cur.fetchall())

            for rank, r in enumerate(rows, 1):
                chunk_id = r[0]
                notif_no = r[1]
                notif_date = str(r[2]) if r[2] else None
                eff_date = str(r[3]) if r[3] else None
                target_notif = r[4]
                sched = r[5]
                rate_raw = r[6]
                tax_treat = r[7]
                serials = r[8] or []
                class_raw = r[9]
                op_type = r[10]
                page_start = r[11]
                page_end = r[12]
                content = r[13]
                tokens = r[14]
                meta = r[15] or {}
                distance = float(r[16])
                score = round(1.0 - distance, 4)

                ref_parts = [f"Notification No. {notif_no}"]
                if eff_date:
                    ref_parts.append(f"Effective: {eff_date}")
                if sched:
                    ref_parts.append(sched)
                if serials:
                    ref_parts.append(f"S. No. {', '.join(serials)}")

                results.append({
                    "rank": rank,
                    "document_type": "notification",
                    "chunk_id": chunk_id,
                    "notification_number": notif_no,
                    "reference": " | ".join(ref_parts),
                    "title": f"Notification {notif_no}" + (f" (Amends {target_notif})" if target_notif else ""),
                    "content": content,
                    "snippet": _snippet(content),
                    "score": score,
                    "reranker_score": score,
                    "source_metadata": {
                        "notification_number": notif_no,
                        "notification_date": notif_date,
                        "effective_date": eff_date,
                        "target_notification": target_notif,
                        "schedule": sched,
                        "schedule_rate_raw": rate_raw,
                        "tax_treatment": tax_treat,
                        "serial_numbers": serials,
                        "classification_raw": class_raw,
                        "operation_type": op_type,
                        "source_page_start": page_start,
                        "source_page_end": page_end,
                        "token_count": tokens,
                        **meta,
                    },
                })

    return results[:top_k]
