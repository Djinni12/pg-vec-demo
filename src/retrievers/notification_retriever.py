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


def _format_notification_chunk(
    r: tuple,
    rank: int = 1,
    score: float | None = None,
) -> dict[str, Any]:
    """Normalize a database row from notification_chunks into a standard chunk dictionary."""
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
    norm_hsn = r[10] or []
    op_type = r[11]
    page_start = r[12]
    page_end = r[13]
    content = r[14]
    tokens = r[15]
    meta = r[16] or {}
    distance = float(r[17])
    final_score = score if score is not None else round(max(0.0, 1.0 - distance), 4)

    ref_parts = [f"Notification No. {notif_no}"]
    if eff_date:
        ref_parts.append(f"Effective: {eff_date}")
    if sched:
        ref_parts.append(sched)
    if serials:
        ref_parts.append(f"S. No. {', '.join(str(s) for s in serials)}")

    title = f"Notification {notif_no}"
    if target_notif:
        title += f" (Amends {target_notif})"
    if op_type:
        title += f" [{op_type}]"

    return {
        "rank": rank,
        "document_type": "notification",
        "chunk_id": chunk_id,
        "notification_number": notif_no,
        "reference": " | ".join(ref_parts),
        "title": title,
        "content": content,
        "snippet": _snippet(content),
        "score": final_score,
        "reranker_score": final_score,
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
            "normalized_hsn": norm_hsn,
            "operation_type": op_type,
            "source_page_start": page_start,
            "source_page_end": page_end,
            "token_count": tokens,
            **meta,
        },
    }


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
                        chunk_id, notification_number, notification_date, effective_date,
                        target_notification, schedule, schedule_rate_raw, tax_treatment,
                        serial_numbers, classification_raw, normalized_hsn, operation_type,
                        source_page_start, source_page_end, content, token_count, metadata,
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
                        chunk_id, notification_number, notification_date, effective_date,
                        target_notification, schedule, schedule_rate_raw, tax_treatment,
                        serial_numbers, classification_raw, normalized_hsn, operation_type,
                        source_page_start, source_page_end, content, token_count, metadata,
                        embedding <=> %s AS distance
                    FROM notification_chunks
                    WHERE chunk_id != ALL(%s)
                    ORDER BY distance ASC
                    LIMIT %s
                """
                cur.execute(sql_vec, (query_embedding, exclude_ids, remaining))
                rows.extend(cur.fetchall())

            for rank, r in enumerate(rows, 1):
                results.append(_format_notification_chunk(r, rank=rank))

    return results[:top_k]


def retrieve_supporting_notifications(
    query: str,
    *,
    support_metadata: dict[str, Any] | None = None,
    top_k: int = 2,
    db_url: str | None = None,
    model=None,
    threshold: float = 0.48,
) -> list[dict[str, Any]]:
    """Retrieve top materially relevant supporting notification chunks via vector search & metadata boosting.

    Uses BGE-M3 query embedding to search over notification_chunks in pgvector,
    boosts candidates matching target notification, HSN, serial number, and amending operations,
    and discards non-material chunks.
    """
    if not query or not query.strip():
        return []

    meta = support_metadata or {}
    target_notif_raw = meta.get("target_notification") or meta.get("notification_no") or ""
    extracted = extract_notification_numbers(str(target_notif_raw)) if target_notif_raw else []
    if not extracted:
        extracted = extract_notification_numbers(query)
    clean_target = extracted[0] if extracted else None

    hsn_code = str(meta.get("hsn_code") or meta.get("code") or "").strip()
    hsn_4 = hsn_code[:4] if len(hsn_code) >= 4 else hsn_code
    serial_no = str(meta.get("serial_no") or "").strip().rstrip(".")

    # Embed query using BGE-M3
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
    candidate_rows_by_id: dict[str, tuple] = {}

    with psycopg.connect(conn_url) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            # 1. Global semantic vector search (top 20)
            sql_vec = """
                SELECT
                    chunk_id, notification_number, notification_date, effective_date,
                    target_notification, schedule, schedule_rate_raw, tax_treatment,
                    serial_numbers, classification_raw, normalized_hsn, operation_type,
                    source_page_start, source_page_end, content, token_count, metadata,
                    embedding <=> %s AS distance
                FROM notification_chunks
                ORDER BY distance ASC
                LIMIT 20
            """
            cur.execute(sql_vec, (query_embedding,))
            for r in cur.fetchall():
                candidate_rows_by_id[r[0]] = r

            # 2. Scoped search if target notification exists
            if clean_target:
                target_pat = f"%{clean_target}%"
                sql_target = """
                    SELECT
                        chunk_id, notification_number, notification_date, effective_date,
                        target_notification, schedule, schedule_rate_raw, tax_treatment,
                        serial_numbers, classification_raw, normalized_hsn, operation_type,
                        source_page_start, source_page_end, content, token_count, metadata,
                        embedding <=> %s AS distance
                    FROM notification_chunks
                    WHERE notification_number ILIKE %s OR target_notification ILIKE %s
                    ORDER BY distance ASC
                    LIMIT 10
                """
                cur.execute(sql_target, (query_embedding, target_pat, target_pat))
                for r in cur.fetchall():
                    if r[0] not in candidate_rows_by_id:
                        candidate_rows_by_id[r[0]] = r

    # 3. Score and filter candidates
    scored_candidates = []
    for chunk_id, r in candidate_rows_by_id.items():
        notif_no = r[1] or ""
        target_notif = r[4] or ""
        serials = [str(s).rstrip(".") for s in (r[8] or [])]
        norm_hsns = [str(h) for h in (r[10] or [])]
        op_type = (r[11] or "").upper()
        tax_treat = (r[7] or "").upper()
        distance = float(r[17])
        base_similarity = max(0.0, 1.0 - distance)

        has_target_match = False
        if clean_target:
            if clean_target in notif_no or clean_target in target_notif:
                has_target_match = True

        has_hsn_match = False
        if hsn_code:
            if hsn_code in norm_hsns or (hsn_4 and any(h.startswith(hsn_4) for h in norm_hsns)):
                has_hsn_match = True

        has_serial_match = False
        if serial_no and serial_no in serials:
            has_serial_match = True

        is_amending_op = op_type in ("SUBSTITUTE", "AMENDMENT", "INSERT", "OMIT", "OMISSION")
        is_conditional = tax_treat in ("EXEMPTION", "CONDITIONAL", "NIL")

        # Materiality filter
        is_material = (
            (has_target_match and (has_hsn_match or has_serial_match or is_amending_op))
            or (has_hsn_match and base_similarity >= 0.45)
            or (has_target_match and base_similarity >= 0.48)
            or (base_similarity >= threshold)
        )

        min_allowed = min(threshold, 0.45) if (has_hsn_match or has_target_match) else threshold
        if not is_material or base_similarity < min_allowed:
            continue

        # Metadata-aware boosting
        boosted_score = base_similarity
        if has_target_match:
            boosted_score += 0.15
        if has_hsn_match:
            boosted_score += 0.10
        if has_serial_match:
            boosted_score += 0.10
        if is_amending_op:
            boosted_score += 0.08
        if is_conditional:
            boosted_score += 0.05

        scored_candidates.append({
            "row": r,
            "boosted_score": round(boosted_score, 4),
            "base_similarity": round(base_similarity, 4),
            "is_amending": is_amending_op,
            "has_hsn_match": has_hsn_match,
        })

    # Sort descending by boosted_score
    scored_candidates.sort(key=lambda x: x["boosted_score"], reverse=True)

    results = []
    for rank, item in enumerate(scored_candidates[:top_k], 1):
        chunk_dict = _format_notification_chunk(
            item["row"],
            rank=rank,
            score=min(1.0, item["boosted_score"]),
        )
        chunk_dict["source_metadata"]["boosted_score"] = item["boosted_score"]
        chunk_dict["source_metadata"]["base_similarity"] = item["base_similarity"]
        chunk_dict["source_metadata"]["is_amending"] = item["is_amending"]
        results.append(chunk_dict)

    return results
