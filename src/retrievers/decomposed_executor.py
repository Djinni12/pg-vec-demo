"""Execution engine for decomposed retrieval subqueries.

Executes subqueries independently and concurrently using asyncio.to_thread
and asyncio.gather, attaches provenance metadata, merges and deduplicates results,
and controls the final evidence size while preserving error isolation.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import time
from typing import Any, Optional

from src.retrieval_inspector import DEFAULT_TOP_K, LoadedModels, inspect_retrieval
from src.retrievers.rate_retriever import retrieve_rates

logger = logging.getLogger(__name__)

# Module-level cache for heavy retrieval models when not passed explicitly
_CACHED_MODELS: Optional[LoadedModels] = None


def get_cached_models() -> LoadedModels:
    """Lazily load and cache models for decomposed legal retrieval."""
    global _CACHED_MODELS
    if _CACHED_MODELS is None:
        from src.retrieval_inspector import load_models
        _CACHED_MODELS = load_models()
    return _CACHED_MODELS


def get_chunk_dedup_key(chunk: dict[str, Any]) -> str:
    """Generate a stable deduplication identifier for legal chunks.

    Prioritizes chunk_id, document_id + chunk_index, or reference metadata
    over raw text.
    """
    if chunk.get("chunk_id") is not None:
        return f"id:{chunk['chunk_id']}"
    if chunk.get("document_id") is not None and chunk.get("chunk_index") is not None:
        return f"doc:{chunk['document_id']}_{chunk['chunk_index']}"
    doc_type = str(chunk.get("document_type") or "").strip().lower()
    ref = str(chunk.get("reference") or "").strip().lower()
    idx = chunk.get("chunk_index")
    if ref:
        return f"ref:{doc_type}:{ref}:{idx if idx is not None else ''}"
    content = (chunk.get("content") or chunk.get("snippet") or "").strip()
    return f"content:{content[:200]}"


def get_rate_dedup_key(rate: dict[str, Any]) -> str:
    """Generate a stable deduplication identifier for structured rate items.

    Prioritizes id or tariff classification tuple (code, serial_no, notif, sched, rate).
    """
    if rate.get("id") is not None:
        return f"id:{rate['id']}"
    code = str(rate.get("code") or rate.get("hsn_code") or "").strip().lower()
    serial = str(rate.get("serial_no") or "").strip().lower()
    notif = str(rate.get("notification_no") or rate.get("notification_number") or "").strip().lower()
    sched = str(rate.get("schedule") or "").strip().lower()
    source_rate = str(rate.get("source_rate") or rate.get("total_gst_rate") or "").strip().lower()
    if code or serial or notif:
        return f"rate:{code}:{serial}:{notif}:{sched}:{source_rate}"
    desc = str(rate.get("description") or "").strip().lower()
    return f"desc:{desc[:100]}"


def attach_provenance(
    item: dict[str, Any],
    subquery_id: str,
    query_text: str,
    query_type: str,
) -> dict[str, Any]:
    """Attach subquery provenance to an item without modifying existing fields."""
    tagged = dict(item)
    tagged["subquery_id"] = subquery_id
    tagged["retrieval_subquery"] = query_text
    tagged["retrieval_subquery_type"] = query_type
    tagged["subquery_ids"] = [subquery_id]
    tagged["matched_subqueries"] = [
        {
            "subquery_id": subquery_id,
            "retrieval_subquery": query_text,
            "retrieval_subquery_type": query_type,
        }
    ]
    return tagged


def merge_and_deduplicate(
    subquery_results: list[list[dict[str, Any]]],
    item_type: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Merge results from multiple subqueries in round-robin order.

    Deduplicates using stable identifiers while aggregating provenance across
    all matching subqueries. Caps the final list to `limit` items to control
    final evidence size and ensure balanced representation across concepts.
    """
    if not subquery_results:
        return []

    max_len = max(len(items) for items in subquery_results)
    merged: list[dict[str, Any]] = []
    seen_keys: dict[str, dict[str, Any]] = {}

    for col_idx in range(max_len):
        for subq_items in subquery_results:
            if col_idx < len(subq_items):
                item = subq_items[col_idx]
                key = (
                    get_chunk_dedup_key(item)
                    if item_type == "legal"
                    else get_rate_dedup_key(item)
                )
                if key in seen_keys:
                    existing = seen_keys[key]
                    for sid in item.get("subquery_ids", []):
                        if sid not in existing.get("subquery_ids", []):
                            existing.setdefault("subquery_ids", []).append(sid)
                    for m_sub in item.get("matched_subqueries", []):
                        if m_sub not in existing.get("matched_subqueries", []):
                            existing.setdefault("matched_subqueries", []).append(m_sub)
                else:
                    seen_keys[key] = item
                    merged.append(item)

    final_results = merged[:limit]

    # Re-assign sequential ranks for legal chunks
    if item_type == "legal":
        for idx, chunk in enumerate(final_results, 1):
            chunk["rank"] = idx

    return final_results


async def _execute_single_subquery(
    subquery: dict[str, Any] | Any,
    subquery_id: str,
    *,
    models: LoadedModels | None,
    db_url: str | None,
    top_k: int,
    rate_limit: int,
) -> dict[str, Any]:
    """Execute a single retrieval subquery in a worker thread with error isolation."""
    if isinstance(subquery, dict):
        q_type = str(subquery.get("type", "legal")).strip().lower()
        q_text = str(subquery.get("query", "")).strip()
    else:
        q_type = str(getattr(subquery, "type", "legal")).strip().lower()
        q_text = str(getattr(subquery, "query", "")).strip()

    t0 = time.perf_counter()
    if not q_text:
        return {
            "subquery_id": subquery_id,
            "type": q_type,
            "query": q_text,
            "status": "empty",
            "results": [],
            "timing_ms": 0.0,
            "error": None,
        }

    try:
        if q_type == "rate":
            rates = await asyncio.to_thread(
                retrieve_rates,
                q_text,
                db_url=db_url,
                limit=rate_limit,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
            tagged = [
                attach_provenance(r, subquery_id, q_text, "rate")
                for r in rates
            ]
            return {
                "subquery_id": subquery_id,
                "type": "rate",
                "query": q_text,
                "status": "success",
                "results": tagged,
                "timing_ms": elapsed_ms,
                "error": None,
            }
        else:
            # Legal retrieval
            eff_models = models or get_cached_models()
            data = await asyncio.to_thread(
                inspect_retrieval,
                q_text,
                top_k=top_k,
                models=eff_models,
                db_url=db_url,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
            raw_results = data.get("results", []) if isinstance(data, dict) else []
            tagged = [
                attach_provenance(c, subquery_id, q_text, "legal")
                for c in raw_results
            ]
            return {
                "subquery_id": subquery_id,
                "type": "legal",
                "query": q_text,
                "status": "success",
                "results": tagged,
                "timing_ms": elapsed_ms,
                "raw_data": data,
                "error": None,
            }
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        logger.warning(
            f"Error in decomposed subquery [{subquery_id}] ({q_type}): {exc}"
        )
        return {
            "subquery_id": subquery_id,
            "type": q_type,
            "query": q_text,
            "status": "error",
            "results": [],
            "timing_ms": elapsed_ms,
            "error": str(exc),
        }


async def execute_decomposed_subqueries_async(
    subqueries: list[dict[str, Any] | Any],
    *,
    models: LoadedModels | None = None,
    db_url: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    rate_limit: int = 10,
) -> dict[str, Any]:
    """Asynchronously execute retrieval subqueries concurrently using asyncio.gather.

    Dispatches by subquery type ('legal' -> inspect_retrieval, 'rate' -> retrieve_rates).
    Performs per-subquery error isolation, tags provenance, merges, deduplicates,
    and returns bounded evidence sets.
    """
    t0 = time.perf_counter()
    if not subqueries:
        return {
            "legal_results": [],
            "rate_results": [],
            "tools_executed": [],
            "timings_ms": {"total": 0.0, "legal": 0.0, "rate": 0.0},
            "subquery_errors": [],
            "subquery_outputs": [],
        }

    # Assign stable subquery IDs (e.g. legal_1, legal_2, rate_1)
    type_counts: dict[str, int] = {}
    tasks = []
    for sq in subqueries:
        sq_type = (
            sq.get("type", "legal")
            if isinstance(sq, dict)
            else getattr(sq, "type", "legal")
        )
        norm_type = "rate" if "rate" in str(sq_type).lower() else "legal"
        type_counts[norm_type] = type_counts.get(norm_type, 0) + 1
        sq_id = f"{norm_type}_{type_counts[norm_type]}"
        tasks.append(
            _execute_single_subquery(
                subquery=sq,
                subquery_id=sq_id,
                models=models,
                db_url=db_url,
                top_k=top_k,
                rate_limit=rate_limit,
            )
        )

    # Execute all subqueries concurrently
    subquery_outputs = await asyncio.gather(*tasks)

    legal_candidate_lists: list[list[dict[str, Any]]] = []
    rate_candidate_lists: list[list[dict[str, Any]]] = []
    tools_executed: list[dict[str, Any]] = []
    subquery_errors: list[dict[str, Any]] = []
    legal_timing = 0.0
    rate_timing = 0.0

    for out in subquery_outputs:
        sq_type = out["type"]
        sq_id = out["subquery_id"]
        status = out["status"]
        results = out["results"]
        timing = out["timing_ms"]
        error = out["error"]

        if sq_type == "rate":
            rate_timing = max(rate_timing, timing)
            if status == "success":
                rate_candidate_lists.append(results)
            tools_executed.append({
                "tool": "retrieve_rates",
                "subquery_id": sq_id,
                "type": "rate",
                "query": out["query"],
                "results_count": len(results),
                "status": status,
                "timing_ms": timing,
                "error": error,
            })
        else:
            legal_timing = max(legal_timing, timing)
            if status == "success":
                legal_candidate_lists.append(results)
            tools_executed.append({
                "tool": "inspect_retrieval",
                "subquery_id": sq_id,
                "type": "legal",
                "query": out["query"],
                "results_count": len(results),
                "status": status,
                "timing_ms": timing,
                "error": error,
            })

        if status == "error" and error:
            subquery_errors.append({
                "subquery_id": sq_id,
                "type": sq_type,
                "query": out["query"],
                "error": error,
            })

    # Merge and deduplicate per type
    merged_legal = merge_and_deduplicate(legal_candidate_lists, "legal", top_k)
    merged_rates = merge_and_deduplicate(rate_candidate_lists, "rate", rate_limit)

    total_timing = round((time.perf_counter() - t0) * 1000, 3)

    return {
        "legal_results": merged_legal,
        "rate_results": merged_rates,
        "tools_executed": tools_executed,
        "timings_ms": {
            "total": total_timing,
            "legal": legal_timing,
            "rate": rate_timing,
        },
        "subquery_errors": subquery_errors,
        "subquery_outputs": subquery_outputs,
    }


def execute_decomposed_subqueries(
    subqueries: list[dict[str, Any] | Any],
    *,
    models: LoadedModels | None = None,
    db_url: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    rate_limit: int = 10,
) -> dict[str, Any]:
    """Synchronous entry point for executing decomposed retrieval subqueries."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Avoid nested loop conflict when called from an existing async context
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run,
                execute_decomposed_subqueries_async(
                    subqueries,
                    models=models,
                    db_url=db_url,
                    top_k=top_k,
                    rate_limit=rate_limit,
                ),
            ).result()
    else:
        return asyncio.run(
            execute_decomposed_subqueries_async(
                subqueries,
                models=models,
                db_url=db_url,
                top_k=top_k,
                rate_limit=rate_limit,
            )
        )
