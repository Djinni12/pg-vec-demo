"""Node implementations for the GST LangGraph architecture.

Reuses existing pipeline components:
- Planner: src.routers.planner.plan_capabilities
- Legal Retrieval: src.retrieval_inspector.inspect_retrieval (Dense BGE-M3 + BM25 + RRF + Reranker)
- Rate Lookup: src.retrievers.rate_retriever.retrieve_rates (Structured Tariff Database / CSV)
- Direct Reasoning: src.tools.calculator.execute_calculator
- Grounded Synthesis: src.generators.answer_generator.generate_answer & extract_combined_sources
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional

from openai import OpenAI
from langchain_core.messages import AIMessage, HumanMessage

from src.generators.answer_generator import (
    extract_combined_sources,
    generate_answer,
    get_configured_model,
)
from src.graph.state import GSTGraphState
from src.observability import trace_store
from src.retrieval_inspector import LoadedModels, inspect_retrieval, load_models
from src.retrievers.decomposed_executor import execute_decomposed_subqueries
from src.retrievers.rate_retriever import retrieve_rates
from src.routers.planner import plan_capabilities
from src.tools.calculator import CalculationInputs, execute_calculator

logger = logging.getLogger(__name__)

# Module-level cache for heavy retrieval models (embedding + reranker)
_CACHED_MODELS: Optional[LoadedModels] = None


def get_cached_models() -> LoadedModels:
    """Lazily load and cache embedding and reranker models."""
    global _CACHED_MODELS
    if _CACHED_MODELS is None:
        _CACHED_MODELS = load_models()
    return _CACHED_MODELS


# -----------------------------------------------------------------------------
# Node 1: Planner
# -----------------------------------------------------------------------------
def planner_node(
    state: GSTGraphState,
    *,
    use_llm: bool | None = None,
) -> dict[str, Any]:
    """Analyzes user query and determines required capabilities.

    Does not force single-intent routing. Multi-capability queries (e.g. rate + legal)
    activate both needs_rate and needs_legal flags.
    """
    t0 = time.perf_counter()
    query = state["user_query"]
    prior_messages = state.get("messages", [])
    execution_id = state.get("execution_id")
    plan = plan_capabilities(query, use_llm=use_llm, history=prior_messages, request_id=execution_id)

    needs_legal = bool(plan.get("needs_legal_retrieval", False))
    needs_rate = bool(
        plan.get("needs_structured_rate_lookup", False)
        or plan.get("needs_hsn_lookup", False)
    )
    if plan.get("needs_query_decomposition"):
        for sq in plan.get("retrieval_subqueries", []):
            sq_t = sq.get("type") if isinstance(sq, dict) else getattr(sq, "type", "")
            if sq_t == "legal":
                needs_legal = True
            elif sq_t == "rate":
                needs_rate = True
    needs_notification = bool(plan.get("needs_notification_retrieval", False))
    needs_calculation = bool(plan.get("needs_calculation", False))

    clean_queries = plan.get("clean_subqueries", {})
    user_premises = plan.get("user_premises", {})

    # If rate query is generic/unnamed and user premise already states exempt/hypothetical without a named commodity
    rate_q = (clean_queries.get("rate_query") or "").lower()
    if needs_rate and any(g in rate_q for g in ["the product", "for the product", "unnamed product", "an item", "status for the product"]):
        needs_rate = False

    # Grounded reasoning is required when retrievals are active AND reasoning/calculation is needed
    needs_grounded = bool(
        (needs_legal or needs_rate or needs_notification) and (
            needs_calculation
            or plan.get("needs_temporal_reasoning", False)
            or plan.get("needs_comparison", False)
            or plan.get("needs_exception_reasoning", False)
            or bool(user_premises.get("itc_balances"))
            or (bool(user_premises) and plan.get("needs_direct_reasoning", False))
        )
    )

    # Direct reasoning is strictly for retrieval-free queries
    needs_direct = bool(
        not (needs_legal or needs_rate or needs_notification) and (
            plan.get("needs_direct_reasoning", False)
            or (not needs_calculation and bool(user_premises))
        )
    )

    if not needs_rate and not needs_legal and not needs_notification:
        route_str = "direct"
    elif needs_rate and needs_legal:
        route_str = "mixed"
    elif needs_rate:
        route_str = "rate"
    elif needs_legal:
        route_str = "legal"
    elif needs_notification:
        route_str = "notification"
    else:
        route_str = "direct"

    planner_ms = round((time.perf_counter() - t0) * 1000, 3)

    execution_id = state.get("execution_id")
    if execution_id:
        clean_subq_with_route = dict(clean_queries)
        clean_subq_with_route["route"] = route_str
        trace_store.record_planner(
            execution_id,
            capability_flags={
                "needs_legal": needs_legal,
                "needs_rate": needs_rate,
                "needs_notification": needs_notification,
                "needs_direct_reasoning": needs_direct,
                "needs_calculation": needs_calculation,
                "needs_grounded_reasoning": needs_grounded,
                "needs_clarification": bool(plan.get("needs_clarification", False)),
                "needs_query_decomposition": bool(plan.get("needs_query_decomposition", False)),
            },
            clean_subqueries=clean_subq_with_route,
            extracted_user_premises=dict(user_premises),
            clarification_decision={
                "needs_clarification": bool(plan.get("needs_clarification", False)),
                "clarification_prompt": plan.get("clarification_prompt"),
            },
            query_decomposition={
                "needs_query_decomposition": bool(plan.get("needs_query_decomposition", False)),
                "retrieval_subqueries": [
                    (sq if isinstance(sq, dict) else (sq.model_dump() if hasattr(sq, "model_dump") else dict(sq)))
                    for sq in plan.get("retrieval_subqueries", [])
                ],
            },
            timing_ms=planner_ms,
        )
        selected_nodes = []
        if needs_legal:
            selected_nodes.append("legal_retrieval")
        if needs_rate:
            selected_nodes.append("rate_lookup")
        if needs_legal or needs_rate or needs_notification:
            selected_nodes.append("notification_support")
        if needs_grounded:
            selected_nodes.append("grounded_reasoning")
        if needs_direct:
            selected_nodes.append("direct_reasoning")
        if needs_calculation:
            selected_nodes.append("calculation")
        selected_nodes.append("synthesis")
        trace_store.record_selected_nodes(execution_id, selected_nodes)
        trace_store.record_node_execution(
            execution_id,
            "planner",
            status="success",
            timing_ms=planner_ms,
        )

    return {
        "needs_legal": needs_legal,
        "needs_rate": needs_rate,
        "needs_notification": needs_notification,
        "needs_direct_reasoning": needs_direct,
        "needs_calculation": needs_calculation,
        "needs_grounded_reasoning": needs_grounded,
        "clean_queries": clean_queries,
        "user_premises": user_premises,
        "legal_results": [],
        "rate_results": [],
        "notification_results": [],
        "reasoning_result": None,
        "calculation_inputs": None,
        "calculation_result": None,
        "route": route_str,
        "plan": plan,
        "messages": [HumanMessage(content=query)],
    }


# -----------------------------------------------------------------------------
# Node 2: Legal Retrieval
# -----------------------------------------------------------------------------
def legal_retrieval_node(
    state: GSTGraphState,
    *,
    models: Optional[LoadedModels] = None,
    db_url: Optional[str] = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Executes full legal hybrid retrieval (BGE-M3 + BM25 + RRF + Reranker).

    Preserves document type, reference (Section/Rule/Form), content, and scores.
    """
    t0 = time.perf_counter()
    plan = state.get("plan") or {}
    needs_decomp = bool(plan.get("needs_query_decomposition", False))
    retrieval_subqueries = plan.get("retrieval_subqueries", [])
    legal_subqueries = [
        sq for sq in retrieval_subqueries
        if (isinstance(sq, dict) and sq.get("type") == "legal")
        or (hasattr(sq, "type") and getattr(sq, "type", "") == "legal")
    ]

    if needs_decomp and legal_subqueries:
        retrieval_models = models or get_cached_models()
        decomp_data = execute_decomposed_subqueries(
            legal_subqueries,
            models=retrieval_models,
            db_url=db_url,
            top_k=top_k,
        )
        chunks = decomp_data.get("legal_results", [])
        elapsed_ms = decomp_data.get("timings_ms", {}).get("total", round((time.perf_counter() - t0) * 1000, 3))
        execution_id = state.get("execution_id")
        if execution_id:
            subquery_texts = [
                (sq.get("query") if isinstance(sq, dict) else getattr(sq, "query", ""))
                for sq in legal_subqueries
            ]
            trace_store.record_legal_retrieval(
                execution_id,
                retrieval_query="; ".join(subquery_texts),
                retrieval_data={"results": chunks, "timings_ms": decomp_data.get("timings_ms", {})},
                timing_ms=elapsed_ms,
            )
            trace_store.record_node_execution(
                execution_id,
                "legal_retrieval",
                status="success" if chunks else "empty",
                timing_ms=elapsed_ms,
            )
        return {"legal_results": chunks}

    legal_q = (
        state.get("clean_queries", {}).get("legal_query")
        or state["user_query"]
    )

    q_lower = (state.get("user_query") or "").lower()
    is_exempt_tax_scenario = any(
        w in q_lower for w in ["exempt", "exemption", "bhulthi", "galti", "ભૂલથી"]
    ) and any(
        w in q_lower for w in ["collect", "tax", "gst", "18%", "5%", "12%", "28%", "ટેક્સ", "લીધો"]
    ) and not any(
        w in q_lower for w in ["interstate", "intra-state", "inter-state", "intrastate", "igst instead of", "cgst instead of"]
    )
    if is_exempt_tax_scenario and "76" not in legal_q and "credit note" not in legal_q.lower():
        legal_q = f"{legal_q} tax collected on exempt supply but not paid to Government Section 76 credit note Section 34"

    retrieval_models = models or get_cached_models()
    data: dict[str, Any] = {}

    try:
        data = inspect_retrieval(
            legal_q,
            top_k=top_k,
            models=retrieval_models,
            db_url=db_url,
        )
        chunks = data.get("results", [])
    except Exception as exc:
        logger.warning(f"Legal retrieval error: {exc}")
        chunks = []

    # Apply Legal Applicability Factual Scope Filter:
    # If the user scenario is tax collected on exempt supplies,
    # exclude provisions that strictly govern inter-state vs intra-state mismatch (Section 77, 19, 12).
    if is_exempt_tax_scenario and chunks:
        chunks = [
            c for c in chunks
            if not (
                ("section 77" in (c.get("reference") or "").lower() or
                 "section 19" in (c.get("reference") or "").lower() or
                 "section 12" in (c.get("reference") or "").lower()) and
                "wrongfully collected" in (c.get("title") or "").lower()
            )
        ]

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    execution_id = state.get("execution_id")
    if execution_id:
        trace_store.record_legal_retrieval(
            execution_id,
            retrieval_query=legal_q,
            retrieval_data=data,
            timing_ms=elapsed_ms,
            final_chunks=chunks,
        )
        trace_store.record_node_execution(
            execution_id,
            "legal_retrieval",
            status="success" if chunks else "empty",
            timing_ms=elapsed_ms,
        )

    return {"legal_results": chunks}


# -----------------------------------------------------------------------------
# Node 3: Rate Lookup
# -----------------------------------------------------------------------------
def rate_lookup_node(
    state: GSTGraphState,
    *,
    db_url: Optional[str] = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Retrieves structured GST rate records from the verified rate database/CSV.

    Rates are never guessed or inferred by an LLM.
    """
    t0 = time.perf_counter()
    plan = state.get("plan") or {}
    needs_decomp = bool(plan.get("needs_query_decomposition", False))
    retrieval_subqueries = plan.get("retrieval_subqueries", [])
    rate_subqueries = [
        sq for sq in retrieval_subqueries
        if (isinstance(sq, dict) and sq.get("type") == "rate")
        or (hasattr(sq, "type") and getattr(sq, "type", "") == "rate")
    ]

    if needs_decomp and rate_subqueries:
        decomp_data = execute_decomposed_subqueries(
            rate_subqueries,
            db_url=db_url,
            rate_limit=limit,
        )
        rates = decomp_data.get("rate_results", [])
        elapsed_ms = decomp_data.get("timings_ms", {}).get("total", round((time.perf_counter() - t0) * 1000, 3))
        selected_rate = rates[0] if rates else None
        execution_id = state.get("execution_id")
        if execution_id:
            subquery_texts = [
                (sq.get("query") if isinstance(sq, dict) else getattr(sq, "query", ""))
                for sq in rate_subqueries
            ]
            trace_store.record_rate_retrieval(
                execution_id,
                lookup_query="; ".join(subquery_texts),
                returned_candidates=rates,
                selected_rate_used_downstream=selected_rate,
                timing_ms=elapsed_ms,
            )
            trace_store.record_node_execution(
                execution_id,
                "rate_lookup",
                status="success" if rates else "empty",
                timing_ms=elapsed_ms,
            )
        return {"rate_results": rates}

    rate_q = (
        state.get("clean_queries", {}).get("rate_query")
        or state["user_query"]
    )

    try:
        rates = retrieve_rates(rate_q, db_url=db_url, limit=limit)
    except Exception as exc:
        logger.warning(f"Rate lookup error: {exc}")
        rates = []

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    selected_rate = rates[0] if rates else None

    execution_id = state.get("execution_id")
    if execution_id:
        trace_store.record_rate_retrieval(
            execution_id,
            lookup_query=rate_q,
            returned_candidates=rates,
            selected_rate_used_downstream=selected_rate,
            timing_ms=elapsed_ms,
        )
        trace_store.record_node_execution(
            execution_id,
            "rate_lookup",
            status="success" if rates else "empty",
            timing_ms=elapsed_ms,
        )

    return {"rate_results": rates}


# -----------------------------------------------------------------------------
# Node 3b: Supporting Notification Retrieval
# -----------------------------------------------------------------------------
def notification_support_node(
    state: GSTGraphState,
    *,
    models: Optional[LoadedModels] = None,
    db_url: Optional[str] = None,
    top_k: int = 2,
) -> dict[str, Any]:
    """Retrieves materially relevant supporting Gazette notification chunks via vector search.

    Always triggered post-primary retrieval (rate_lookup or legal_retrieval), or when an explicit
    notification query is present.
    Constructs a semantic support query from discovered metadata, computes BGE-M3 dense query embedding,
    searches pgvector, applies metadata-aware boosting, and preserves only materially relevant chunks.
    """
    t0 = time.perf_counter()
    rate_results = state.get("rate_results", [])
    legal_results = state.get("legal_results", [])
    clean_queries = state.get("clean_queries", {}) or {}
    explicit_notif_q = clean_queries.get("notification_query")
    user_q = state.get("user_query", "")

    support_metadata: dict[str, Any] = {}
    query_parts: list[str] = []

    if explicit_notif_q:
        query_parts.append(explicit_notif_q)

    if rate_results:
        r0 = rate_results[0]
        desc = r0.get("description") or ""
        hsn = r0.get("code") or r0.get("hsn_code") or ""
        notif_no = r0.get("notification_no") or r0.get("notification_number") or ""
        sched = r0.get("schedule") or ""
        serial = r0.get("serial_no") or ""
        support_metadata.update({
            "target_notification": notif_no,
            "hsn_code": hsn,
            "serial_no": serial,
            "schedule": sched,
        })
        if desc:
            query_parts.append(desc)
        if hsn:
            query_parts.append(f"HSN {hsn}")
        if notif_no:
            query_parts.append(f"notification {notif_no}")
        if sched:
            query_parts.append(sched)
        if serial:
            query_parts.append(f"entry {serial}")
        query_parts.append("rate amendment exemption")

    if legal_results and not rate_results:
        legal_q = clean_queries.get("legal_query") or user_q
        query_parts.append(legal_q)

    if not query_parts:
        query_parts.append(user_q)

    semantic_query = " ".join(p for p in query_parts if p).strip()

    retrieval_models = models or get_cached_models()
    embed_model = retrieval_models.embedding_model if retrieval_models else None

    from src.retrievers.notification_retriever import retrieve_supporting_notifications

    try:
        notif_chunks = retrieve_supporting_notifications(
            semantic_query,
            support_metadata=support_metadata,
            top_k=top_k,
            db_url=db_url,
            model=embed_model,
        )
    except Exception as exc:
        logger.warning(f"Notification support retrieval error: {exc}")
        notif_chunks = []

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    execution_id = state.get("execution_id")
    if execution_id:
        trace_store.record_notification_retrieval(
            execution_id,
            query=semantic_query,
            chunks=notif_chunks,
            timing_ms=elapsed_ms,
            support_metadata=support_metadata,
        )
        trace_store.record_node_execution(
            execution_id,
            "notification_support",
            status="success" if notif_chunks else "empty",
            timing_ms=elapsed_ms,
        )

    return {"notification_results": notif_chunks}


# -----------------------------------------------------------------------------
# Node 3c: Fallback Web Search Retrieval
# -----------------------------------------------------------------------------
def web_search_node(
    state: GSTGraphState,
    *,
    db_url: Optional[str] = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Executes fallback web search when local retrievals report missing evidence.

    1. Searches authoritative official GST sources.
    2. Extracts missing identifiers (e.g. HSN/SAC codes) and feeds them back
       into local retrieve_rates() to preserve verified local rate records.
    3. Preserves web source URLs and snippets for grounded synthesis citations.
    """
    t0 = time.perf_counter()
    from src.retrievers.web_search_retriever import search_web_fallback

    clean_queries = state.get("clean_queries", {}) or {}
    rate_q = clean_queries.get("rate_query")
    legal_q = clean_queries.get("legal_query")
    user_q = state.get("user_query", "")

    if state.get("needs_rate") and not state.get("rate_results"):
        search_q = rate_q or user_q
        search_type = "rate"
    elif state.get("needs_legal") and not state.get("legal_results"):
        search_q = legal_q or user_q
        search_type = "legal"
    else:
        search_q = user_q
        search_type = "general"

    search_data = search_web_fallback(
        search_q,
        search_type=search_type,
        db_url=db_url,
        limit=limit,
    )

    new_rate_results = list(state.get("rate_results", []))
    if search_data.get("rate_results"):
        for r in search_data["rate_results"]:
            if not any(
                existing.get("hsn_code") == r.get("hsn_code")
                and existing.get("serial_no") == r.get("serial_no")
                for existing in new_rate_results
            ):
                new_rate_results.append(r)

    web_chunks = search_data.get("web_results", [])
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)

    execution_id = state.get("execution_id")
    if execution_id:
        if hasattr(trace_store, "record_web_search"):
            trace_store.record_web_search(
                execution_id,
                query=search_q,
                discovered_hsn=search_data.get("discovered_hsn"),
                web_results=web_chunks,
                timing_ms=elapsed_ms,
            )
        trace_store.record_node_execution(
            execution_id,
            "web_search",
            status="success" if (web_chunks or new_rate_results) else "empty",
            timing_ms=elapsed_ms,
        )

    return {
        "web_results": web_chunks,
        "rate_results": new_rate_results,
        "web_search_attempted": True,
    }


# -----------------------------------------------------------------------------
# Node 4: Grounded Reasoning
# -----------------------------------------------------------------------------
def grounded_reasoning_node(state: GSTGraphState) -> dict[str, Any]:
    """Reasons over retrieved legal, rate, and notification evidence.

    Determines how retrieved facts apply to the user scenario and prepares
    structured inputs for the calculation_node according to the Source Interpretation Rules.
    Does NOT perform arithmetic itself.
    """
    t0 = time.perf_counter()
    user_premises = state.get("user_premises", {}) or {}
    rate_results = state.get("rate_results", [])
    legal_results = state.get("legal_results", [])
    notification_results = state.get("notification_results", [])

    # Resolve applicable rate from rate_results or user premise (never invent)
    from src.generators.answer_generator import extract_rate_pct
    assumed_rate = user_premises.get("assumed_rate")
    rate_source: Optional[str] = None

    if rate_results:
        calc_rate = extract_rate_pct(rate_results)
        if calc_rate is not None:
            assumed_rate = calc_rate
            r0 = rate_results[0]
            hsn = r0.get("code") or ""
            desc = r0.get("description") or "Tariff item"
            rate_source = f"Tariff HSN {hsn} ({desc}) - {calc_rate}%"
    elif user_premises.get("rate_is_user_assumed"):
        rate_source = f"User hypothetical assumption ({assumed_rate}%)"
    elif assumed_rate is not None:
        rate_source = f"Statutory tariff rate from context ({assumed_rate}%)"

    taxable_amount = user_premises.get("taxable_amount") or user_premises.get("base_amount")
    discount_pct = float(user_premises.get("discount_pct") or 0.0)
    itc_balances = user_premises.get("itc_balances") or {}
    supply_type = user_premises.get("supply_type") or "interstate"

    reasoning_parts: list[str] = []
    calc_inputs: Optional[dict[str, Any]] = None

    # Case A: Input Tax Credit balance query (e.g. Butter + interstate + CGST/SGST/IGST)
    if itc_balances:
        total_credit = sum(float(v) for v in itc_balances.values())
        breakdown_str = ", ".join(f"{k.upper()}: ₹{float(v):,.2f}" for k, v in itc_balances.items())

        legal_ref = "Section 49 and Rule 88A of the CGST Act"
        for chunk in legal_results:
            ref = chunk.get("reference") or chunk.get("doc_reference") or ""
            if "49" in ref or "88" in ref:
                legal_ref = ref
                break

        if supply_type == "interstate":
            rule_text = (
                f"Under {legal_ref}, input tax credit of IGST must first be completely exhausted towards payment of IGST liability. "
                "Thereafter, input tax credit of CGST and SGST can be utilized towards payment of IGST in any order and in any proportion. "
                f"Therefore, the entire accumulated credit of ₹{total_credit:,.2f} ({breakdown_str}) is fully eligible to discharge outward IGST liability."
            )
        else:
            rule_text = (
                f"Under {legal_ref}, IGST credit is first utilized towards IGST, then CGST and SGST. "
                "CGST credit cannot be cross-utilized to pay SGST liability, and SGST credit cannot be cross-utilized to pay CGST liability. "
                f"Available credit balances: {breakdown_str}."
            )
        reasoning_parts.append(rule_text)

        if assumed_rate is not None and assumed_rate > 0:
            reasoning_parts.append(
                f"For the outward supply ({rate_source}), the applicable GST rate is {assumed_rate}%. "
                f"The maximum outward taxable supply value that can be discharged without cash payment is: Total Eligible Credit (₹{total_credit:,.2f}) divided by the GST rate ({assumed_rate}%)."
            )
            calc_inputs = {
                "operation": "max_taxable_value_from_credit",
                "available_eligible_credit": total_credit,
                "tax_rate_pct": float(assumed_rate),
                "credit_breakdown": itc_balances,
            }
        else:
            reasoning_parts.append(
                f"Available input tax credit is ₹{total_credit:,.2f}. Applicable GST rate must be confirmed from official tariff records."
            )

    # Case B: Taxable Amount with discount and/or verified rate
    elif taxable_amount is not None:
        amt = float(taxable_amount)
        if discount_pct > 0 and assumed_rate is not None:
            reasoning_parts.append(
                f"A discount of {discount_pct}% applies to base amount ₹{amt:,.2f}, followed by {assumed_rate}% GST ({rate_source})."
            )
            calc_inputs = {
                "operation": "discount_and_tax",
                "base_amount": amt,
                "discount_pct": discount_pct,
                "tax_rate_pct": float(assumed_rate),
            }
        elif discount_pct > 0:
            reasoning_parts.append(f"A discount of {discount_pct}% applies to base amount ₹{amt:,.2f}.")
            calc_inputs = {
                "operation": "discount_only",
                "base_amount": amt,
                "discount_pct": discount_pct,
            }
        elif assumed_rate is not None:
            q_lower = (state.get("user_query") or "").lower()
            is_exempt_tax_scenario = any(
                w in q_lower for w in ["exempt", "exemption", "bhulthi", "galti", "ભૂલથી"]
            ) and any(
                w in q_lower for w in ["collect", "tax", "gst", "18%", "5%", "12%", "28%", "ટેક્સ", "લીધો"]
            )
            if is_exempt_tax_scenario:
                err_tax = amt * float(assumed_rate) / 100.0
                reasoning_parts.append(
                    f"Taxable value of ₹{amt:,.2f} had {assumed_rate}% GST (₹{err_tax:,.2f}) collected in error on an exempt product. "
                    "Under Section 76(1) of the CGST Act, any amount collected as tax must be paid to the Government, whether the supply is taxable or not. "
                    f"Under Section 34 of the CGST Act, the supplier may issue a Credit Note to the customer to rectify the excess tax and refund or adjust the ₹{err_tax:,.2f}."
                )
            else:
                reasoning_parts.append(
                    f"Taxable value of ₹{amt:,.2f} is subject to GST at {assumed_rate}% ({rate_source})."
                )
            calc_inputs = {
                "operation": "tax_on_value",
                "taxable_value": amt,
                "tax_rate_pct": float(assumed_rate),
            }
        else:
            reasoning_parts.append(
                f"Taxable value is ₹{amt:,.2f}, but no GST rate was specified or found in tariff records."
            )
    else:
        reasoning_parts.append("Evaluated statutory provisions and tariff classifications for user scenario.")

    # Reconcile supporting Gazette notifications per Source Interpretation Rules
    for nchunk in notification_results:
        sm = nchunk.get("source_metadata", {})
        notif_no = sm.get("notification_number")
        target_notif = sm.get("target_notification")
        op_type = sm.get("operation_type")
        eff_date = sm.get("effective_date")
        if op_type and target_notif:
            reasoning_parts.append(
                f"Amending Notification No. {notif_no} ({op_type}) modifies {target_notif} effective from {eff_date or 'as notified'}."
            )
        elif notif_no and eff_date:
            reasoning_parts.append(
                f"Notification No. {notif_no} effective from {eff_date} provides authoritative Gazette grounding."
            )

    reasoning_text = "\n\n".join(reasoning_parts)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)

    execution_id = state.get("execution_id")
    if execution_id:
        trace_store.record_grounded_reasoning(
            execution_id,
            reasoning_output=reasoning_text,
            calculation_inputs=calc_inputs,
            timing_ms=elapsed_ms,
        )
        trace_store.record_node_execution(
            execution_id,
            "grounded_reasoning",
            status="success",
            timing_ms=elapsed_ms,
        )

    return {
        "reasoning_result": reasoning_text,
        "calculation_inputs": calc_inputs,
    }


# -----------------------------------------------------------------------------
# Node 5: Direct Reasoning (Retrieval-Free)
# -----------------------------------------------------------------------------
def direct_reasoning_node(state: GSTGraphState) -> dict[str, Any]:
    """Retrieval-free reasoning node for pure logic / direct premises queries.

    Prepares structured calculation inputs when user premises provide direct numbers.
    Does NOT perform arithmetic itself.
    """
    t0 = time.perf_counter()
    user_premises = state.get("user_premises", {}) or {}
    taxable_amount = user_premises.get("taxable_amount") or user_premises.get("base_amount")
    discount_pct = float(user_premises.get("discount_pct") or 0.0)
    assumed_rate = user_premises.get("assumed_rate")
    rate_source: Optional[str] = None

    if user_premises.get("rate_is_user_assumed"):
        rate_source = f"User hypothetical assumption ({assumed_rate}%)"
    elif assumed_rate is not None:
        rate_source = f"Statutory tariff rate from context ({assumed_rate}%)"
    else:
        m_rate = re.search(
            r"\b(\d+(?:\.\d+)?)\s*%\s*(?:gst|tax)\b",
            state.get("user_query", ""),
            re.IGNORECASE,
        )
        if m_rate:
            try:
                assumed_rate = float(m_rate.group(1))
                rate_source = f"Extracted from query ({assumed_rate}%)"
            except ValueError:
                assumed_rate = None

    calc_inputs: Optional[dict[str, Any]] = None
    reasoning_parts: list[str] = []

    if taxable_amount is not None:
        amt = float(taxable_amount)
        if discount_pct > 0 and assumed_rate is not None:
            reasoning_parts.append(
                f"Base amount: ₹{amt:,.2f}, Discount: {discount_pct}%, Assumed GST rate: {assumed_rate}% ({rate_source})."
            )
            calc_inputs = {
                "operation": "discount_and_tax",
                "base_amount": amt,
                "discount_pct": discount_pct,
                "tax_rate_pct": float(assumed_rate),
            }
        elif discount_pct > 0:
            reasoning_parts.append(f"Base amount: ₹{amt:,.2f}, Discount: {discount_pct}%.")
            calc_inputs = {
                "operation": "discount_only",
                "base_amount": amt,
                "discount_pct": discount_pct,
            }
        elif assumed_rate is not None:
            reasoning_parts.append(
                f"Taxable value: ₹{amt:,.2f}, Assumed GST rate: {assumed_rate}% ({rate_source})."
            )
            calc_inputs = {
                "operation": "tax_on_value",
                "taxable_value": amt,
                "tax_rate_pct": float(assumed_rate),
            }
        else:
            reasoning_parts.append(f"Taxable amount: ₹{amt:,.2f} (no GST rate provided).")
    else:
        reasoning_parts.append("Direct logical reasoning verified without external retrieval.")

    reasoning_text = "\n\n".join(reasoning_parts)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)

    execution_id = state.get("execution_id")
    if execution_id:
        trace_store.record_direct_reasoning(
            execution_id,
            reasoning_output=reasoning_text,
            calculation_inputs=calc_inputs,
            timing_ms=elapsed_ms,
        )
        trace_store.record_node_execution(
            execution_id,
            "direct_reasoning",
            status="success",
            timing_ms=elapsed_ms,
        )

    return {
        "reasoning_result": reasoning_text,
        "calculation_inputs": calc_inputs,
    }


# -----------------------------------------------------------------------------
# Node 6: Deterministic Calculation
# -----------------------------------------------------------------------------
def calculation_node(state: GSTGraphState) -> dict[str, Any]:
    """Dedicated deterministic calculation node.

    - Uses structured calculation inputs (from state or built from user premises and rate).
    - Calls execute_calculator().
    - Never invents/defaults a GST rate.
    - Uses either user-provided rate or verified rate from rate_results.
    - Stores structured calculation_result in state.
    """
    t0 = time.perf_counter()
    raw_inputs = state.get("calculation_inputs")
    user_premises = state.get("user_premises", {}) or {}
    rate_results = state.get("rate_results", [])
    query = state.get("user_query", "")

    # If calculation_inputs was not pre-built by a reasoning node, construct it now
    if not raw_inputs:
        assumed_rate = user_premises.get("assumed_rate")
        rate_source: Optional[str] = None
        if assumed_rate is not None:
            rate_source = "User query premise"
        else:
            m_rate = re.search(
                r"\b(\d+(?:\.\d+)?)\s*%\s*(?:gst|tax)\b",
                query,
                re.IGNORECASE,
            )
            if m_rate:
                try:
                    assumed_rate = float(m_rate.group(1))
                    rate_source = f"Extracted from query ({assumed_rate}%)"
                except ValueError:
                    assumed_rate = None

        if assumed_rate is None and rate_results:
            from src.generators.answer_generator import extract_rate_pct
            calc_rate = extract_rate_pct(rate_results)
            if calc_rate is not None:
                assumed_rate = calc_rate
                r0 = rate_results[0]
                hsn = r0.get("code") or ""
                desc = r0.get("description") or "Tariff item"
                rate_source = f"Tariff HSN {hsn} ({desc}) - {calc_rate}%"

        taxable_amount = user_premises.get("taxable_amount") or user_premises.get("base_amount")
        discount_pct = float(user_premises.get("discount_pct") or 0.0)
        itc_balances = user_premises.get("itc_balances") or {}

        if itc_balances and assumed_rate is not None and assumed_rate > 0:
            total_credit = sum(float(v) for v in itc_balances.values())
            raw_inputs = {
                "operation": "max_taxable_value_from_credit",
                "available_eligible_credit": total_credit,
                "tax_rate_pct": float(assumed_rate),
                "credit_breakdown": itc_balances,
            }
        elif taxable_amount is not None:
            amt = float(taxable_amount)
            if discount_pct > 0 and assumed_rate is not None:
                raw_inputs = {
                    "operation": "discount_and_tax",
                    "base_amount": amt,
                    "discount_pct": discount_pct,
                    "tax_rate_pct": float(assumed_rate),
                }
            elif discount_pct > 0:
                raw_inputs = {
                    "operation": "discount_only",
                    "base_amount": amt,
                    "discount_pct": discount_pct,
                }
            elif assumed_rate is not None:
                raw_inputs = {
                    "operation": "tax_on_value",
                    "taxable_value": amt,
                    "tax_rate_pct": float(assumed_rate),
                }

    calc_dict: Optional[dict[str, Any]] = None
    calc_in: Optional[CalculationInputs] = None
    rate_used: Optional[float] = None
    rate_source_resolved: Optional[str] = None

    if raw_inputs:
        try:
            calc_in = CalculationInputs(**raw_inputs)
            rate_used = calc_in.tax_rate_pct
            if rate_used is not None:
                if rate_results:
                    r0 = rate_results[0]
                    hsn = r0.get("code") or ""
                    desc = r0.get("description") or "Tariff item"
                    rate_source_resolved = f"Tariff HSN {hsn} ({desc}) - {rate_used}%"
                elif user_premises.get("rate_is_user_assumed"):
                    rate_source_resolved = f"User hypothetical assumption ({rate_used}%)"
                elif user_premises.get("assumed_rate") == rate_used:
                    rate_source_resolved = f"Statutory tariff rate from context ({rate_used}%)"
                else:
                    rate_source_resolved = f"Extracted from query ({rate_used}%)"

            calc_res = execute_calculator(calc_in)
            if calc_res.status == "success":
                calc_dict = calc_res.model_dump()
        except Exception as exc:
            logger.warning(f"Deterministic calculator error: {exc}")
            calc_dict = None

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)

    execution_id = state.get("execution_id")
    if execution_id:
        trace_store.record_calculation(
            execution_id,
            calculation_inputs=raw_inputs,
            rate_used=rate_used,
            source_of_rate=rate_source_resolved,
            operation=calc_in.operation if calc_in else (raw_inputs.get("operation") if raw_inputs else None),
            deterministic_result=calc_dict,
            timing_ms=elapsed_ms,
        )
        trace_store.record_node_execution(
            execution_id,
            "calculation",
            status="success" if calc_dict else "skipped",
            timing_ms=elapsed_ms,
        )

    return {
        "calculation_inputs": raw_inputs,
        "calculation_result": calc_dict,
    }


# -----------------------------------------------------------------------------
# Node 5: Grounded Synthesis
# -----------------------------------------------------------------------------
def synthesis_node(
    state: GSTGraphState,
    *,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Synthesizes final grounded response using whatever evidence is present in state.

    Integrates:
    - legal_results (if present)
    - rate_results (if present)
    - reasoning_result & calculation_result (if present)
    Preserves full source metadata and citations.
    """
    t0 = time.perf_counter()
    query = state["user_query"]
    legal_results = state.get("legal_results", [])
    rate_results = state.get("rate_results", [])
    notification_results = state.get("notification_results", [])
    reasoning_res = state.get("reasoning_result")
    calc_res = state.get("calculation_result")
    user_premises = state.get("user_premises")

    is_direct = (
        state.get("needs_direct_reasoning", False)
        and not state.get("needs_legal", False)
        and not state.get("needs_rate", False)
        and not state.get("needs_notification", False)
    )

    # Extract unified citations
    web_results = state.get("web_results") or []
    all_chunks = list(legal_results) + list(notification_results) + list(web_results)
    sources = extract_combined_sources(chunks=all_chunks, rate_results=rate_results)
    model_name = get_configured_model(model)

    combined_chunks = list(legal_results) + list(web_results)

    # Extract prior dialogue turns (all messages before the current query)
    all_msgs = state.get("messages", [])
    prior_turns = all_msgs[:-1] if len(all_msgs) > 1 else []

    # Attempt LLM generation
    try:
        gen = generate_answer(
            query=query,
            chunks=combined_chunks if combined_chunks else None,
            rate_results=rate_results if rate_results else None,
            notification_chunks=notification_results if notification_results else None,
            user_premises=user_premises,
            calculation_result=calc_res,
            client=client,
            model=model,
            direct_reasoning=is_direct,
            history=prior_turns,
            request_id=state.get("execution_id"),
        )
        answer = gen.get("answer", "")
        if gen.get("sources"):
            sources = gen["sources"]
        model_name = gen.get("model_used", model_name)
    except Exception as exc:
        logger.info(f"LLM generation unavailable or failed ({exc}), using deterministic grounded synthesis.")
        answer = _build_deterministic_synthesis(
            query=query,
            rate_results=rate_results,
            legal_results=legal_results,
            notification_results=notification_results,
            web_results=web_results,
            reasoning_result=reasoning_res,
            calculation_result=calc_res,
        )

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    execution_id = state.get("execution_id")
    if execution_id:
        trace_store.record_synthesis(
            execution_id,
            sources_provided=sources,
            model_name=model_name,
            generation_timing_ms=elapsed_ms,
            final_answer=answer,
        )
        trace_store.record_node_execution(
            execution_id,
            "synthesis",
            status="success",
            timing_ms=elapsed_ms,
        )

    return {
        "final_answer": answer,
        "sources": sources,
        "model_used": model_name,
        "messages": [AIMessage(content=answer)],
    }


def _build_deterministic_synthesis(
    query: str,
    rate_results: list[dict[str, Any]],
    legal_results: list[dict[str, Any]],
    notification_results: Optional[list[dict[str, Any]]] = None,
    web_results: Optional[list[dict[str, Any]]] = None,
    reasoning_result: Optional[str] = None,
    calculation_result: Optional[dict[str, Any]] = None,
) -> str:
    """Generates a structured, grounded answer directly from retrieved evidence.

    Used when an LLM endpoint is offline or in mock testing environments.
    """
    sections: list[str] = []

    # 1. Rate Details
    if rate_results:
        r0 = rate_results[0]
        desc = r0.get("description", "Item")
        hsn = r0.get("code") or r0.get("hsn_code", "N/A")
        rate_cat = r0.get("rate_category", "")
        source_rate = r0.get("source_rate") or r0.get("gst_rate") or "N/A"
        total_gst = r0.get("total_gst_rate")
        cgst = r0.get("cgst_rate")
        sgst = r0.get("sgst_rate")
        notif = r0.get("notification_number") or r0.get("notification_no", "N/A")

        if rate_cat == "EXEMPTION" or total_gst in ("0%", "Nil") or source_rate in ("Nil", "0%"):
            rate_line = f"{desc} is exempt from GST, so no GST is charged (0%) under HSN {hsn}."
        elif total_gst and cgst and sgst:
            rate_line = f"{desc} attracts {total_gst} GST under HSN {hsn}. For an intra-state supply, this consists of {cgst} CGST and {sgst} SGST."
        else:
            rate_line = f"{desc} attracts {source_rate} GST under HSN {hsn}."

        rate_sec = [
            rate_line,
            "",
            "Details:",
            f"- HSN / Tariff Code: {hsn}",
            f"- Description: {desc}",
            f"- Total GST Rate: {total_gst or source_rate}",
            f"- Notification: {notif}",
        ]
        if cgst and sgst:
            rate_sec.append(f"- CGST Rate: {cgst} | SGST Rate: {sgst}")
        sections.append("\n".join(rate_sec))

    # 2. Supporting Notifications
    if notification_results:
        notif_lines = ["Supporting Gazette Evidence:"]
        for idx, chunk in enumerate(notification_results[:2], 1):
            ref = chunk.get("reference") or "Notification"
            title = chunk.get("title") or ""
            content = (chunk.get("content") or chunk.get("snippet") or "").strip()
            first_sentence = content.split("\n")[0] if content else ""
            notif_lines.append(f"{idx}. {ref} ({title}):\n   {first_sentence}")
        sections.append("\n\n".join(notif_lines))

    # 3. Web Verification Sources (Fallback)
    if web_results and not legal_results:
        web_lines = ["Official Web Verification Sources:"]
        for idx, chunk in enumerate(web_results[:3], 1):
            title = chunk.get("title") or "Web Source"
            url = chunk.get("url") or chunk.get("reference") or ""
            snip = chunk.get("snippet") or ""
            first_sentence = snip.split("\n")[0] if snip else ""
            web_lines.append(f"{idx}. {title} ({url}):\n   {first_sentence}")
        sections.append("\n\n".join(web_lines))

    # 4. Legal Findings
    if legal_results:
        legal_lines = ["Applicable Legal Provisions:"]
        for idx, chunk in enumerate(legal_results[:3], 1):
            ref = chunk.get("reference") or "Statutory Provision"
            title = chunk.get("title") or ""
            content = (chunk.get("content") or chunk.get("snippet") or "").strip()
            first_sentence = content.split("\n")[0] if content else ""
            legal_lines.append(f"{idx}. {ref} ({title}):\n   {first_sentence}")
        sections.append("\n\n".join(legal_lines))

    # 4. Direct Reasoning / Calculations
    if calculation_result and calculation_result.get("status") == "success":
        calc_lines = [
            f"Verified Calculation: ₹{calculation_result.get('result_value', 0.0):,.2f}",
            f"Formula: {calculation_result.get('formula')}",
        ]
        for step in calculation_result.get("steps", []):
            calc_lines.append(f"- {step}")
        sections.append("\n".join(calc_lines))
    elif reasoning_result:
        sections.append(f"Calculation Breakdown:\n{reasoning_result}")

    if not sections:
        return "No specific rate or legal records matching your query were found in the database."

    return "\n\n".join(sections)
