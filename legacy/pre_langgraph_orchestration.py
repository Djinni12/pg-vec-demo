"""LEGACY / DEPRECATED: Pre-LangGraph procedural orchestration pipeline.

This module contains the original pre-LangGraph monolithic procedural orchestration
functions (`run_gst_answer_flow` and `stream_gst_answer_flow`).

STATUS: DEPRECATED / REFERENCE ONLY
The active production orchestration layer is LangGraph located in `src/graph/`.
This legacy module is retained strictly for historical reference and verification.
It must NOT be imported or called by the active API, UI, or LangGraph runtime.
"""

from __future__ import annotations

import time
from typing import Any

from openai import OpenAI

from src.generators.answer_generator import (
    DEFAULT_ANSWER_TOP_K,
    extract_rate_pct,
    generate_answer,
    get_configured_model,
    stream_answer,
)
from src.generators.legal_interpreter import (
    StructuredLegalFindings,
    interpret_legal_findings,
)
from src.retrieval_inspector import LoadedModels, inspect_retrieval
from src.retrievers.notification_retriever import retrieve_notifications
from src.retrievers.rate_retriever import retrieve_rates
from src.routers.query_router import RouteType, plan_capabilities
from src.tools.calculator import (
    CalculationInputs,
    CalculationResult,
    execute_calculator,
)


def run_gst_answer_flow(
    query: str,
    top_k: int = DEFAULT_ANSWER_TOP_K,
    *,
    models: LoadedModels,
    openai_model: str | None = None,
    openai_client: OpenAI | None = None,
    db_url: str | None = None,
    route_override: str | None = None,
) -> dict[str, Any]:
    """[LEGACY] Execute pre-LangGraph procedural retrieval pipeline.

    DEPRECATED: Use `src.graph.invoke_gst_graph` or `src.graph.run_graph_chat` instead.
    """
    plan = plan_capabilities(query)

    # 1. Clarification fast-path if query is ambiguous
    if plan.get("needs_clarification") and not route_override:
        clarification_text = plan.get("clarification_prompt") or "Please provide more details."
        obs = {
            "plan": plan,
            "tools_executed": [],
            "retrieved_evidence": {"rates_count": 0, "legal_chunks_count": 0, "notification_chunks_count": 0},
            "structured_legal_findings": None,
            "calculation_inputs": None,
            "calculation_result": None,
            "sources_used": [],
            "knowledge_base_gap": None,
        }
        return {
            "query": query,
            "route": "clarification",
            "answer": clarification_text,
            "rate_results": [],
            "sources": [],
            "sources_used": [],
            "retrieval_timing": 0.0,
            "generation_timing": 0.0,
            "total_timing": 0.0,
            "model_used": get_configured_model(openai_model),
            "model": get_configured_model(openai_model),
            "plan": plan,
            "tools_executed": [],
            "structured_legal_findings": None,
            "calculation_inputs": None,
            "calculation_result": None,
            "observability": obs,
            "timings_ms": {
                "retrieval": 0.0,
                "generation": 0.0,
                "total": 0.0,
                "rate_lookup": 0.0,
                "notification_lookup": 0.0,
                "dense": 0.0,
                "bm25": 0.0,
                "rrf": 0.0,
                "reranker": 0.0,
            },
            "retrieval_debug": {
                "results": [],
                "dense_results": [],
                "bm25_results": [],
                "hybrid_results": [],
                "metadata": {},
                "models": {},
                "config": {},
            },
        }

    # 2. Capability Tool Execution
    if route_override:
        route = RouteType(route_override)
        needs_rate = route in (RouteType.RATE, RouteType.MIXED)
        needs_legal = route in (RouteType.LEGAL, RouteType.MIXED)
        needs_notif = False
        is_direct = route == RouteType.DIRECT
        route_str = route.value
    else:
        needs_rate = plan.get("needs_structured_rate_lookup", False) or plan.get("needs_hsn_lookup", False)
        needs_legal = plan.get("needs_legal_retrieval", False)
        needs_notif = plan.get("needs_notification_retrieval", False)
        is_direct = (
            plan.get("needs_direct_reasoning", False)
            and not needs_rate
            and not needs_legal
            and not needs_notif
        )
        if is_direct:
            route_str = "direct"
        elif needs_rate and (needs_legal or needs_notif):
            route_str = "mixed"
        elif needs_rate:
            route_str = "rate"
        elif needs_legal or needs_notif:
            route_str = "legal"
        else:
            route_str = "legal"

    rate_results: list[dict[str, Any]] = []
    legal_chunks: list[dict[str, Any]] = []
    notif_chunks: list[dict[str, Any]] = []
    rate_timing = 0.0
    notif_timing = 0.0
    tools_executed: list[dict[str, Any]] = []

    retrieval_data: dict[str, Any] = {
        "results": [],
        "dense_results": [],
        "bm25_results": [],
        "hybrid_results": [],
        "metadata": {},
        "models": {},
        "config": {},
        "timings_ms": {"total": 0.0, "dense": 0.0, "bm25": 0.0, "rrf": 0.0, "reranker": 0.0},
    }

    if needs_rate:
        rate_start = time.perf_counter()
        rate_q = plan.get("clean_subqueries", {}).get("rate_query") or query
        rate_results = retrieve_rates(rate_q, db_url=db_url, limit=top_k)
        rate_timing = round((time.perf_counter() - rate_start) * 1000, 3)
        tools_executed.append({
            "tool": "retrieve_rates",
            "query": rate_q,
            "results_count": len(rate_results),
            "timing_ms": rate_timing,
        })

    if needs_legal:
        legal_q = plan.get("clean_subqueries", {}).get("legal_query") or query
        retrieval_data = inspect_retrieval(legal_q, top_k=top_k, models=models, db_url=db_url)
        legal_chunks = retrieval_data.get("results") or []
        legal_timing = retrieval_data.get("timings_ms", {}).get("total", 0.0)
        tools_executed.append({
            "tool": "inspect_retrieval",
            "query": legal_q,
            "results_count": len(legal_chunks),
            "timing_ms": legal_timing,
        })

    if needs_notif:
        notif_start = time.perf_counter()
        notif_q = plan.get("clean_subqueries", {}).get("notification_query") or query
        notif_chunks = retrieve_notifications(
            notif_q,
            top_k=top_k,
            db_url=db_url,
            model=models.embedding_model if models else None,
        )
        notif_timing = round((time.perf_counter() - notif_start) * 1000, 3)
        tools_executed.append({
            "tool": "retrieve_notifications",
            "query": notif_q,
            "results_count": len(notif_chunks),
            "timing_ms": notif_timing,
        })

    legal_timing = retrieval_data.get("timings_ms", {}).get("total", 0.0)
    retrieval_timing = round(rate_timing + legal_timing + notif_timing, 3)
    final_chunks = legal_chunks + notif_chunks

    # --- Stage 2: Structured Legal Findings ---
    legal_findings: StructuredLegalFindings | None = None
    user_premises = plan.get("user_premises") or {}
    has_itc_balances = bool(user_premises.get("itc_balances"))

    if needs_legal and (plan.get("needs_calculation") or has_itc_balances):
        interp_start = time.perf_counter()
        legal_findings = interpret_legal_findings(
            legal_chunks=legal_chunks,
            user_premises=user_premises,
            query=query,
        )
        interp_timing = round((time.perf_counter() - interp_start) * 1000, 3)
        tools_executed.append({
            "tool": "interpret_legal_findings",
            "status": legal_findings.status,
            "supply_type": legal_findings.supply_type,
            "usable_credit_ledgers": legal_findings.usable_credit_ledgers,
            "total_usable_credit": legal_findings.total_usable_credit,
            "unresolved_reason": legal_findings.unresolved_reason,
            "timing_ms": interp_timing,
        })

    # --- Stage 2: Deterministic Calculator Execution ---
    calc_inputs: CalculationInputs | None = None
    calc_result: CalculationResult | None = None

    if plan.get("needs_calculation"):
        calc_start = time.perf_counter()
        if has_itc_balances:
            if legal_findings is not None and legal_findings.status == "unresolved":
                tools_executed.append({
                    "tool": "execute_calculator",
                    "status": "skipped",
                    "reason": "unresolved_legal_findings",
                    "details": legal_findings.unresolved_reason,
                })
            elif legal_findings is not None and legal_findings.status == "resolved":
                tax_rate = extract_rate_pct(rate_results, user_premises)
                if tax_rate and tax_rate > 0 and legal_findings.total_usable_credit is not None:
                    calc_inputs = CalculationInputs(
                        operation="max_taxable_value_from_credit",
                        available_eligible_credit=legal_findings.total_usable_credit,
                        tax_rate_pct=tax_rate,
                        credit_breakdown=legal_findings.usable_credit_balances,
                    )
                    calc_result = execute_calculator(calc_inputs)
                    calc_timing = round((time.perf_counter() - calc_start) * 1000, 3)
                    tools_executed.append({
                        "tool": "execute_calculator",
                        "operation": calc_inputs.operation,
                        "status": calc_result.status,
                        "result_value": calc_result.result_value,
                        "formula": calc_result.formula,
                        "timing_ms": calc_timing,
                    })
        elif user_premises.get("taxable_amount") is not None or user_premises.get("base_amount") is not None:
            base_amt = float(user_premises.get("taxable_amount") or user_premises.get("base_amount") or 0.0)
            disc_pct = float(user_premises.get("discount_pct") or 0.0)
            tax_rate = extract_rate_pct(rate_results, user_premises) or 5.0

            if disc_pct > 0:
                calc_inputs = CalculationInputs(
                    operation="discount_and_tax",
                    base_amount=base_amt,
                    discount_pct=disc_pct,
                    tax_rate_pct=tax_rate,
                )
            else:
                calc_inputs = CalculationInputs(
                    operation="tax_on_value",
                    taxable_value=base_amt,
                    tax_rate_pct=tax_rate,
                )
            calc_result = execute_calculator(calc_inputs)
            calc_timing = round((time.perf_counter() - calc_start) * 1000, 3)
            tools_executed.append({
                "tool": "execute_calculator",
                "operation": calc_inputs.operation,
                "status": calc_result.status,
                "result_value": calc_result.result_value,
                "formula": calc_result.formula,
                "timing_ms": calc_timing,
            })

    gen_data = generate_answer(
        query=query,
        chunks=legal_chunks,
        notification_chunks=notif_chunks,
        rate_results=rate_results,
        user_premises=plan.get("user_premises"),
        legal_findings=legal_findings,
        calculation_result=calc_result,
        model=openai_model,
        client=openai_client,
        direct_reasoning=is_direct,
    )

    generation_timing = gen_data["generation_timing"]
    total_timing = round(retrieval_timing + generation_timing, 3)

    observability = {
        "plan": plan,
        "tools_executed": tools_executed,
        "retrieved_evidence": {
            "rates_count": len(rate_results),
            "legal_chunks_count": len(legal_chunks),
            "notification_chunks_count": len(notif_chunks),
        },
        "structured_legal_findings": (
            legal_findings.model_dump() if legal_findings else None
        ),
        "calculation_inputs": (
            calc_inputs.model_dump() if calc_inputs else None
        ),
        "calculation_result": (
            calc_result.model_dump() if calc_result else None
        ),
        "sources_used": gen_data["sources"],
        "knowledge_base_gap": (
            legal_findings.unresolved_reason
            if (legal_findings and legal_findings.status == "unresolved")
            else None
        ),
    }

    return {
        "query": query,
        "route": route_str,
        "answer": gen_data["answer"],
        "rate_results": rate_results,
        "sources": gen_data["sources"],
        "sources_used": gen_data["sources"],
        "retrieval_timing": retrieval_timing,
        "generation_timing": generation_timing,
        "total_timing": total_timing,
        "model_used": gen_data["model_used"],
        "model": gen_data["model_used"],
        "plan": plan,
        "tools_executed": tools_executed,
        "structured_legal_findings": (
            legal_findings.model_dump() if legal_findings else None
        ),
        "calculation_inputs": (
            calc_inputs.model_dump() if calc_inputs else None
        ),
        "calculation_result": (
            calc_result.model_dump() if calc_result else None
        ),
        "observability": observability,
        "timings_ms": {
            "retrieval": retrieval_timing,
            "generation": generation_timing,
            "total": total_timing,
            "rate_lookup": rate_timing,
            "notification_lookup": notif_timing,
            "dense": retrieval_data["timings_ms"].get("dense", 0.0),
            "bm25": retrieval_data["timings_ms"].get("bm25", 0.0),
            "rrf": retrieval_data["timings_ms"].get("rrf", 0.0),
            "reranker": retrieval_data["timings_ms"].get("reranker", 0.0),
        },
        "retrieval_debug": {
            "results": final_chunks,
            "dense_results": retrieval_data.get("dense_results", []),
            "bm25_results": retrieval_data.get("bm25_results", []),
            "hybrid_results": retrieval_data.get("hybrid_results", []),
            "metadata": retrieval_data.get("metadata", {}),
            "models": retrieval_data.get("models", {}),
            "config": retrieval_data.get("config", {}),
        },
    }


def stream_gst_answer_flow(
    query: str,
    top_k: int = DEFAULT_ANSWER_TOP_K,
    *,
    models: LoadedModels,
    openai_model: str | None = None,
    openai_client: OpenAI | None = None,
    db_url: str | None = None,
    route_override: str | None = None,
):
    """[LEGACY] Stream retrieval and generation events for real-time live answers.

    DEPRECATED: Use `src.graph.stream_graph_chat` instead.
    """
    plan = plan_capabilities(query)
    model_name = get_configured_model(openai_model)

    if plan.get("needs_clarification") and not route_override:
        clarification_text = plan.get("clarification_prompt") or "Please provide more details."
        obs = {
            "plan": plan,
            "tools_executed": [],
            "retrieved_evidence": {"rates_count": 0, "legal_chunks_count": 0, "notification_chunks_count": 0},
            "structured_legal_findings": None,
            "calculation_inputs": None,
            "calculation_result": None,
            "sources_used": [],
            "knowledge_base_gap": None,
        }
        yield {
            "type": "meta",
            "query": query,
            "route": "clarification",
            "rate_results": [],
            "sources": [],
            "sources_used": [],
            "retrieval_timing": 0.0,
            "model_used": model_name,
            "model": model_name,
            "plan": plan,
            "tools_executed": [],
            "structured_legal_findings": None,
            "calculation_inputs": None,
            "calculation_result": None,
            "observability": obs,
            "retrieval_debug": {},
        }
        yield {
            "type": "token",
            "delta": clarification_text,
        }
        yield {
            "type": "done",
            "answer": clarification_text,
            "route": "clarification",
            "rate_results": [],
            "generation_timing": 0.0,
            "total_timing": 0.0,
            "plan": plan,
            "tools_executed": [],
            "structured_legal_findings": None,
            "calculation_inputs": None,
            "calculation_result": None,
            "observability": obs,
            "timings_ms": {"retrieval": 0.0, "generation": 0.0, "total": 0.0},
        }
        return

    if route_override:
        route = RouteType(route_override)
        needs_rate = route in (RouteType.RATE, RouteType.MIXED)
        needs_legal = route in (RouteType.LEGAL, RouteType.MIXED)
        needs_notif = False
        is_direct = route == RouteType.DIRECT
        route_str = route.value
    else:
        needs_rate = plan.get("needs_structured_rate_lookup", False) or plan.get("needs_hsn_lookup", False)
        needs_legal = plan.get("needs_legal_retrieval", False)
        needs_notif = plan.get("needs_notification_retrieval", False)
        is_direct = (
            plan.get("needs_direct_reasoning", False)
            and not needs_rate
            and not needs_legal
            and not needs_notif
        )
        if is_direct:
            route_str = "direct"
        elif needs_rate and (needs_legal or needs_notif):
            route_str = "mixed"
        elif needs_rate:
            route_str = "rate"
        elif needs_legal or needs_notif:
            route_str = "legal"
        else:
            route_str = "legal"

    rate_results: list[dict[str, Any]] = []
    legal_chunks: list[dict[str, Any]] = []
    notif_chunks: list[dict[str, Any]] = []
    rate_timing = 0.0
    notif_timing = 0.0
    tools_executed: list[dict[str, Any]] = []

    retrieval_data: dict[str, Any] = {
        "results": [],
        "dense_results": [],
        "bm25_results": [],
        "hybrid_results": [],
        "metadata": {},
        "models": {},
        "config": {},
        "timings_ms": {"total": 0.0, "dense": 0.0, "bm25": 0.0, "rrf": 0.0, "reranker": 0.0},
    }

    if needs_rate:
        rate_start = time.perf_counter()
        rate_q = plan.get("clean_subqueries", {}).get("rate_query") or query
        rate_results = retrieve_rates(rate_q, db_url=db_url, limit=top_k)
        rate_timing = round((time.perf_counter() - rate_start) * 1000, 3)
        tools_executed.append({
            "tool": "retrieve_rates",
            "query": rate_q,
            "results_count": len(rate_results),
            "timing_ms": rate_timing,
        })

    if needs_legal:
        legal_q = plan.get("clean_subqueries", {}).get("legal_query") or query
        retrieval_data = inspect_retrieval(legal_q, top_k=top_k, models=models, db_url=db_url)
        legal_chunks = retrieval_data.get("results") or []
        legal_timing = retrieval_data.get("timings_ms", {}).get("total", 0.0)
        tools_executed.append({
            "tool": "inspect_retrieval",
            "query": legal_q,
            "results_count": len(legal_chunks),
            "timing_ms": legal_timing,
        })

    if needs_notif:
        notif_start = time.perf_counter()
        notif_q = plan.get("clean_subqueries", {}).get("notification_query") or query
        notif_chunks = retrieve_notifications(
            notif_q,
            top_k=top_k,
            db_url=db_url,
            model=models.embedding_model if models else None,
        )
        notif_timing = round((time.perf_counter() - notif_start) * 1000, 3)
        tools_executed.append({
            "tool": "retrieve_notifications",
            "query": notif_q,
            "results_count": len(notif_chunks),
            "timing_ms": notif_timing,
        })

    legal_timing = retrieval_data.get("timings_ms", {}).get("total", 0.0)
    retrieval_timing = round(rate_timing + legal_timing + notif_timing, 3)
    final_chunks = legal_chunks + notif_chunks

    user_premises = plan.get("user_premises") or {}
    has_itc_balances = bool(user_premises.get("itc_balances"))
    legal_findings: StructuredLegalFindings | None = None

    if needs_legal and (plan.get("needs_calculation") or has_itc_balances):
        interp_start = time.perf_counter()
        legal_findings = interpret_legal_findings(
            legal_chunks=legal_chunks,
            user_premises=user_premises,
            query=query,
        )
        interp_timing = round((time.perf_counter() - interp_start) * 1000, 3)
        tools_executed.append({
            "tool": "interpret_legal_findings",
            "status": legal_findings.status,
            "supply_type": legal_findings.supply_type,
            "usable_credit_ledgers": legal_findings.usable_credit_ledgers,
            "total_usable_credit": legal_findings.total_usable_credit,
            "unresolved_reason": legal_findings.unresolved_reason,
            "timing_ms": interp_timing,
        })

    calc_inputs: CalculationInputs | None = None
    calc_result: CalculationResult | None = None

    if plan.get("needs_calculation"):
        calc_start = time.perf_counter()
        if has_itc_balances:
            if legal_findings is not None and legal_findings.status == "unresolved":
                tools_executed.append({
                    "tool": "execute_calculator",
                    "status": "skipped",
                    "reason": "unresolved_legal_findings",
                    "details": legal_findings.unresolved_reason,
                })
            elif legal_findings is not None and legal_findings.status == "resolved":
                tax_rate = extract_rate_pct(rate_results, user_premises)
                if tax_rate and tax_rate > 0 and legal_findings.total_usable_credit is not None:
                    calc_inputs = CalculationInputs(
                        operation="max_taxable_value_from_credit",
                        available_eligible_credit=legal_findings.total_usable_credit,
                        tax_rate_pct=tax_rate,
                        credit_breakdown=legal_findings.usable_credit_balances,
                    )
                    calc_result = execute_calculator(calc_inputs)
                    calc_timing = round((time.perf_counter() - calc_start) * 1000, 3)
                    tools_executed.append({
                        "tool": "execute_calculator",
                        "operation": calc_inputs.operation,
                        "status": calc_result.status,
                        "result_value": calc_result.result_value,
                        "formula": calc_result.formula,
                        "timing_ms": calc_timing,
                    })
        elif user_premises.get("taxable_amount") is not None or user_premises.get("base_amount") is not None:
            base_amt = float(user_premises.get("taxable_amount") or user_premises.get("base_amount") or 0.0)
            disc_pct = float(user_premises.get("discount_pct") or 0.0)
            tax_rate = extract_rate_pct(rate_results, user_premises) or 5.0

            if disc_pct > 0:
                calc_inputs = CalculationInputs(
                    operation="discount_and_tax",
                    base_amount=base_amt,
                    discount_pct=disc_pct,
                    tax_rate_pct=tax_rate,
                )
            else:
                calc_inputs = CalculationInputs(
                    operation="tax_on_value",
                    taxable_value=base_amt,
                    tax_rate_pct=tax_rate,
                )
            calc_result = execute_calculator(calc_inputs)
            calc_timing = round((time.perf_counter() - calc_start) * 1000, 3)
            tools_executed.append({
                "tool": "execute_calculator",
                "operation": calc_inputs.operation,
                "status": calc_result.status,
                "result_value": calc_result.result_value,
                "formula": calc_result.formula,
                "timing_ms": calc_timing,
            })

    from src.generators.answer_generator import extract_combined_sources

    sources = extract_combined_sources(final_chunks, rate_results)

    observability = {
        "plan": plan,
        "tools_executed": tools_executed,
        "retrieved_evidence": {
            "rates_count": len(rate_results),
            "legal_chunks_count": len(legal_chunks),
            "notification_chunks_count": len(notif_chunks),
        },
        "structured_legal_findings": (
            legal_findings.model_dump() if legal_findings else None
        ),
        "calculation_inputs": (
            calc_inputs.model_dump() if calc_inputs else None
        ),
        "calculation_result": (
            calc_result.model_dump() if calc_result else None
        ),
        "sources_used": sources,
        "knowledge_base_gap": (
            legal_findings.unresolved_reason
            if (legal_findings and legal_findings.status == "unresolved")
            else None
        ),
    }

    yield {
        "type": "meta",
        "query": query,
        "route": route_str,
        "rate_results": rate_results,
        "sources": sources,
        "sources_used": sources,
        "retrieval_timing": retrieval_timing,
        "model_used": model_name,
        "model": model_name,
        "plan": plan,
        "tools_executed": tools_executed,
        "structured_legal_findings": (
            legal_findings.model_dump() if legal_findings else None
        ),
        "calculation_inputs": (
            calc_inputs.model_dump() if calc_inputs else None
        ),
        "calculation_result": (
            calc_result.model_dump() if calc_result else None
        ),
        "observability": observability,
        "retrieval_debug": {
            "results": final_chunks,
            "dense_results": retrieval_data.get("dense_results", []),
            "bm25_results": retrieval_data.get("bm25_results", []),
            "hybrid_results": retrieval_data.get("hybrid_results", []),
            "metadata": retrieval_data.get("metadata", {}),
            "models": retrieval_data.get("models", {}),
            "config": retrieval_data.get("config", {}),
        },
    }

    # 2. Generation streaming
    gen_start = time.perf_counter()
    full_answer_parts: list[str] = []

    for _, delta in stream_answer(
        query=query,
        chunks=legal_chunks,
        notification_chunks=notif_chunks,
        rate_results=rate_results,
        user_premises=plan.get("user_premises"),
        legal_findings=legal_findings,
        calculation_result=calc_result,
        model=openai_model,
        client=openai_client,
        direct_reasoning=is_direct,
    ):
        full_answer_parts.append(delta)
        yield {
            "type": "token",
            "delta": delta,
        }

    gen_ms = round((time.perf_counter() - gen_start) * 1000, 3)
    total_ms = round(retrieval_timing + gen_ms, 3)
    full_answer = "".join(full_answer_parts).strip()

    yield {
        "type": "done",
        "answer": full_answer,
        "route": route_str,
        "rate_results": rate_results,
        "generation_timing": gen_ms,
        "total_timing": total_ms,
        "plan": plan,
        "tools_executed": tools_executed,
        "structured_legal_findings": (
            legal_findings.model_dump() if legal_findings else None
        ),
        "calculation_inputs": (
            calc_inputs.model_dump() if calc_inputs else None
        ),
        "calculation_result": (
            calc_result.model_dump() if calc_result else None
        ),
        "observability": observability,
        "timings_ms": {
            "retrieval": retrieval_timing,
            "generation": gen_ms,
            "total": total_ms,
            "rate_lookup": rate_timing,
            "notification_lookup": notif_timing,
            "dense": retrieval_data["timings_ms"].get("dense", 0.0),
            "bm25": retrieval_data["timings_ms"].get("bm25", 0.0),
            "rrf": retrieval_data["timings_ms"].get("rrf", 0.0),
            "reranker": retrieval_data["timings_ms"].get("reranker", 0.0),
        },
    }
