"""LangGraph workflow definition for the GST Assistant pipeline.

Wired according to the target architecture:
START
  ↓
planner
  ↓
conditional routing (route_capabilities)
  ├── legal_retrieval ──┐
  ├── rate_lookup ──────┼──→ synthesis ──→ END
  └── direct_reasoning ─┘
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import re
import time
from typing import Any, Generator, Optional
import uuid

from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from openai import OpenAI

from src.generators.answer_generator import (
    extract_combined_sources,
    get_configured_model,
    stream_answer,
)
from src.graph.nodes import (
    calculation_node,
    direct_reasoning_node,
    get_cached_models,
    grounded_reasoning_node,
    legal_retrieval_node,
    notification_support_node,
    planner_node,
    rate_lookup_node,
    synthesis_node,
)
from src.graph.routing import (
    route_capabilities,
    route_post_direct_reasoning,
    route_post_grounded_reasoning,
    route_post_notification_support,
    route_post_retrieval,
)
from src.graph.state import GSTGraphState
from src.observability import trace_store
from src.retrieval_inspector import LoadedModels, inspect_retrieval
from src.retrievers.rate_retriever import retrieve_rates
from src.routers.planner import plan_capabilities
from src.tools.calculator import CalculationInputs, execute_calculator

logger = logging.getLogger(__name__)

# Shared in-memory checkpointer instance for state persistence across graph executions
shared_checkpointer = MemorySaver()


def build_gst_graph(
    *,
    models: Optional[LoadedModels] = None,
    openai_client: Optional[OpenAI] = None,
    openai_model: Optional[str] = None,
    db_url: Optional[str] = None,
    use_llm_planner: Optional[bool] = None,
) -> StateGraph:
    """Constructs and returns the uncompiled LangGraph StateGraph builder.

    Args:
        models: Pre-loaded BGE-M3 and reranker models (optional, lazy-loaded if None).
        openai_client: Configured OpenAI client for generation (optional).
        openai_model: Target model name (e.g. gpt-4o-mini or gemini-3.1-flash-lite).
        db_url: PostgreSQL connection string (optional, reads from env if None).
        use_llm_planner: Whether to force LLM planner instead of deterministic heuristics.
    """
    builder = StateGraph(GSTGraphState)

    # 1. Register Nodes
    builder.add_node(
        "planner",
        lambda state: planner_node(state, use_llm=use_llm_planner),
    )
    builder.add_node(
        "legal_retrieval",
        lambda state: legal_retrieval_node(state, models=models, db_url=db_url),
    )
    builder.add_node(
        "rate_lookup",
        lambda state: rate_lookup_node(state, db_url=db_url),
    )
    builder.add_node(
        "notification_support",
        lambda state: notification_support_node(state, models=models, db_url=db_url),
    )
    builder.add_node(
        "grounded_reasoning",
        lambda state: grounded_reasoning_node(state),
    )
    builder.add_node(
        "direct_reasoning",
        lambda state: direct_reasoning_node(state),
    )
    builder.add_node(
        "calculation",
        lambda state: calculation_node(state),
    )
    builder.add_node(
        "synthesis",
        lambda state: synthesis_node(state, client=openai_client, model=openai_model),
    )

    # 2. Add Start Edge
    builder.add_edge(START, "planner")

    # 3. Add Conditional Routing from Planner
    builder.add_conditional_edges(
        "planner",
        route_capabilities,
        ["legal_retrieval", "rate_lookup", "notification_support", "direct_reasoning", "calculation", "synthesis"],
    )

    # 4. Primary retrieval nodes route to notification_support for supporting Gazette evidence
    builder.add_edge("legal_retrieval", "notification_support")
    builder.add_edge("rate_lookup", "notification_support")

    # 5. Notification support routes to grounded_reasoning, calculation, or synthesis
    builder.add_conditional_edges(
        "notification_support",
        route_post_notification_support,
        ["grounded_reasoning", "calculation", "synthesis"],
    )

    # 5. Grounded reasoning conditional edge
    builder.add_conditional_edges(
        "grounded_reasoning",
        route_post_grounded_reasoning,
        ["calculation", "synthesis"],
    )

    # 6. Direct reasoning conditional edge
    builder.add_conditional_edges(
        "direct_reasoning",
        route_post_direct_reasoning,
        ["calculation", "synthesis"],
    )

    # 7. Deterministic calculation flows into synthesis
    builder.add_edge("calculation", "synthesis")

    # 8. Synthesis completes the graph
    builder.add_edge("synthesis", END)

    return builder


def compile_gst_graph(
    *,
    models: Optional[LoadedModels] = None,
    openai_client: Optional[OpenAI] = None,
    openai_model: Optional[str] = None,
    db_url: Optional[str] = None,
    use_llm_planner: Optional[bool] = None,
) -> CompiledStateGraph:
    """Builds and compiles the GST LangGraph.

    Returns:
        A compiled runnable graph that can be invoked with .invoke(state) or .stream(state).
    """
    builder = build_gst_graph(
        models=models,
        openai_client=openai_client,
        openai_model=openai_model,
        db_url=db_url,
        use_llm_planner=use_llm_planner,
    )
    return builder.compile(checkpointer=shared_checkpointer)


def invoke_gst_graph(
    query: str,
    *,
    execution_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    models: Optional[LoadedModels] = None,
    openai_client: Optional[OpenAI] = None,
    openai_model: Optional[str] = None,
    db_url: Optional[str] = None,
    use_llm_planner: Optional[bool] = None,
) -> GSTGraphState:
    """Convenience helper to invoke the GST LangGraph directly with a user query string.

    Registers and updates structured traces in trace_store.
    """
    if not execution_id:
        execution_id = str(uuid.uuid4())
    if not thread_id:
        thread_id = str(uuid.uuid4())

    trace_store.create_trace(execution_id, query)

    graph = compile_gst_graph(
        models=models,
        openai_client=openai_client,
        openai_model=openai_model,
        db_url=db_url,
        use_llm_planner=use_llm_planner,
    )

    initial_state: GSTGraphState = {
        "user_query": query,
        "execution_id": execution_id,
        "thread_id": thread_id,
        "needs_legal": False,
        "needs_rate": False,
        "needs_notification": False,
        "needs_direct_reasoning": False,
        "needs_calculation": False,
        "needs_grounded_reasoning": False,
        "clean_queries": {},
        "user_premises": {},
        "legal_results": [],
        "rate_results": [],
        "notification_results": [],
        "reasoning_result": None,
        "calculation_inputs": None,
        "calculation_result": None,
        "final_answer": "",
        "sources": [],
        "route": None,
        "plan": None,
        "timings_ms": None,
        "model_used": None,
        "error": None,
    }

    t0 = time.perf_counter()
    err_msg: Optional[str] = None
    try:
        res = graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )
        res["execution_id"] = execution_id
        res["thread_id"] = thread_id
        return res
    except Exception as exc:
        err_msg = str(exc)
        raise
    finally:
        total_ms = round((time.perf_counter() - t0) * 1000, 3)
        trace_store.finalize(
            execution_id,
            total_request_ms=total_ms,
            status="error" if err_msg else "success",
            error=err_msg,
        )


def run_graph_chat(
    query: str,
    top_k: int = 10,
    *,
    execution_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    models: Optional[LoadedModels] = None,
    openai_model: Optional[str] = None,
    openai_client: Optional[OpenAI] = None,
    db_url: Optional[str] = None,
) -> dict[str, Any]:
    """Production runtime entry point for /chat and /answer endpoints via LangGraph.

    Returns the standard response payload expected by the API and Web UI, including execution_id and thread_id.
    """
    if not execution_id:
        execution_id = str(uuid.uuid4())
    if not thread_id:
        thread_id = str(uuid.uuid4())

    start_time = time.perf_counter()
    output_state = invoke_gst_graph(
        query=query,
        execution_id=execution_id,
        thread_id=thread_id,
        models=models,
        openai_client=openai_client,
        openai_model=openai_model,
        db_url=db_url,
    )
    total_elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    plan = output_state.get("plan") or {}
    route_str = output_state.get("route") or "direct"
    rate_results = output_state.get("rate_results", [])
    legal_results = output_state.get("legal_results", [])
    sources = output_state.get("sources", [])
    final_answer = output_state.get("final_answer", "")
    model_name = output_state.get("model_used") or get_configured_model(openai_model)

    return {
        "execution_id": execution_id,
        "thread_id": thread_id,
        "query": query,
        "route": route_str,
        "answer": final_answer,
        "rate_results": rate_results,
        "sources": sources,
        "sources_used": sources,
        "retrieval_timing": 0.0,
        "generation_timing": total_elapsed_ms,
        "total_timing": total_elapsed_ms,
        "model_used": model_name,
        "model": model_name,
        "plan": plan,
        "timings_ms": {
            "total": total_elapsed_ms,
            "retrieval": 0.0,
            "generation": total_elapsed_ms,
        },
        "retrieval_debug": {
            "results": legal_results,
            "dense_results": [],
            "bm25_results": [],
            "hybrid_results": [],
            "metadata": {},
        },
    }


def stream_graph_chat(
    query: str,
    top_k: int = 10,
    *,
    execution_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    models: Optional[LoadedModels] = None,
    openai_model: Optional[str] = None,
    openai_client: Optional[OpenAI] = None,
    db_url: Optional[str] = None,
) -> Generator[dict[str, Any], None, None]:
    """Production runtime entry point for /chat/stream SSE endpoint via LangGraph.

    Executes planning and retrieval via graph tools, emits metadata SSE events,
    streams synthesis tokens in real-time, and yields a terminal 'done' event.
    Records structured traces in trace_store.
    """
    if not execution_id:
        execution_id = str(uuid.uuid4())
    if not thread_id:
        thread_id = str(uuid.uuid4())

    total_start = time.perf_counter()
    trace_store.create_trace(execution_id, query)

    # 1. Planner
    t_plan0 = time.perf_counter()
    compiled_g = compile_gst_graph(
        models=models,
        openai_client=openai_client,
        openai_model=openai_model,
        db_url=db_url,
    )
    thread_state = compiled_g.get_state({"configurable": {"thread_id": thread_id}})
    prior_messages = thread_state.values.get("messages", []) if thread_state and thread_state.values else []

    plan = plan_capabilities(query, history=prior_messages)
    plan_ms = round((time.perf_counter() - t_plan0) * 1000, 3)
    model_name = get_configured_model(openai_model)

    needs_rate = bool(
        plan.get("needs_structured_rate_lookup") or plan.get("needs_hsn_lookup")
    )
    needs_legal = bool(plan.get("needs_legal_retrieval"))
    needs_calculation = bool(plan.get("needs_calculation", False))
    user_premises = plan.get("user_premises", {}) or {}

    needs_grounded = bool(
        (needs_legal or needs_rate) and (
            needs_calculation
            or plan.get("needs_temporal_reasoning", False)
            or plan.get("needs_comparison", False)
            or plan.get("needs_exception_reasoning", False)
            or bool(user_premises.get("itc_balances"))
            or (bool(user_premises) and plan.get("needs_direct_reasoning", False))
        )
    )
    needs_direct = bool(
        not (needs_legal or needs_rate) and (
            plan.get("needs_direct_reasoning", False)
            or (not needs_calculation and bool(user_premises))
        )
    )

    # Route resolution
    if not needs_rate and not needs_legal:
        route_str = "direct"
    elif needs_rate and needs_legal:
        route_str = "mixed"
    elif needs_rate:
        route_str = "rate"
    elif needs_legal:
        route_str = "legal"
    else:
        route_str = "direct"

    clean_queries = plan.get("clean_subqueries", {})
    clean_subq_with_route = dict(clean_queries)
    clean_subq_with_route["route"] = route_str

    trace_store.record_planner(
        execution_id,
        capability_flags={
            "needs_legal": needs_legal,
            "needs_rate": needs_rate,
            "needs_direct_reasoning": needs_direct,
            "needs_calculation": needs_calculation,
            "needs_grounded_reasoning": needs_grounded,
            "needs_clarification": bool(plan.get("needs_clarification", False)),
        },
        clean_subqueries=clean_subq_with_route,
        extracted_user_premises=dict(user_premises),
        clarification_decision={
            "needs_clarification": bool(plan.get("needs_clarification", False)),
            "clarification_prompt": plan.get("clarification_prompt"),
        },
        timing_ms=plan_ms,
    )
    needs_notification = bool(plan.get("needs_notification_retrieval", False))
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
        timing_ms=plan_ms,
    )

    current_state: GSTGraphState = {
        "user_query": query,
        "execution_id": execution_id,
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
        "final_answer": "",
        "sources": [],
        "route": route_str,
        "plan": plan,
    }

    # 2. Primary retrievals (rate lookup & legal retrieval)
    if needs_rate and needs_legal:
        retrieval_models = models or get_cached_models()
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_rate = executor.submit(rate_lookup_node, current_state, db_url=db_url, limit=top_k)
            fut_legal = executor.submit(legal_retrieval_node, current_state, models=retrieval_models, db_url=db_url, top_k=top_k)
            current_state.update(fut_rate.result())
            current_state.update(fut_legal.result())
    elif needs_rate:
        rate_out = rate_lookup_node(current_state, db_url=db_url, limit=top_k)
        current_state.update(rate_out)
    elif needs_legal:
        retrieval_models = models or get_cached_models()
        legal_out = legal_retrieval_node(current_state, models=retrieval_models, db_url=db_url, top_k=top_k)
        current_state.update(legal_out)

    # 3b. Notification support lookup (automatic for rate/legal or explicit notification query)
    if needs_rate or needs_legal or needs_notification:
        retrieval_models = models or get_cached_models()
        notif_out = notification_support_node(current_state, models=retrieval_models, db_url=db_url, top_k=2)
        current_state.update(notif_out)

    # 4. Grounded reasoning (dependent on retrievals)
    if needs_grounded:
        grounded_out = grounded_reasoning_node(current_state)
        current_state.update(grounded_out)

    # 5. Direct reasoning (retrieval-free)
    if needs_direct:
        direct_out = direct_reasoning_node(current_state)
        current_state.update(direct_out)

    # 6. Deterministic calculation
    if needs_calculation:
        calc_out = calculation_node(current_state)
        current_state.update(calc_out)

    legal_results = current_state.get("legal_results", [])
    rate_results = current_state.get("rate_results", [])
    notification_results = current_state.get("notification_results", [])
    calc_res = current_state.get("calculation_result")
    reasoning_res = current_state.get("reasoning_result")

    all_chunks = list(legal_results) + list(notification_results)
    sources = extract_combined_sources(all_chunks, rate_results)
    retrieval_ms = round((time.perf_counter() - total_start) * 1000, 2)

    # 1. Yield metadata event
    yield {
        "type": "meta",
        "execution_id": execution_id,
        "thread_id": thread_id,
        "query": query,
        "route": route_str,
        "rate_results": rate_results,
        "notification_results": notification_results,
        "sources": sources,
        "sources_used": sources,
        "retrieval_timing": retrieval_ms,
        "model_used": model_name,
        "model": model_name,
        "plan": plan,
        "timings_ms": {
            "retrieval": retrieval_ms,
        },
    }

    # 2. Stream generation tokens
    gen_start = time.perf_counter()
    full_answer_parts: list[str] = []
    is_direct = route_str == "direct"

    try:
        for _, delta in stream_answer(
            query=query,
            chunks=legal_results if legal_results else None,
            rate_results=rate_results if rate_results else None,
            notification_chunks=notification_results if notification_results else None,
            user_premises=plan.get("user_premises"),
            calculation_result=calc_res,
            model=openai_model,
            client=openai_client,
            direct_reasoning=is_direct,
            history=prior_messages,
        ):
            full_answer_parts.append(delta)
            # Break large incoming deltas into word-level pieces with calm pacing
            pieces = re.findall(r"\S+|\s+", delta)
            for p in pieces:
                yield {
                    "type": "token",
                    "delta": p,
                }
                time.sleep(0.015)
        full_answer = "".join(full_answer_parts).strip()
    except Exception:
        # Offline or mock fallback synthesis
        from src.graph.nodes import _build_deterministic_synthesis

        full_answer = _build_deterministic_synthesis(
            query=query,
            rate_results=rate_results,
            legal_results=legal_results,
            notification_results=notification_results,
            reasoning_result=reasoning_res,
            calculation_result=calc_res,
        )
        pieces = re.findall(r"\S+|\s+", full_answer)
        for p in pieces:
            yield {
                "type": "token",
                "delta": p,
            }
            time.sleep(0.015)

    gen_ms = round((time.perf_counter() - gen_start) * 1000, 2)
    total_ms = round(retrieval_ms + gen_ms, 2)

    # Commit messages to checkpointer for subsequent turns
    try:
        from langchain_core.messages import AIMessage, HumanMessage
        compiled_g.update_state(
            {"configurable": {"thread_id": thread_id}},
            {"messages": [HumanMessage(content=query), AIMessage(content=full_answer)]},
        )
    except Exception:
        pass

    trace_store.record_synthesis(
        execution_id,
        sources_provided=sources,
        model_name=model_name,
        generation_timing_ms=gen_ms,
        final_answer=full_answer,
    )
    trace_store.record_node_execution(
        execution_id,
        "synthesis",
        status="success",
        timing_ms=gen_ms,
    )
    trace_store.finalize(
        execution_id,
        total_request_ms=total_ms,
        status="success",
    )

    # 3. Yield done event
    yield {
        "type": "done",
        "execution_id": execution_id,
        "thread_id": thread_id,
        "answer": full_answer,
        "route": route_str,
        "rate_results": rate_results,
        "generation_timing": gen_ms,
        "total_timing": total_ms,
        "plan": plan,
        "timings_ms": {
            "retrieval": retrieval_ms,
            "generation": gen_ms,
            "total": total_ms,
        },
    }
