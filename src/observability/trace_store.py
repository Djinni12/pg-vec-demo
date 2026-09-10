"""In-memory bounded trace store for GST Bot developer observability.

Tracks per-request structured traces across the LangGraph execution flow:
- Original user query & unique execution_id
- Planner capability flags, clean subqueries, extracted user premises
- LangGraph executed nodes sequence, status, per-node timing
- Legal hybrid retrieval candidates (dense, BM25, RRF, reranker, final chunks)
- Structured rate lookup candidates, selected rate used downstream
- Deterministic calculation inputs, rate source, operation, formula, verified result
- Synthesis model name, timing, provided sources, final answer
- Overall timing waterfall

Never stores or exposes model internal chain-of-thought.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import threading
from typing import Any, Optional


@dataclass
class ExecutionTrace:
    """Structured developer execution trace for a single user query."""

    execution_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    query: str = ""
    status: str = "running"  # 'running', 'success', 'error'
    error: Optional[str] = None

    # 1. Planner Decisions
    planner: dict[str, Any] = field(default_factory=lambda: {
        "capability_flags": {
            "needs_legal": False,
            "needs_rate": False,
            "needs_direct_reasoning": False,
            "needs_calculation": False,
            "needs_grounded_reasoning": False,
            "needs_clarification": False,
        },
        "clean_subqueries": {
            "legal_query": None,
            "rate_query": None,
            "route": None,
        },
        "extracted_user_premises": {},
        "clarification_decision": {
            "needs_clarification": False,
            "clarification_prompt": None,
        },
        "timing_ms": 0.0,
    })

    # 2. LangGraph Execution Flow
    graph_execution: dict[str, Any] = field(default_factory=lambda: {
        "selected_nodes": [],
        "actual_executed_nodes": [],
        "execution_order": [],
        "node_status": {},
        "node_timings_ms": {},
    })

    # 3. Legal Retrieval (Hybrid Inspection)
    legal_retrieval: dict[str, Any] = field(default_factory=lambda: {
        "retrieval_query": "",
        "dense_candidates": [],
        "bm25_candidates": [],
        "rrf_candidates": [],
        "reranker_results": [],
        "final_chunks": [],
        "document_metadata": {},
        "timing_ms": 0.0,
        "stages_timings_ms": {
            "dense": 0.0,
            "bm25": 0.0,
            "rrf": 0.0,
            "reranker": 0.0,
            "total": 0.0,
        },
    })

    # 4. Structured Rate Retrieval
    rate_retrieval: dict[str, Any] = field(default_factory=lambda: {
        "lookup_query": "",
        "returned_candidates": [],
        "selected_rate_used_downstream": None,
        "timing_ms": 0.0,
    })

    # 5. Reasoning Stages
    reasoning: dict[str, Any] = field(default_factory=lambda: {
        "grounded_reasoning_output": None,
        "direct_reasoning_output": None,
        "calculation_inputs": None,
        "grounded_timing_ms": 0.0,
        "direct_timing_ms": 0.0,
    })

    # 6. Calculation
    calculation: dict[str, Any] = field(default_factory=lambda: {
        "calculation_inputs": None,
        "rate_used": None,
        "source_of_rate": None,
        "operation": None,
        "deterministic_result": None,
        "timing_ms": 0.0,
    })

    # 7. Synthesis
    synthesis: dict[str, Any] = field(default_factory=lambda: {
        "sources_provided": [],
        "model_name": "",
        "generation_timing_ms": 0.0,
        "final_answer": "",
    })

    # 8. Overall Timings Waterfall
    timings: dict[str, float] = field(default_factory=lambda: {
        "planner_ms": 0.0,
        "dense_ms": 0.0,
        "bm25_ms": 0.0,
        "rrf_ms": 0.0,
        "reranker_ms": 0.0,
        "rate_lookup_ms": 0.0,
        "grounded_reasoning_ms": 0.0,
        "direct_reasoning_ms": 0.0,
        "calculation_ms": 0.0,
        "synthesis_ms": 0.0,
        "total_request_ms": 0.0,
    })

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to a deep-copied JSON-serializable dictionary."""
        return deepcopy(asdict(self))


class TraceStore:
    """Thread-safe bounded in-memory store of execution traces."""

    def __init__(self, max_size: int = 200) -> None:
        self.max_size = max_size
        self._traces: dict[str, ExecutionTrace] = {}
        self._order: deque[str] = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def create_trace(self, execution_id: str, query: str) -> ExecutionTrace:
        """Initialize and register a new execution trace."""
        with self._lock:
            if len(self._order) >= self.max_size and execution_id not in self._traces:
                oldest_id = self._order.popleft()
                self._traces.pop(oldest_id, None)

            trace = ExecutionTrace(execution_id=execution_id, query=query)
            self._traces[execution_id] = trace
            if execution_id not in self._order:
                self._order.append(execution_id)
            return trace

    def get_trace(self, execution_id: str) -> Optional[dict[str, Any]]:
        """Retrieve full trace dictionary by execution_id."""
        with self._lock:
            trace = self._traces.get(execution_id)
            return trace.to_dict() if trace else None

    def list_traces(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recent execution summaries (newest first)."""
        with self._lock:
            ids = list(reversed(self._order))[:limit]
            summaries: list[dict[str, Any]] = []
            for eid in ids:
                t = self._traces.get(eid)
                if not t:
                    continue

                flags = t.planner.get("capability_flags", {})
                route = t.planner.get("clean_subqueries", {}).get("route")
                if not route:
                    if flags.get("needs_clarification"):
                        route = "clarification"
                    elif flags.get("needs_legal") and flags.get("needs_rate"):
                        route = "mixed"
                    elif flags.get("needs_rate"):
                        route = "rate"
                    elif flags.get("needs_legal"):
                        route = "legal"
                    elif flags.get("needs_direct_reasoning"):
                        route = "direct"
                    else:
                        route = "direct"

                summaries.append({
                    "execution_id": t.execution_id,
                    "timestamp": t.timestamp,
                    "query": t.query,
                    "status": t.status,
                    "route": route,
                    "executed_nodes": list(t.graph_execution.get("actual_executed_nodes", [])),
                    "total_timing_ms": round(t.timings.get("total_request_ms", 0.0), 2),
                    "model_used": t.synthesis.get("model_name", ""),
                    "has_legal": bool(t.legal_retrieval.get("final_chunks")),
                    "has_rate": bool(t.rate_retrieval.get("returned_candidates")),
                    "has_calculation": bool(t.calculation.get("deterministic_result")),
                })
            return summaries

    def record_planner(
        self,
        execution_id: str,
        *,
        capability_flags: dict[str, Any],
        clean_subqueries: dict[str, Any],
        extracted_user_premises: dict[str, Any],
        clarification_decision: dict[str, Any],
        timing_ms: float,
    ) -> None:
        """Record structured planner decisions."""
        with self._lock:
            trace = self._traces.get(execution_id)
            if not trace:
                return
            trace.planner = {
                "capability_flags": dict(capability_flags),
                "clean_subqueries": dict(clean_subqueries),
                "extracted_user_premises": dict(extracted_user_premises),
                "clarification_decision": dict(clarification_decision),
                "timing_ms": round(timing_ms, 3),
            }
            trace.timings["planner_ms"] = round(timing_ms, 3)

    def record_selected_nodes(self, execution_id: str, selected_nodes: list[str]) -> None:
        """Record the capability nodes selected by the planner."""
        with self._lock:
            trace = self._traces.get(execution_id)
            if not trace:
                return
            trace.graph_execution["selected_nodes"] = list(selected_nodes)

    def record_node_execution(
        self,
        execution_id: str,
        node_name: str,
        *,
        status: str = "success",
        timing_ms: float = 0.0,
    ) -> None:
        """Record completion of a single node in the LangGraph execution flow."""
        with self._lock:
            trace = self._traces.get(execution_id)
            if not trace:
                return
            if node_name not in trace.graph_execution["actual_executed_nodes"]:
                trace.graph_execution["actual_executed_nodes"].append(node_name)
            trace.graph_execution["execution_order"].append(node_name)
            trace.graph_execution["node_status"][node_name] = status
            trace.graph_execution["node_timings_ms"][node_name] = round(timing_ms, 3)

    def record_legal_retrieval(
        self,
        execution_id: str,
        *,
        retrieval_query: str,
        retrieval_data: dict[str, Any],
        timing_ms: float,
    ) -> None:
        """Record multi-stage hybrid legal retrieval results."""
        with self._lock:
            trace = self._traces.get(execution_id)
            if not trace:
                return

            timings_dict = retrieval_data.get("timings_ms", {})
            dense_ms = float(timings_dict.get("dense", 0.0))
            bm25_ms = float(timings_dict.get("bm25", 0.0))
            rrf_ms = float(timings_dict.get("rrf", 0.0))
            reranker_ms = float(timings_dict.get("reranker", 0.0))
            total_retrieval_ms = float(timings_dict.get("total", timing_ms))

            trace.legal_retrieval = {
                "retrieval_query": retrieval_query,
                "dense_candidates": list(retrieval_data.get("dense_results", [])),
                "bm25_candidates": list(retrieval_data.get("bm25_results", [])),
                "rrf_candidates": list(retrieval_data.get("hybrid_results", [])),
                "reranker_results": list(retrieval_data.get("results", [])),
                "final_chunks": list(retrieval_data.get("results", [])),
                "document_metadata": dict(retrieval_data.get("metadata", {})),
                "timing_ms": round(total_retrieval_ms, 3),
                "stages_timings_ms": {
                    "dense": round(dense_ms, 3),
                    "bm25": round(bm25_ms, 3),
                    "rrf": round(rrf_ms, 3),
                    "reranker": round(reranker_ms, 3),
                    "total": round(total_retrieval_ms, 3),
                },
            }
            trace.timings["dense_ms"] = round(dense_ms, 3)
            trace.timings["bm25_ms"] = round(bm25_ms, 3)
            trace.timings["rrf_ms"] = round(rrf_ms, 3)
            trace.timings["reranker_ms"] = round(reranker_ms, 3)

    def record_rate_retrieval(
        self,
        execution_id: str,
        *,
        lookup_query: str,
        returned_candidates: list[dict[str, Any]],
        selected_rate_used_downstream: Optional[dict[str, Any]],
        timing_ms: float,
    ) -> None:
        """Record structured rate lookup results."""
        with self._lock:
            trace = self._traces.get(execution_id)
            if not trace:
                return
            trace.rate_retrieval = {
                "lookup_query": lookup_query,
                "returned_candidates": list(returned_candidates),
                "selected_rate_used_downstream": dict(selected_rate_used_downstream) if selected_rate_used_downstream else None,
                "timing_ms": round(timing_ms, 3),
            }
            trace.timings["rate_lookup_ms"] = round(timing_ms, 3)

    def record_grounded_reasoning(
        self,
        execution_id: str,
        *,
        reasoning_output: str,
        calculation_inputs: Optional[dict[str, Any]],
        timing_ms: float,
    ) -> None:
        """Record grounded reasoning applied over retrieved legal/rate evidence."""
        with self._lock:
            trace = self._traces.get(execution_id)
            if not trace:
                return
            trace.reasoning["grounded_reasoning_output"] = reasoning_output
            trace.reasoning["calculation_inputs"] = dict(calculation_inputs) if calculation_inputs else None
            trace.reasoning["grounded_timing_ms"] = round(timing_ms, 3)
            trace.timings["grounded_reasoning_ms"] = round(timing_ms, 3)

    def record_direct_reasoning(
        self,
        execution_id: str,
        *,
        reasoning_output: str,
        calculation_inputs: Optional[dict[str, Any]],
        timing_ms: float,
    ) -> None:
        """Record retrieval-free direct reasoning."""
        with self._lock:
            trace = self._traces.get(execution_id)
            if not trace:
                return
            trace.reasoning["direct_reasoning_output"] = reasoning_output
            trace.reasoning["calculation_inputs"] = dict(calculation_inputs) if calculation_inputs else None
            trace.reasoning["direct_timing_ms"] = round(timing_ms, 3)
            trace.timings["direct_reasoning_ms"] = round(timing_ms, 3)

    def record_calculation(
        self,
        execution_id: str,
        *,
        calculation_inputs: Optional[dict[str, Any]],
        rate_used: Optional[float],
        source_of_rate: Optional[str],
        operation: Optional[str],
        deterministic_result: Optional[dict[str, Any]],
        timing_ms: float,
    ) -> None:
        """Record deterministic calculation details."""
        with self._lock:
            trace = self._traces.get(execution_id)
            if not trace:
                return
            trace.calculation = {
                "calculation_inputs": dict(calculation_inputs) if calculation_inputs else None,
                "rate_used": rate_used,
                "source_of_rate": source_of_rate,
                "operation": operation,
                "deterministic_result": dict(deterministic_result) if deterministic_result else None,
                "timing_ms": round(timing_ms, 3),
            }
            trace.timings["calculation_ms"] = round(timing_ms, 3)

    def record_synthesis(
        self,
        execution_id: str,
        *,
        sources_provided: list[dict[str, Any]],
        model_name: str,
        generation_timing_ms: float,
        final_answer: str,
    ) -> None:
        """Record synthesis inputs and generated answer."""
        with self._lock:
            trace = self._traces.get(execution_id)
            if not trace:
                return
            trace.synthesis = {
                "sources_provided": list(sources_provided),
                "model_name": model_name,
                "generation_timing_ms": round(generation_timing_ms, 3),
                "final_answer": final_answer,
            }
            trace.timings["synthesis_ms"] = round(generation_timing_ms, 3)

    def finalize(
        self,
        execution_id: str,
        *,
        total_request_ms: float,
        status: str = "success",
        error: Optional[str] = None,
    ) -> None:
        """Mark execution trace as completed and seal overall timings."""
        with self._lock:
            trace = self._traces.get(execution_id)
            if not trace:
                return
            trace.status = status
            trace.error = error
            trace.timings["total_request_ms"] = round(total_request_ms, 3)

    def clear(self) -> None:
        """Clear all stored traces (useful in testing)."""
        with self._lock:
            self._traces.clear()
            self._order.clear()


# Global singleton instance for runtime observability
trace_store = TraceStore(max_size=200)
