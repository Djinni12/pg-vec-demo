"""Unit and integration tests for developer observability and execution tracing."""

import threading
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app import app
from src.observability.trace_store import ExecutionTrace, TraceStore, trace_store


@pytest.fixture(autouse=True)
def clean_trace_store():
    """Ensure trace store is clean before and after each test."""
    trace_store.clear()
    yield
    trace_store.clear()


@pytest.fixture
def test_client():
    app.state.models = MagicMock()
    return TestClient(app)


# -----------------------------------------------------------------------------
# 1. TraceStore Unit Tests
# -----------------------------------------------------------------------------
def test_trace_store_create_and_get():
    store = TraceStore(max_size=5)
    t = store.create_trace("exec-123", "What is GST on paneer?")

    assert t.execution_id == "exec-123"
    assert t.query == "What is GST on paneer?"
    assert t.status == "running"

    retrieved = store.get_trace("exec-123")
    assert retrieved is not None
    assert retrieved["execution_id"] == "exec-123"
    assert retrieved["query"] == "What is GST on paneer?"


def test_trace_store_bounded_ring_buffer():
    store = TraceStore(max_size=3)
    for i in range(5):
        store.create_trace(f"exec-{i}", f"Query {i}")

    # Should only retain the 3 most recent
    assert len(store._order) == 3
    assert store.get_trace("exec-0") is None
    assert store.get_trace("exec-1") is None
    assert store.get_trace("exec-2") is not None
    assert store.get_trace("exec-3") is not None
    assert store.get_trace("exec-4") is not None


def test_trace_store_records_all_trace_dimensions():
    store = TraceStore(max_size=10)
    eid = "test-full-trace-1"
    store.create_trace(eid, "Calculate GST on ₹10,000 for butter")

    # 1. Planner
    store.record_planner(
        eid,
        capability_flags={
            "needs_legal": False,
            "needs_rate": True,
            "needs_direct_reasoning": True,
            "needs_calculation": True,
            "needs_clarification": False,
        },
        clean_subqueries={"rate_query": "butter", "legal_query": None, "route": "mixed"},
        extracted_user_premises={"taxable_amount": 10000.0},
        clarification_decision={"needs_clarification": False, "clarification_prompt": None},
        timing_ms=3.5,
    )
    store.record_selected_nodes(eid, ["rate_lookup", "direct_reasoning", "synthesis"])
    store.record_node_execution(eid, "planner", status="success", timing_ms=3.5)

    # 2. Rate Lookup
    store.record_rate_retrieval(
        eid,
        lookup_query="butter",
        returned_candidates=[{
            "code": "0405",
            "description": "Butter and other fats and oils derived from milk",
            "igst_rate_pct": 5.0,
            "cgst_rate_pct": 2.5,
            "sgst_rate_pct": 2.5,
        }],
        selected_rate_used_downstream={"code": "0405", "rate": 5.0},
        timing_ms=12.0,
    )
    store.record_node_execution(eid, "rate_lookup", status="success", timing_ms=12.0)

    # 3. Calculation
    store.record_calculation(
        eid,
        calculation_inputs={"taxable_value": 10000.0, "tax_rate_pct": 5.0},
        rate_used=5.0,
        source_of_rate="Tariff HSN 0405 (Butter) - 5.0%",
        operation="tax_on_value",
        deterministic_result={"result_value": 500.0, "steps": ["Tax = ₹10,000.00 × 5.0% = ₹500.00"]},
        timing_ms=1.2,
    )
    store.record_node_execution(eid, "direct_reasoning", status="success", timing_ms=1.2)

    # 4. Synthesis
    store.record_synthesis(
        eid,
        sources_provided=[{"title": "Tariff Item 0405", "source_type": "tariff"}],
        model_name="mock-gpt",
        generation_timing_ms=45.0,
        final_answer="The GST on ₹10,000 butter at 5% is ₹500.",
    )
    store.record_node_execution(eid, "synthesis", status="success", timing_ms=45.0)

    # 5. Finalize
    store.finalize(eid, total_request_ms=62.0, status="success")

    trace = store.get_trace(eid)
    assert trace["status"] == "success"
    assert trace["planner"]["capability_flags"]["needs_rate"] is True
    assert trace["planner"]["timing_ms"] == 3.5
    assert trace["graph_execution"]["selected_nodes"] == ["rate_lookup", "direct_reasoning", "synthesis"]
    assert "rate_lookup" in trace["graph_execution"]["actual_executed_nodes"]
    assert trace["rate_retrieval"]["selected_rate_used_downstream"]["code"] == "0405"
    assert trace["calculation"]["rate_used"] == 5.0
    assert trace["calculation"]["source_of_rate"] == "Tariff HSN 0405 (Butter) - 5.0%"
    assert trace["calculation"]["deterministic_result"]["result_value"] == 500.0
    assert trace["synthesis"]["final_answer"] == "The GST on ₹10,000 butter at 5% is ₹500."
    assert trace["timings"]["total_request_ms"] == 62.0

    # Ensure no hidden chain-of-thought is exposed
    assert "chain_of_thought" not in trace
    assert "thought" not in trace["planner"]
    assert "thinking" not in trace["synthesis"]


def test_trace_store_concurrent_thread_safety():
    store = TraceStore(max_size=100)
    errors = []

    def worker(worker_id: int):
        try:
            for i in range(10):
                eid = f"worker-{worker_id}-{i}"
                store.create_trace(eid, f"Query {i}")
                store.record_node_execution(eid, "planner", status="success", timing_ms=1.0)
                store.finalize(eid, total_request_ms=5.0)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    summaries = store.list_traces(limit=100)
    assert len(summaries) == 50


# -----------------------------------------------------------------------------
# 2. FastAPI Debug Endpoints Tests
# -----------------------------------------------------------------------------
def test_debug_html_endpoint(test_client):
    response = test_client.get("/debug")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "GST Bot — Developer Observability" in response.text
    assert "debug.jsx" in response.text


def test_debug_executions_endpoint_empty_and_populated(test_client):
    # Initially empty
    res = test_client.get("/debug/executions")
    assert res.status_code == 200
    assert res.json() == []

    # Populate one trace
    trace_store.create_trace("exec-abc", "Test query")
    trace_store.record_node_execution("exec-abc", "planner", status="success", timing_ms=2.0)
    trace_store.finalize("exec-abc", total_request_ms=10.0, status="success")

    res = test_client.get("/debug/executions")
    assert res.status_code == 200
    items = res.json()
    assert len(items) == 1
    assert items[0]["execution_id"] == "exec-abc"
    assert items[0]["query"] == "Test query"
    assert items[0]["status"] == "success"


def test_debug_execution_detail_endpoint(test_client):
    trace_store.create_trace("exec-xyz", "What is Section 16?")
    trace_store.record_legal_retrieval(
        "exec-xyz",
        retrieval_query="Section 16",
        retrieval_data={
            "results": [{"doc_reference": "Section 16", "content": "Eligibility for ITC"}],
            "timings_ms": {"dense": 5.0, "bm25": 2.0, "rrf": 1.0, "reranker": 8.0, "total": 16.0},
        },
        timing_ms=16.0,
    )
    trace_store.finalize("exec-xyz", total_request_ms=20.0, status="success")

    res = test_client.get("/debug/executions/exec-xyz")
    assert res.status_code == 200
    data = res.json()
    assert data["execution_id"] == "exec-xyz"
    assert len(data["legal_retrieval"]["final_chunks"]) == 1
    assert data["legal_retrieval"]["final_chunks"][0]["doc_reference"] == "Section 16"

    # Unknown ID returns 404
    res_404 = test_client.get("/debug/executions/non-existent-id")
    assert res_404.status_code == 404


def test_debug_clear_endpoint(test_client):
    trace_store.create_trace("exec-to-clear", "Test")
    assert len(trace_store.list_traces()) == 1

    res = test_client.post("/debug/clear")
    assert res.status_code == 200
    assert res.json() == {"status": "cleared"}
    assert len(trace_store.list_traces()) == 0


# -----------------------------------------------------------------------------
# 3. Live LangGraph Chat Request Observability Integration
# -----------------------------------------------------------------------------
def test_chat_generates_trace_without_rerun(test_client):
    """Verifies /chat executes once, creates a trace, and /debug inspects it."""
    mock_rate = {
        "code": "0406",
        "description": "Chena or paneer",
        "total_gst_rate": "0%",
        "rate_category": "EXEMPTION",
    }

    with patch("src.graph.nodes.retrieve_rates", return_value=[mock_rate]):
        response = test_client.post(
            "/chat",
            json={"query": "What is the GST rate for paneer?"},
        )

    assert response.status_code == 200
    data = response.json()
    execution_id = data.get("execution_id")
    assert execution_id is not None
    assert data["answer"] != ""

    # Check that debug endpoint retrieves the exact same trace
    debug_res = test_client.get(f"/debug/executions/{execution_id}")
    assert debug_res.status_code == 200
    trace = debug_res.json()
    assert trace["execution_id"] == execution_id
    assert trace["query"] == "What is the GST rate for paneer?"
    assert trace["status"] == "success"
    assert "rate_lookup" in trace["graph_execution"]["actual_executed_nodes"]
    assert "synthesis" in trace["graph_execution"]["actual_executed_nodes"]
    assert len(trace["rate_retrieval"]["returned_candidates"]) == 1
    assert trace["rate_retrieval"]["returned_candidates"][0]["code"] == "0406"
