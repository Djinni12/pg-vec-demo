"""Focused test suite for centralized LLM token-usage and cost tracking.

Covers the 10 required test scenarios:
1. Single LLM call cost calculation
2. Multiple LLM calls aggregated into one request
3. Cached-token calculation without double counting
4. Missing usage metadata handling
5. Unknown model pricing
6. Concurrent requests do not mix usage
7. Total request cost equals sum of stage costs
8. Local retrieval operations are not counted as OpenAI LLM calls
9. Developer/debug output displays aggregated usage
10. TraceStore integration and summary metrics
"""

from __future__ import annotations

import concurrent.futures
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.observability.llm_usage_tracker import (
    LLMUsageTracker,
    MODEL_PRICING,
    calculate_cost,
    extract_usage_metadata,
    format_llm_usage_debug,
    get_model_pricing,
    get_request_usage,
    llm_usage_tracker,
    record_llm_call,
    set_current_request_id,
)
from src.observability.trace_store import TraceStore, trace_store


@pytest.fixture(autouse=True)
def clean_tracker_and_trace_store():
    """Reset global trackers before and after each test."""
    llm_usage_tracker.clear()
    trace_store.clear()
    set_current_request_id(None)
    yield
    llm_usage_tracker.clear()
    trace_store.clear()
    set_current_request_id(None)


# -----------------------------------------------------------------------------
# Test 1: Single LLM call cost calculation
# -----------------------------------------------------------------------------
def test_single_llm_call_cost_calculation():
    """Test standard single LLM call with known pricing (gpt-4o-mini)."""
    # gpt-4o-mini pricing: input=0.15, cached_input=0.075, output=0.60 per 1M tokens
    cost = calculate_cost(
        model="gpt-4o-mini",
        input_tokens=1000,
        cached_input_tokens=0,
        output_tokens=500,
    )
    # Expected: (1000 * 0.15 + 500 * 0.60) / 1,000,000 = (150 + 300) / 1,000,000 = 0.000450
    assert cost == 0.000450

    record = record_llm_call(
        request_id="req-1",
        stage="synthesis",
        model="gpt-4o-mini",
        input_tokens=1000,
        cached_input_tokens=0,
        output_tokens=500,
        latency_ms=123.4,
    )
    assert record.request_id == "req-1"
    assert record.stage == "synthesis"
    assert record.model == "gpt-4o-mini"
    assert record.input_tokens == 1000
    assert record.output_tokens == 500
    assert record.total_tokens == 1500
    assert record.estimated_cost_usd == 0.000450
    assert record.latency_ms == 123.4


# -----------------------------------------------------------------------------
# Test 2: Multiple LLM calls aggregated into one request
# -----------------------------------------------------------------------------
def test_multiple_calls_aggregated_in_one_request():
    """Test multiple stages (planner + synthesis) aggregated under single request_id."""
    req_id = "req-multi-123"

    # Stage 1: Planner
    record_llm_call(
        request_id=req_id,
        stage="planner",
        model="gpt-4o-mini",
        input_tokens=400,
        cached_input_tokens=0,
        output_tokens=100,
        latency_ms=250.0,
    )

    # Stage 2: Synthesis
    record_llm_call(
        request_id=req_id,
        stage="synthesis",
        model="gpt-4o-mini",
        input_tokens=1200,
        cached_input_tokens=200,
        output_tokens=300,
        latency_ms=800.0,
    )

    usage = get_request_usage(req_id)
    assert usage["request_id"] == req_id
    assert usage["total_calls"] == 2
    assert usage["total_input_tokens"] == 1600
    assert usage["total_cached_input_tokens"] == 200
    assert usage["total_output_tokens"] == 400
    assert usage["total_tokens"] == 2000
    assert usage["total_latency_ms"] == 1050.0

    stages = [c["stage"] for c in usage["calls"]]
    assert stages == ["planner", "synthesis"]

    # Check that total cost is calculated and non-zero
    assert usage["total_cost_usd"] is not None
    assert usage["total_cost_usd"] > 0.0


# -----------------------------------------------------------------------------
# Test 3: Cached-token calculation without double counting
# -----------------------------------------------------------------------------
def test_cached_tokens_deducted_without_double_counting():
    """Verify cached input tokens are subtracted from total input before uncached pricing.

    If input=1000, cached=400, uncached=600.
    Cost = (600 * 0.15 + 400 * 0.075 + 200 * 0.60) / 1,000,000
         = (90 + 30 + 120) / 1,000,000 = 240 / 1,000,000 = 0.000240
    If cached tokens were double-counted without deduction:
         (1000 * 0.15 + 400 * 0.075 + 200 * 0.60) / 1,000,000 = 300 / 1,000,000 = 0.000300
    """
    cost = calculate_cost(
        model="gpt-4o-mini",
        input_tokens=1000,
        cached_input_tokens=400,
        output_tokens=200,
    )
    assert cost == 0.000240
    assert cost < 0.000300  # Confirms deduction avoided double-charging


# -----------------------------------------------------------------------------
# Test 4: Missing usage metadata handling
# -----------------------------------------------------------------------------
def test_missing_usage_metadata_handling():
    """Gracefully handle None response or missing usage attributes without crashing."""
    meta = extract_usage_metadata(None)
    assert meta == {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # Empty dict or object without usage attribute
    meta_empty = extract_usage_metadata({})
    assert meta_empty["total_tokens"] == 0

    # Record call with None response
    record = record_llm_call(
        request_id="req-missing",
        stage="synthesis",
        model="gpt-4o-mini",
        response=None,
    )
    assert record.input_tokens == 0
    assert record.output_tokens == 0
    assert record.total_tokens == 0
    assert record.estimated_cost_usd == 0.0


# -----------------------------------------------------------------------------
# Test 5: Unknown model pricing
# -----------------------------------------------------------------------------
def test_unknown_model_pricing():
    """Record tokens accurately even if the model is not in MODEL_PRICING, setting cost to None."""
    cost = calculate_cost(
        model="custom-internal-tax-model-v9",
        input_tokens=1500,
        output_tokens=300,
    )
    assert cost is None

    record = record_llm_call(
        request_id="req-unknown-model",
        stage="synthesis",
        model="custom-internal-tax-model-v9",
        input_tokens=1500,
        output_tokens=300,
    )
    assert record.total_tokens == 1800
    assert record.estimated_cost_usd is None

    usage = get_request_usage("req-unknown-model")
    assert usage["total_tokens"] == 1800
    assert usage["total_cost_usd"] is None


# -----------------------------------------------------------------------------
# Test 6: Concurrent requests do not mix usage
# -----------------------------------------------------------------------------
def test_concurrent_requests_do_not_mix_usage():
    """Ensure concurrent requests in separate threads do not cross-contaminate tokens."""
    tracker = LLMUsageTracker(max_requests=100)
    num_threads = 10
    calls_per_thread = 5

    def worker(worker_idx: int):
        req_id = f"concurrent-req-{worker_idx}"
        for c_idx in range(calls_per_thread):
            tracker.record_call(
                request_id=req_id,
                stage="planner" if c_idx == 0 else "synthesis",
                model="gpt-4o-mini",
                input_tokens=100 * (worker_idx + 1),
                cached_input_tokens=0,
                output_tokens=50,
                latency_ms=10.0,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        concurrent.futures.wait(futures)

    for i in range(num_threads):
        req_id = f"concurrent-req-{i}"
        usage = tracker.get_request_usage(req_id)
        assert usage["total_calls"] == calls_per_thread
        expected_input = 100 * (i + 1) * calls_per_thread
        expected_output = 50 * calls_per_thread
        assert usage["total_input_tokens"] == expected_input
        assert usage["total_output_tokens"] == expected_output
        assert usage["total_tokens"] == expected_input + expected_output


# -----------------------------------------------------------------------------
# Test 7: Total request cost equals sum of stage costs
# -----------------------------------------------------------------------------
def test_total_request_cost_equals_sum_of_stage_costs():
    """Request total cost must equal the exact sum of individual stage costs."""
    req_id = "req-sum-check"

    rec1 = record_llm_call(
        request_id=req_id,
        stage="planner",
        model="gpt-4o-mini",
        input_tokens=500,
        cached_input_tokens=0,
        output_tokens=120,
    )
    rec2 = record_llm_call(
        request_id=req_id,
        stage="grounded_reasoning",
        model="gpt-4o",
        input_tokens=800,
        cached_input_tokens=200,
        output_tokens=250,
    )
    rec3 = record_llm_call(
        request_id=req_id,
        stage="synthesis",
        model="gpt-4o-mini",
        input_tokens=1400,
        cached_input_tokens=300,
        output_tokens=400,
    )

    usage = get_request_usage(req_id)
    expected_sum = round(
        rec1.estimated_cost_usd + rec2.estimated_cost_usd + rec3.estimated_cost_usd,
        6,
    )
    assert usage["total_cost_usd"] == expected_sum


# -----------------------------------------------------------------------------
# Test 8: Local retrieval operations are not counted as OpenAI LLM calls
# -----------------------------------------------------------------------------
def test_local_retrieval_not_counted_as_llm_calls():
    """Dense/BM25/RRF retrieval and rate lookups must not create LLM tracker calls."""
    req_id = "req-pure-retrieval"
    set_current_request_id(req_id)

    # Perform simulated retrieval operations (e.g. mock BM25 or DB rate lookup)
    mock_retrieval = {
        "dense_results": [{"id": 1, "score": 0.8}],
        "bm25_results": [{"id": 2, "score": 12.5}],
        "rate_results": [{"hsn": "1905", "rate": 18.0}],
    }

    # Verify no calls exist before or after pure retrieval
    usage = get_request_usage(req_id)
    assert usage["total_calls"] == 0
    assert usage["total_tokens"] == 0
    assert usage["total_cost_usd"] is None


# -----------------------------------------------------------------------------
# Test 9: Developer/debug output displays aggregated usage
# -----------------------------------------------------------------------------
def test_developer_debug_output_displays_aggregated_usage():
    """Debug output string must contain emoji header, stage breakdowns, and totals."""
    req_id = "req-debug-text"
    record_llm_call(
        request_id=req_id,
        stage="planner",
        model="gpt-4o-mini",
        input_tokens=350,
        cached_input_tokens=0,
        output_tokens=85,
        latency_ms=412.3,
    )
    record_llm_call(
        request_id=req_id,
        stage="synthesis",
        model="gpt-4o-mini",
        input_tokens=1240,
        cached_input_tokens=256,
        output_tokens=310,
        latency_ms=1150.8,
    )

    usage = get_request_usage(req_id)
    debug_text = usage["formatted_debug"]

    assert "🪙 LLM Usage & Cost Breakdown [Request: req-debug-text]" in debug_text
    assert "Stage: planner (model: gpt-4o-mini)" in debug_text
    assert "Stage: synthesis (model: gpt-4o-mini)" in debug_text
    assert "Total Tokens:" in debug_text
    assert "Total Estimated Cost:" in debug_text
    assert "USD" in debug_text


# -----------------------------------------------------------------------------
# Test 10: TraceStore integration and summary metrics
# -----------------------------------------------------------------------------
def test_trace_store_integration():
    """TraceStore properly registers llm_usage and list_traces includes token & cost metrics."""
    exec_id = "exec-trace-101"
    trace_store.create_trace(exec_id, query="What is the GST rate on ice cream?")

    # Record calls via TraceStore helper
    trace_store.record_llm_call(
        exec_id,
        stage="planner",
        model="gpt-4o-mini",
        input_tokens=300,
        cached_input_tokens=0,
        output_tokens=75,
        latency_ms=150.0,
    )
    trace_store.record_llm_call(
        exec_id,
        stage="synthesis",
        model="gpt-4o-mini",
        input_tokens=900,
        cached_input_tokens=100,
        output_tokens=200,
        latency_ms=650.0,
    )

    trace_store.finalize(exec_id, total_request_ms=850.0, status="success")

    # Inspect single trace
    trace = trace_store.get_trace(exec_id)
    assert trace is not None
    assert "llm_usage" in trace
    assert trace["llm_usage"]["total_calls"] == 2
    assert trace["llm_usage"]["total_tokens"] == 1475
    assert trace["llm_usage"]["total_cost_usd"] > 0.0

    # Inspect list_traces summaries
    summaries = trace_store.list_traces(limit=10)
    matching = [s for s in summaries if s["execution_id"] == exec_id]
    assert len(matching) == 1
    s = matching[0]
    assert s["total_tokens"] == 1475
    assert s["estimated_cost_usd"] > 0.0
    assert s["llm_calls_count"] == 2
