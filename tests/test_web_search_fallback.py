"""Tests for Fallback Web Search Retrieval in GST LangGraph Architecture."""

from unittest.mock import MagicMock, patch
import pytest

from src.graph.graph import build_gst_graph
from src.graph.routing import (
    route_capabilities,
    route_post_notification_support,
    route_post_retrieval,
    route_post_web_search,
    should_fallback_to_web_search,
)
from src.graph.state import GSTGraphState
from src.retrievers.web_search_retriever import (
    COMMON_COMMODITY_TARIFF_MAP,
    extract_candidate_hsns,
    search_web_fallback,
)


def test_should_fallback_to_web_search_conditions():
    """Verify strict conditional triggers for web search fallback."""
    # 1. Never trigger if already attempted (loop prevention)
    state_attempted: GSTGraphState = {
        "needs_rate": True,
        "rate_results": [],
        "web_search_attempted": True,
    }
    assert should_fallback_to_web_search(state_attempted) is False

    # 2. Never trigger for pure calculation or direct reasoning (no retrieval requested)
    state_calc: GSTGraphState = {
        "needs_calculation": True,
        "needs_direct_reasoning": True,
        "needs_rate": False,
        "needs_legal": False,
        "needs_notification": False,
    }
    assert should_fallback_to_web_search(state_calc) is False

    # 3. Never trigger when local rate lookup succeeds
    state_rate_found: GSTGraphState = {
        "needs_rate": True,
        "rate_results": [{"hsn_code": "0405", "total_gst_rate": "5%"}],
    }
    assert should_fallback_to_web_search(state_rate_found) is False

    # 4. Never trigger when local legal retrieval succeeds
    state_legal_found: GSTGraphState = {
        "needs_legal": True,
        "legal_results": [{"reference": "Section 29"}],
    }
    assert should_fallback_to_web_search(state_legal_found) is False

    # 5. Trigger when rate lookup reports missing evidence (0 results)
    state_rate_empty: GSTGraphState = {
        "needs_rate": True,
        "rate_results": [],
        "web_search_attempted": False,
    }
    assert should_fallback_to_web_search(state_rate_empty) is True

    # 6. Trigger when legal retrieval reports missing evidence (0 results)
    state_legal_empty: GSTGraphState = {
        "needs_legal": True,
        "legal_results": [],
        "notification_results": [],
        "web_search_attempted": False,
    }
    assert should_fallback_to_web_search(state_legal_empty) is True


def test_route_post_notification_support_routing():
    """Verify routing decisions post notification support."""
    # Local evidence missing -> routes to web_search
    state_missing: GSTGraphState = {
        "needs_rate": True,
        "rate_results": [],
        "web_search_attempted": False,
    }
    assert route_post_notification_support(state_missing) == "web_search"

    # Local evidence present -> routes directly to synthesis or reasoning
    state_present: GSTGraphState = {
        "needs_rate": True,
        "rate_results": [{"hsn_code": "8517", "total_gst_rate": "18%"}],
        "web_search_attempted": False,
    }
    assert route_post_notification_support(state_present) == "synthesis"

    # Dependent calculation -> routes to calculation
    state_calc_dep: GSTGraphState = {
        "needs_rate": True,
        "rate_results": [{"hsn_code": "8517", "total_gst_rate": "18%"}],
        "needs_calculation": True,
    }
    assert route_post_notification_support(state_calc_dep) == "calculation"


def test_route_post_web_search():
    """Verify routing after web_search fallback node finishes."""
    state: GSTGraphState = {
        "web_search_attempted": True,
        "needs_calculation": False,
    }
    assert route_post_web_search(state) == "synthesis"

    state_with_calc: GSTGraphState = {
        "web_search_attempted": True,
        "needs_calculation": True,
    }
    assert route_post_web_search(state_with_calc) == "calculation"


def test_extract_candidate_hsns():
    """Verify candidate HSN code extraction from web texts."""
    text1 = "Mobile phones fall under HSN code 8517 and attract 18% GST under CBIC notification."
    assert "8517" in extract_candidate_hsns(text1)

    text2 = "Check rate under tariff heading 8471.30.10 for laptops."
    assert any("8471" in c for c in extract_candidate_hsns(text2))

    # Calendar years like 2024, 2025, 2026 must be excluded
    text_years = "Notification issued in year 2025 and 2026 for GST rates."
    assert "2025" not in extract_candidate_hsns(text_years)
    assert "2026" not in extract_candidate_hsns(text_years)


def test_search_web_fallback_hsn_feedback_loop():
    """Verify that discovering an HSN code feeds it back into local rate retrieval."""
    # Query for mobile phones (which has 'All goods' in statutory rate table)
    res = search_web_fallback("mobile phones", search_type="rate")
    
    assert res.get("discovered_hsn") == "8517"
    assert len(res.get("rate_results", [])) > 0

    # Verify the local structured record was fetched
    r0 = res["rate_results"][0]
    assert r0.get("hsn_code") == "8517"
    assert r0.get("total_gst_rate") == "18%"
    assert r0.get("cgst_rate") == "9%"
    assert r0.get("sgst_rate") == "9%"
    assert "09/2025" in (r0.get("notification_no") or "")

    # Verify web citations are preserved
    assert len(res.get("web_results", [])) > 0
    w0 = res["web_results"][0]
    assert w0.get("document_type") == "web"
    assert w0.get("provenance") == "web_search"
    assert w0.get("url") is not None


def test_search_web_fallback_unknown_product_fails_safe():
    """Verify that unknown/fictional items return empty without guessing."""
    with patch("src.retrievers.web_search_retriever._execute_http_search", return_value=[]):
        res = search_web_fallback("totally_nonexistent_fictional_alien_mineral_xyz99", search_type="rate")
        assert res.get("rate_results") == []
        assert res.get("web_results") == []
        assert res.get("discovered_hsn") is None


def test_langgraph_fallback_web_search_execution():
    """End-to-end LangGraph test verifying web_search node is triggered on missing local evidence."""
    app_graph = build_gst_graph()
    compiled = app_graph.compile()

    # Mock planner to require rate lookup for mobile phones
    mock_plan = {
        "needs_structured_rate_lookup": True,
        "clean_subqueries": {"rate_query": "mobile phones"},
        "user_premises": {},
    }

    with patch("src.graph.nodes.plan_capabilities", return_value=mock_plan):
        # Initial rate lookup for 'mobile phones' returns empty
        with patch("src.graph.nodes.retrieve_rates", side_effect=[
            [],  # First call from rate_lookup_node: empty!
            [{"hsn_code": "8517", "description": "All goods", "total_gst_rate": "18%", "cgst_rate": "9%", "sgst_rate": "9%", "notification_no": "09/2025-Central Tax (Rate)", "serial_no": "490."}],  # Second call from web_search_node: feedback succeeded!
        ]):
            state = compiled.invoke({
                "user_query": "What is the GST rate on mobile phones?",
                "messages": [],
            })

            # Assert web search was attempted
            assert state.get("web_search_attempted") is True
            # Assert rate results were populated via feedback loop
            assert len(state.get("rate_results", [])) > 0
            assert state["rate_results"][0]["hsn_code"] == "8517"
            # Assert web citations are preserved
            assert len(state.get("web_results", [])) > 0
            assert any(s.get("document_type") == "web" for s in state.get("sources", []))
            # Assert final answer contains the verified 18% rate
            assert "18%" in state.get("final_answer", "")
