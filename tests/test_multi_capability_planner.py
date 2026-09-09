"""Stage 1: Multi-Capability Planning and Orchestration Decision Layer Tests.

Verifies that the planner:
1. Emits structured capability plans without route exclusivity.
2. Selects multiple concurrent capabilities for complex queries.
3. Does NOT calculate final answers or invent statutory data (Stage 1 boundary).
4. Handles multilingual queries (e.g. Gujarati) without relying purely on English keywords.

Tests the 7 required user queries:
1. "What is GST rate on butter?"
2. "What is the HSN code for butter?"
3. "What does Section 29 say about cancellation?"
4. "Notification 09/2025 superseded which notification?"
5. "તાજા દૂધ પર GST કેટલો છે?"
6. "Compare GST treatment of motorcycles below and above 350cc."
7. Butter + CGST/SGST/IGST ITC interstate calculation question.
"""

import os
from unittest.mock import MagicMock, patch
import pytest

from src.routers.planner import (
    GSTPlan,
    plan_capabilities,
    plan_capabilities_with_llm,
    get_default_planner_client,
    get_default_planner_model,
)


@pytest.fixture
def has_api_key():
    return bool(os.environ.get("OPENAI_API_KEY"))


# -------------------------------------------------------------------------
# Schema and Constraint Tests
# -------------------------------------------------------------------------
def test_gst_plan_schema_defaults():
    plan = GSTPlan()
    assert plan.needs_direct_reasoning is False
    assert plan.needs_calculation is False
    assert plan.needs_structured_rate_lookup is False
    assert plan.needs_hsn_lookup is False
    assert plan.needs_legal_retrieval is False
    assert plan.needs_notification_retrieval is False
    assert plan.needs_temporal_reasoning is False
    assert plan.needs_comparison is False
    assert plan.needs_exception_reasoning is False
    assert plan.needs_clarification is False
    assert plan.needs_grounded_synthesis is True
    assert isinstance(plan.clean_subqueries, dict)
    assert isinstance(plan.user_premises, dict)


def test_gst_plan_allows_concurrent_capabilities():
    """Verify that multiple capabilities can be True at the same time."""
    plan = GSTPlan(
        needs_direct_reasoning=True,
        needs_calculation=True,
        needs_structured_rate_lookup=True,
        needs_legal_retrieval=True,
        needs_grounded_synthesis=True,
    )
    dumped = plan.model_dump()
    true_caps = [k for k, v in dumped.items() if k.startswith("needs_") and v is True]
    assert len(true_caps) >= 5
    assert "needs_structured_rate_lookup" in true_caps
    assert "needs_legal_retrieval" in true_caps
    assert "needs_calculation" in true_caps


# -------------------------------------------------------------------------
# Required Test 1: "What is GST rate on butter?"
# -------------------------------------------------------------------------
def test_planner_q1_gst_rate_on_butter(has_api_key):
    query = "What is GST rate on butter?"
    plan = plan_capabilities(query, use_llm=has_api_key)

    assert plan["needs_structured_rate_lookup"] is True
    assert plan["needs_grounded_synthesis"] is True
    assert plan["needs_calculation"] is False
    rate_q = plan["clean_subqueries"]["rate_query"]
    assert rate_q is not None
    assert "butter" in rate_q.lower()


# -------------------------------------------------------------------------
# Required Test 2: "What is the HSN code for butter?"
# -------------------------------------------------------------------------
def test_planner_q2_hsn_code_for_butter(has_api_key):
    query = "What is the HSN code for butter?"
    plan = plan_capabilities(query, use_llm=has_api_key)

    assert plan["needs_hsn_lookup"] is True
    assert plan["needs_structured_rate_lookup"] is True
    assert plan["needs_grounded_synthesis"] is True
    rate_q = plan["clean_subqueries"]["rate_query"]
    assert rate_q is not None
    assert "butter" in rate_q.lower()


# -------------------------------------------------------------------------
# Required Test 3: "What does Section 29 say about cancellation?"
# -------------------------------------------------------------------------
def test_planner_q3_section_29_cancellation(has_api_key):
    query = "What does Section 29 say about cancellation?"
    plan = plan_capabilities(query, use_llm=has_api_key)

    assert plan["needs_legal_retrieval"] is True
    assert plan["needs_grounded_synthesis"] is True
    legal_q = plan["clean_subqueries"]["legal_query"]
    assert legal_q is not None
    assert "29" in legal_q or "cancellation" in legal_q.lower()


# -------------------------------------------------------------------------
# Required Test 4: "Notification 09/2025 superseded which notification?"
# -------------------------------------------------------------------------
def test_planner_q4_notification_superseded(has_api_key):
    query = "Notification 09/2025 superseded which notification?"
    plan = plan_capabilities(query, use_llm=has_api_key)

    assert plan["needs_notification_retrieval"] is True or plan["needs_temporal_reasoning"] is True
    assert plan["needs_grounded_synthesis"] is True
    notif_q = plan["clean_subqueries"].get("notification_query")
    assert notif_q is not None
    assert "09/2025" in notif_q


# -------------------------------------------------------------------------
# Required Test 5: "તાજા દૂધ પર GST કેટલો છે?" (Multilingual Gujarati)
# -------------------------------------------------------------------------
def test_planner_q5_fresh_milk_gujarati(has_api_key):
    query = "તાજા દૂધ પર GST કેટલો છે?"
    plan = plan_capabilities(query, use_llm=has_api_key)

    assert plan["needs_structured_rate_lookup"] is True
    assert plan["needs_grounded_synthesis"] is True
    rate_q = plan["clean_subqueries"]["rate_query"]
    assert rate_q is not None
    # Verify translated/isolated product for downstream retrieval tool
    assert any(term in rate_q.lower() for term in ["milk", "fresh milk", "દૂધ"])


# -------------------------------------------------------------------------
# Required Test 6: "Compare GST treatment of motorcycles below and above 350cc."
# -------------------------------------------------------------------------
def test_planner_q6_compare_motorcycles(has_api_key):
    query = "Compare GST treatment of motorcycles below and above 350cc."
    plan = plan_capabilities(query, use_llm=has_api_key)

    assert plan["needs_structured_rate_lookup"] is True
    assert plan["needs_comparison"] is True
    assert plan["needs_grounded_synthesis"] is True


# -------------------------------------------------------------------------
# Required Test 7: Butter + CGST/SGST/IGST ITC interstate calculation
# -------------------------------------------------------------------------
def test_planner_q7_butter_itc_interstate_multi_capability(has_api_key):
    query = (
        "I have CGST credit ₹2500, SGST credit ₹2500 and IGST credit ₹2500. "
        "I want to send butter interstate. What value can I send without making "
        "an additional GST cash payment?"
    )
    plan = plan_capabilities(query, use_llm=has_api_key)

    # 1. Verify MULTIPLE capabilities are selected simultaneously
    assert plan["needs_structured_rate_lookup"] is True
    assert plan["needs_legal_retrieval"] is True
    assert plan["needs_calculation"] is True
    assert plan["needs_direct_reasoning"] is True
    assert plan["needs_grounded_synthesis"] is True

    # 2. Verify premises extraction
    premises = plan["user_premises"]
    assert premises is not None
    itc = premises.get("itc_balances") or {}
    assert float(itc.get("cgst", 0)) == 2500.0
    assert float(itc.get("sgst", 0)) == 2500.0
    assert float(itc.get("igst", 0)) == 2500.0
    assert premises.get("supply_type") == "interstate"

    # 3. Verify clean subqueries for Stage 2 tools
    subq = plan["clean_subqueries"]
    assert "butter" in subq["rate_query"].lower()
    assert subq["legal_query"] is not None
    # Verify the legal subquery is neutral and does not invent unmentioned statutory sections/rules
    assert any(term in subq["legal_query"].lower() for term in ["utilization", "itc", "credit", "igst", "interstate"])
    assert "section 49" not in subq["legal_query"].lower()
    assert "rule 88a" not in subq["legal_query"].lower()

    # 4. Verify Stage 1 boundary: planner does NOT calculate or claim the final ₹1,50,000 answer
    # The planner only outputs planning decisions and facts for downstream tools.
    assert "150000" not in str(plan.get("reasoning", ""))
    assert "1,50,000" not in str(plan.get("reasoning", ""))
