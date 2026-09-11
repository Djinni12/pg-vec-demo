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

import json
import os
from unittest.mock import MagicMock, patch
import pytest

from src.routers.planner import (
    GSTPlan,
    RetrievalSubquery,
    normalize_query_decomposition,
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
    assert plan.needs_query_decomposition is False
    assert plan.retrieval_subqueries == []
    assert isinstance(plan.clean_subqueries, dict)
    assert isinstance(plan.user_premises, dict)


def test_retrieval_subquery_model():
    """Verify RetrievalSubquery validation and normalization."""
    sq1 = RetrievalSubquery(type="legal", query="conditions for post-supply discount")
    assert sq1.type == "legal"
    assert sq1.query == "conditions for post-supply discount"

    # Type normalization
    sq2 = RetrievalSubquery(type="RATE", query="GST rate on butter")
    assert sq2.type == "rate"

    # Pydantic serialization
    plan = GSTPlan(
        needs_query_decomposition=True,
        retrieval_subqueries=[sq1, sq2],
    )
    dumped = plan.model_dump()
    assert dumped["needs_query_decomposition"] is True
    assert len(dumped["retrieval_subqueries"]) == 2
    assert dumped["retrieval_subqueries"][0]["type"] == "legal"
    assert dumped["retrieval_subqueries"][1]["type"] == "rate"


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


# -------------------------------------------------------------------------
# Generic Normalization and Validation Unit Tests (Python Layer)
# -------------------------------------------------------------------------
def test_generic_normalization_allow_0_to_4_subqueries():
    """Python must allow 0 to 4 subqueries and cap >4 subqueries to 4."""
    # 0 subqueries
    d0, q0 = normalize_query_decomposition(True, [])
    assert d0 is False
    assert q0 == []

    # 1 subquery (< 2 distinct needs -> False)
    d1, q1 = normalize_query_decomposition(True, [{"type": "legal", "query": "one need"}])
    assert d1 is False
    assert q1 == []

    # 2 subqueries
    d2, q2 = normalize_query_decomposition(True, [
        {"type": "legal", "query": "concept one"},
        {"type": "legal", "query": "concept two"}
    ])
    assert d2 is True
    assert len(q2) == 2

    # 5 subqueries -> capped at 4
    d5, q5 = normalize_query_decomposition(True, [
        {"type": "legal", "query": f"concept {i}"} for i in range(5)
    ])
    assert d5 is True
    assert len(q5) == 4


def test_generic_normalization_allowed_types():
    """Allowed types are 'legal' and 'rate' with case-insensitive normalization."""
    decomp, subqs = normalize_query_decomposition(True, [
        {"type": "RATE", "query": "GST rate on butter"},
        {"type": "legal_statute", "query": "credit note requirements"}
    ])
    assert decomp is True
    assert subqs[0]["type"] == "rate"
    assert subqs[1]["type"] == "legal"


def test_generic_normalization_remove_duplicates():
    """Duplicate queries must be removed; if deduplication reduces count < 2, decomp is False."""
    # 3 subqueries with 1 duplicate -> 2 distinct
    decomp, subqs = normalize_query_decomposition(True, [
        {"type": "legal", "query": "conditions for discount"},
        {"type": "legal", "query": "conditions for discount"},
        {"type": "legal", "query": "procedure for adjustment"}
    ])
    assert decomp is True
    assert len(subqs) == 2

    # All duplicate -> reduces to 1 (<2 distinct) -> decomp is False
    d_dup, q_dup = normalize_query_decomposition(True, [
        {"type": "legal", "query": "conditions for discount"},
        {"type": "legal", "query": "conditions for discount"}
    ])
    assert d_dup is False
    assert q_dup == []


def test_generic_normalization_reject_empty_queries():
    """Empty or whitespace-only queries must be rejected."""
    decomp, subqs = normalize_query_decomposition(True, [
        {"type": "legal", "query": "   "},
        {"type": "legal", "query": "valid query one"},
        {"type": "rate", "query": ""},
        {"type": "rate", "query": "valid query two"}
    ])
    assert decomp is True
    assert len(subqs) == 2
    assert subqs[0]["query"] == "valid query one"
    assert subqs[1]["query"] == "valid query two"


def test_generic_normalization_fewer_than_2_distinct_evidence_needs():
    """If fewer than 2 distinct retrieval evidence needs exist, needs_query_decomposition remains False."""
    assert normalize_query_decomposition(True, None) == (False, [])
    assert normalize_query_decomposition(True, []) == (False, [])
    assert normalize_query_decomposition(False, [{"type": "legal", "query": "q1"}, {"type": "legal", "query": "q2"}]) == (False, [])


def test_generic_normalization_no_legal_concept_injection():
    """Python normalization must NOT inject sections, rules, or unprovided legal concepts."""
    raw_subqueries = [
        {"type": "legal", "query": "user concept alpha"},
        {"type": "legal", "query": "user concept beta"}
    ]
    _, cleaned = normalize_query_decomposition(True, raw_subqueries)
    for sq in cleaned:
        assert "section" not in sq["query"].lower()
        assert "rule" not in sq["query"].lower()
        assert "notification" not in sq["query"].lower()


def test_gst_plan_model_validator_integration():
    """Verify GSTPlan Pydantic model automatically applies generic normalization upon instantiation."""
    # Instantiating with duplicates reduces to 0 (<2 distinct) and sets needs_query_decomposition=False
    plan1 = GSTPlan(
        needs_query_decomposition=True,
        retrieval_subqueries=[
            {"type": "legal", "query": "duplicate query"},
            {"type": "legal", "query": "duplicate query"}
        ]
    )
    assert plan1.needs_query_decomposition is False
    assert plan1.retrieval_subqueries == []

    # Instantiating with 5 subqueries caps at 4
    plan2 = GSTPlan(
        needs_query_decomposition=True,
        retrieval_subqueries=[{"type": "rate", "query": f"rate {i}"} for i in range(5)]
    )
    assert plan2.needs_query_decomposition is True
    assert len(plan2.retrieval_subqueries) == 4


# -------------------------------------------------------------------------
# Helper for Simulating Planner Model Outputs
# -------------------------------------------------------------------------
def make_mock_client(payload: dict) -> MagicMock:
    client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(message=MagicMock(content=json.dumps(payload)))
    ]
    client.chat.completions.create.return_value = mock_resp
    return client


# -------------------------------------------------------------------------
# Model-Driven Planner Tests for the 5 Target Scenarios
# -------------------------------------------------------------------------
def test_model_driven_q1_rule_88a_no_decomposition():
    """Test 1: 'What does Rule 88A say?' -> single-concept statutory query, no decomposition."""
    query = "What does Rule 88A say?"
    mock_client = make_mock_client({
        "needs_direct_reasoning": False,
        "needs_calculation": False,
        "needs_structured_rate_lookup": False,
        "needs_hsn_lookup": False,
        "needs_legal_retrieval": True,
        "needs_notification_retrieval": False,
        "needs_temporal_reasoning": False,
        "needs_comparison": False,
        "needs_exception_reasoning": False,
        "needs_clarification": False,
        "needs_grounded_synthesis": True,
        "needs_query_decomposition": False,
        "retrieval_subqueries": [],
        "clean_subqueries": {
            "rate_query": None,
            "legal_query": "What does Rule 88A say?",
            "notification_query": None
        },
        "user_premises": {},
        "reasoning": "Single provision inquiry regarding Rule 88A."
    })
    plan = plan_capabilities(query, client=mock_client, use_llm=True)

    assert plan["needs_query_decomposition"] is False
    assert plan["retrieval_subqueries"] == []
    assert plan["needs_legal_retrieval"] is True


def test_model_driven_q2_butter_rate_no_decomposition():
    """Test 2: 'What is the GST rate on butter?' -> single-concept rate query, no decomposition."""
    query = "What is the GST rate on butter?"
    mock_client = make_mock_client({
        "needs_direct_reasoning": False,
        "needs_calculation": False,
        "needs_structured_rate_lookup": True,
        "needs_hsn_lookup": False,
        "needs_legal_retrieval": False,
        "needs_notification_retrieval": False,
        "needs_temporal_reasoning": False,
        "needs_comparison": False,
        "needs_exception_reasoning": False,
        "needs_clarification": False,
        "needs_grounded_synthesis": True,
        "needs_query_decomposition": False,
        "retrieval_subqueries": [],
        "clean_subqueries": {
            "rate_query": "butter",
            "legal_query": None,
            "notification_query": None
        },
        "user_premises": {},
        "reasoning": "Single commodity rate inquiry on butter."
    })
    plan = plan_capabilities(query, client=mock_client, use_llm=True)

    assert plan["needs_query_decomposition"] is False
    assert plan["retrieval_subqueries"] == []
    assert plan["needs_structured_rate_lookup"] is True


def test_model_driven_q3_post_supply_discount_two_legal_subqueries():
    """Test 3: 'I gave a discount after invoicing. Can I reduce GST liability and how do I adjust it?'
    -> decomposition=true driven by model with two distinct neutral legal subqueries, no injected unmentioned sections.
    """
    query = "I gave a discount after invoicing. Can I reduce GST liability and how do I adjust it?"
    mock_client = make_mock_client({
        "needs_direct_reasoning": True,
        "needs_calculation": False,
        "needs_structured_rate_lookup": False,
        "needs_hsn_lookup": False,
        "needs_legal_retrieval": True,
        "needs_notification_retrieval": False,
        "needs_temporal_reasoning": False,
        "needs_comparison": False,
        "needs_exception_reasoning": False,
        "needs_clarification": False,
        "needs_grounded_synthesis": True,
        "needs_query_decomposition": True,
        "retrieval_subqueries": [
            {
                "type": "legal",
                "query": "conditions for a post-supply discount to reduce taxable value"
            },
            {
                "type": "legal",
                "query": "procedure for adjusting tax liability after a post-supply discount"
            }
        ],
        "clean_subqueries": {
            "rate_query": None,
            "legal_query": "conditions for a post-supply discount to reduce taxable value and procedure for adjusting tax liability",
            "notification_query": None
        },
        "user_premises": {},
        "reasoning": "Query requires two independent legal evidence needs: conditions to reduce taxable value and adjustment procedure."
    })
    plan = plan_capabilities(query, client=mock_client, use_llm=True)

    assert plan["needs_query_decomposition"] is True
    assert plan["needs_legal_retrieval"] is True
    subqueries = plan["retrieval_subqueries"]
    assert len(subqueries) == 2

    for sq in subqueries:
        assert sq["type"] == "legal"
        # Must NOT inject unmentioned sections (e.g. Section 15(3)(b), Section 34, Rule 53)
        assert "section 15" not in sq["query"].lower()
        assert "section 34" not in sq["query"].lower()
        assert "rule 53" not in sq["query"].lower()

    queries_text = [sq["query"].lower() for sq in subqueries]
    assert queries_text[0] != queries_text[1]
    assert any("condition" in q or "reduce" in q for q in queries_text)
    assert any("procedure" in q or "adjust" in q for q in queries_text)


def test_model_driven_q4_tobacco_historical_and_current_rate_subqueries():
    """Test 4: 'What GST applied to tobacco before September 2025 and what applies now?'
    -> decomposition=true driven by model with separate historical and current evidence needs without inventing references.
    """
    query = "What GST applied to tobacco before September 2025 and what applies now?"
    mock_client = make_mock_client({
        "needs_direct_reasoning": True,
        "needs_calculation": False,
        "needs_structured_rate_lookup": True,
        "needs_hsn_lookup": False,
        "needs_legal_retrieval": False,
        "needs_notification_retrieval": False,
        "needs_temporal_reasoning": True,
        "needs_comparison": True,
        "needs_exception_reasoning": False,
        "needs_clarification": False,
        "needs_grounded_synthesis": True,
        "needs_query_decomposition": True,
        "retrieval_subqueries": [
            {
                "type": "rate",
                "query": "GST rate on tobacco before September 2025"
            },
            {
                "type": "rate",
                "query": "current GST rate on tobacco"
            }
        ],
        "clean_subqueries": {
            "rate_query": "tobacco",
            "legal_query": None,
            "notification_query": None
        },
        "user_premises": {},
        "reasoning": "Temporal comparison across historical and current periods on tobacco rates."
    })
    plan = plan_capabilities(query, client=mock_client, use_llm=True)

    assert plan["needs_query_decomposition"] is True
    subqueries = plan["retrieval_subqueries"]
    assert len(subqueries) >= 2

    queries_text = [sq["query"].lower() for sq in subqueries]
    assert any("before" in q for q in queries_text)
    assert any("now" in q or "current" in q for q in queries_text)
    assert all("tobacco" in q for q in queries_text)
    for q in queries_text:
        assert "section" not in q
        assert "notification" not in q


def test_model_driven_q5_hindi_calculation_no_decomposition():
    """Test 5: '₹50,000 par 18% GST kitna hai?' -> calculation query, no decomposition, no retrieval subqueries."""
    query = "₹50,000 par 18% GST kitna hai?"
    mock_client = make_mock_client({
        "needs_direct_reasoning": True,
        "needs_calculation": True,
        "needs_structured_rate_lookup": False,
        "needs_hsn_lookup": False,
        "needs_legal_retrieval": False,
        "needs_notification_retrieval": False,
        "needs_temporal_reasoning": False,
        "needs_comparison": False,
        "needs_exception_reasoning": False,
        "needs_clarification": False,
        "needs_grounded_synthesis": True,
        "needs_query_decomposition": False,
        "retrieval_subqueries": [],
        "clean_subqueries": {
            "rate_query": None,
            "legal_query": None,
            "notification_query": None
        },
        "user_premises": {
            "assumed_rate": 18.0,
            "rate_is_user_assumed": True,
            "taxable_amount": 50000.0,
            "discount_pct": None,
            "itc_balances": None,
            "supply_type": None
        },
        "reasoning": "Pure computation of 18% GST on ₹50,000."
    })
    plan = plan_capabilities(query, client=mock_client, use_llm=True)

    assert plan["needs_query_decomposition"] is False
    assert plan["retrieval_subqueries"] == []
    assert plan["needs_calculation"] is True
    assert plan["clean_subqueries"]["rate_query"] is None
    assert plan["clean_subqueries"]["legal_query"] is None


# -------------------------------------------------------------------------
# Heuristic Fallback Verification
# -------------------------------------------------------------------------
def test_heuristic_planner_no_scenario_specific_decomposition():
    """Verify that heuristic fallback does NOT contain scenario-specific decomposition regexes,
    and only performs deterministic factual premise extraction.
    """
    from src.routers.query_router import plan_capabilities_heuristic

    test_queries = [
        "What does Rule 88A say?",
        "What is the GST rate on butter?",
        "I gave a discount after invoicing. Can I reduce GST liability and how do I adjust it?",
        "What GST applied to tobacco before September 2025 and what applies now?",
        "₹50,000 par 18% GST kitna hai?",
    ]
    for q in test_queries:
        plan = plan_capabilities_heuristic(q)
        # Heuristic planner never performs scenario-specific decomposition
        assert plan["needs_query_decomposition"] is False
        assert plan["retrieval_subqueries"] == []

    # Deterministic factual premise extraction remains functional
    calc_plan = plan_capabilities_heuristic("₹50,000 par 18% GST kitna hai?")
    assert calc_plan["needs_calculation"] is True
    assert calc_plan["user_premises"]["taxable_amount"] == 50000.0

    butter_plan = plan_capabilities_heuristic("What is the GST rate on butter?")
    assert butter_plan["needs_structured_rate_lookup"] is True
    assert butter_plan["clean_subqueries"]["rate_query"] == "butter"


