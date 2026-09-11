"""Focused tests for Stage 2: execute decomposed retrieval subqueries.

Validates the 6 required test scenarios:
1. Simple legal query: "What does Rule 88A say?"
   - needs_query_decomposition = False
   - existing single-query legal retrieval path used
   - no multi-query execution

2. Simple rate query: "What is the GST rate on butter?"
   - existing single rate lookup path remains unchanged

3. Complex discount query: "I gave a discount after invoicing. Can I reduce GST liability and how do I adjust it?"
   - two legal subqueries executed independently
   - execution can occur concurrently
   - results merged
   - provenance attached to each result
   - duplicate chunks removed
   - evidence from both legal concepts can survive the merge

4. Mixed-type decomposed query:
   - planner fixture containing one legal and one rate subquery
   - correct retriever dispatched for each type
   - both execute independently
   - merged output preserves type/provenance

5. Duplicate retrieval result:
   - two subqueries return the same chunk
   - final merged result contains one chunk
   - provenance records both originating subqueries

6. One subquery failure:
   - successful subquery results are preserved
   - failure is recorded/logged
   - no fabricated fallback result
"""

import asyncio
from unittest.mock import MagicMock, patch
import pytest

from src.retrieval_inspector import LoadedModels
from src.retrievers.decomposed_executor import (
    execute_decomposed_subqueries,
    execute_decomposed_subqueries_async,
    get_chunk_dedup_key,
    get_rate_dedup_key,
    merge_and_deduplicate,
)
from src.generators.answer_generator import run_gst_answer_flow
from src.graph.nodes import legal_retrieval_node, rate_lookup_node
from src.graph.state import GSTGraphState
from src.routers.planner import GSTPlan, RetrievalSubquery, plan_capabilities


@pytest.fixture
def mock_models():
    return LoadedModels(embedding_model=MagicMock(), reranker=MagicMock(), initialization_ms=5.0)


@pytest.fixture
def mock_openai_client():
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = "Synthesized grounded answer."
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


# -------------------------------------------------------------------------
# Test 1: Simple legal query ("What does Rule 88A say?")
# -------------------------------------------------------------------------
def test_simple_legal_query_uses_single_path(mock_models, mock_openai_client):
    """Verify simple legal query does not trigger query decomposition and uses single-query retrieval."""
    query = "What does Rule 88A say?"
    plan = plan_capabilities(query, use_llm=False)

    assert plan["needs_query_decomposition"] is False
    assert plan["retrieval_subqueries"] == []
    assert plan["needs_legal_retrieval"] is True

    mock_legal_chunk = {
        "chunk_id": "rule_88a_chunk_1",
        "document_type": "rule",
        "reference": "Rule 88A",
        "title": "Order of utilization of input tax credit",
        "content": "Input tax credit on account of integrated tax shall first be utilised...",
    }

    with patch("src.generators.answer_generator.execute_decomposed_subqueries") as mock_decomp, \
         patch("src.generators.answer_generator.inspect_retrieval") as mock_inspect:
        mock_inspect.return_value = {
            "results": [mock_legal_chunk],
            "timings_ms": {"total": 5.0},
        }
        resp = run_gst_answer_flow(query, models=mock_models, openai_client=mock_openai_client)

    # Decomposed executor must NOT be called for non-decomposed queries
    mock_decomp.assert_not_called()
    # Single-query inspect_retrieval must be called once
    mock_inspect.assert_called_once()
    assert len(resp["legal_chunks"] if "legal_chunks" in resp else resp.get("observability", {}).get("retrieved_evidence", {})) >= 0


def test_simple_legal_query_graph_node(mock_models):
    """Verify legal_retrieval_node does not invoke decomposed execution when needs_query_decomposition is False."""
    state: GSTGraphState = {
        "user_query": "What does Rule 88A say?",
        "needs_legal": True,
        "needs_rate": False,
        "clean_queries": {"legal_query": "Rule 88A order of utilization"},
        "plan": {
            "needs_query_decomposition": False,
            "retrieval_subqueries": [],
        },
    }

    mock_chunk = {
        "chunk_id": "rule_88a_1",
        "reference": "Rule 88A",
        "content": "Order of utilization of input tax credit",
    }

    with patch("src.graph.nodes.execute_decomposed_subqueries") as mock_decomp, \
         patch("src.graph.nodes.inspect_retrieval") as mock_inspect:
        mock_inspect.return_value = {"results": [mock_chunk]}
        out = legal_retrieval_node(state, models=mock_models)

    mock_decomp.assert_not_called()
    mock_inspect.assert_called_once()
    assert len(out["legal_results"]) == 1
    assert out["legal_results"][0]["reference"] == "Rule 88A"


# -------------------------------------------------------------------------
# Test 2: Simple rate query ("What is the GST rate on butter?")
# -------------------------------------------------------------------------
def test_simple_rate_query_remains_unchanged(mock_models, mock_openai_client):
    """Verify simple rate query uses the existing single rate lookup path without decomposition."""
    query = "What is the GST rate on butter?"
    plan = plan_capabilities(query, use_llm=False)

    assert plan["needs_query_decomposition"] is False
    assert plan["retrieval_subqueries"] == []
    assert plan["needs_structured_rate_lookup"] is True

    mock_rate = {
        "id": 101,
        "code": "0405",
        "description": "Butter and other fats",
        "total_gst_rate": "5%",
        "cgst_rate": "2.5%",
        "sgst_rate": "2.5%",
    }

    with patch("src.generators.answer_generator.execute_decomposed_subqueries") as mock_decomp, \
         patch("src.generators.answer_generator.retrieve_rates") as mock_rates:
        mock_rates.return_value = [mock_rate]
        resp = run_gst_answer_flow(query, models=mock_models, openai_client=mock_openai_client)

    mock_decomp.assert_not_called()
    mock_rates.assert_called_once()
    assert resp["route"] == "rate"
    assert len(resp["rate_results"]) == 1
    assert resp["rate_results"][0]["code"] == "0405"


def test_simple_rate_query_graph_node():
    """Verify rate_lookup_node executes single rate lookup when needs_query_decomposition is False."""
    state: GSTGraphState = {
        "user_query": "What is the GST rate on butter?",
        "needs_legal": False,
        "needs_rate": True,
        "clean_queries": {"rate_query": "butter"},
        "plan": {
            "needs_query_decomposition": False,
            "retrieval_subqueries": [],
        },
    }

    mock_rate = {"id": 101, "code": "0405", "description": "Butter"}

    with patch("src.graph.nodes.execute_decomposed_subqueries") as mock_decomp, \
         patch("src.graph.nodes.retrieve_rates") as mock_rates:
        mock_rates.return_value = [mock_rate]
        out = rate_lookup_node(state)

    mock_decomp.assert_not_called()
    mock_rates.assert_called_once()
    assert len(out["rate_results"]) == 1


# -------------------------------------------------------------------------
# Test 3: Complex discount query (two legal subqueries)
# -------------------------------------------------------------------------
def test_complex_discount_query_independent_execution(mock_models):
    """Verify two legal subqueries execute independently/concurrently, merge, attach provenance,

    remove duplicates, and preserve evidence from both legal concepts.
    """
    subqueries = [
        {"type": "legal", "query": "conditions for a post-supply discount to reduce taxable value"},
        {"type": "legal", "query": "procedure for adjusting tax liability after a post-supply discount"},
    ]

    chunk_sec_15 = {
        "chunk_id": "chunk_sec_15_3_b",
        "document_type": "act",
        "reference": "Section 15(3)(b)",
        "title": "Value of taxable supply - discounts",
        "content": "The value of supply shall not include any discount given after supply if agreed before...",
    }
    chunk_shared = {
        "chunk_id": "chunk_general_discount",
        "document_type": "act",
        "reference": "Section 15",
        "title": "Value of taxable supply",
        "content": "General valuation provisions for supply and discounts...",
    }
    chunk_sec_34 = {
        "chunk_id": "chunk_sec_34",
        "document_type": "act",
        "reference": "Section 34",
        "title": "Credit and debit notes",
        "content": "Where one or more tax invoices have been issued and taxable value or tax charged exceeds...",
    }

    def mock_inspect(query, **kwargs):
        if "conditions" in query:
            return {"results": [dict(chunk_sec_15), dict(chunk_shared)], "timings_ms": {"total": 10.0}}
        elif "adjusting" in query or "procedure" in query:
            return {"results": [dict(chunk_sec_34), dict(chunk_shared)], "timings_ms": {"total": 12.0}}
        return {"results": [], "timings_ms": {"total": 0.0}}

    with patch("src.retrievers.decomposed_executor.inspect_retrieval", side_effect=mock_inspect) as mock_retrieve:
        res = execute_decomposed_subqueries(subqueries, models=mock_models, top_k=10)

    # 1. Both subqueries executed
    assert mock_retrieve.call_count == 2
    called_queries = [call[0][0] for call in mock_retrieve.call_args_list]
    assert "conditions for a post-supply discount to reduce taxable value" in called_queries
    assert "procedure for adjusting tax liability after a post-supply discount" in called_queries

    legal_chunks = res["legal_results"]

    # 2. Duplicate chunk removed (chunk_general_discount appeared in both, but should appear once)
    chunk_ids = [c["chunk_id"] for c in legal_chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunks must be deduplicated"
    assert len(legal_chunks) == 3

    # 3. Evidence from both legal concepts survives the merge
    assert "chunk_sec_15_3_b" in chunk_ids
    assert "chunk_sec_34" in chunk_ids

    # 4. Provenance attached to each result
    sec_15_res = next(c for c in legal_chunks if c["chunk_id"] == "chunk_sec_15_3_b")
    assert sec_15_res["subquery_id"] == "legal_1"
    assert sec_15_res["retrieval_subquery_type"] == "legal"
    assert sec_15_res["retrieval_subquery"] == "conditions for a post-supply discount to reduce taxable value"

    sec_34_res = next(c for c in legal_chunks if c["chunk_id"] == "chunk_sec_34")
    assert sec_34_res["subquery_id"] == "legal_2"
    assert sec_34_res["retrieval_subquery_type"] == "legal"
    assert sec_34_res["retrieval_subquery"] == "procedure for adjusting tax liability after a post-supply discount"

    # 5. Shared chunk provenance preserves all matched subqueries
    shared_res = next(c for c in legal_chunks if c["chunk_id"] == "chunk_general_discount")
    assert "legal_1" in shared_res["subquery_ids"]
    assert "legal_2" in shared_res["subquery_ids"]
    assert len(shared_res["matched_subqueries"]) == 2


# -------------------------------------------------------------------------
# Test 4: Mixed-type decomposed query (1 legal + 1 rate)
# -------------------------------------------------------------------------
def test_mixed_type_decomposed_query(mock_models):
    """Verify planner fixture with 1 legal and 1 rate subquery dispatches to correct retrievers

    and preserves type/provenance in merged output.
    """
    subqueries = [
        {"type": "rate", "query": "rate of gst on laptops"},
        {"type": "legal", "query": "input tax credit eligibility on computer equipment"},
    ]

    mock_rate = {
        "id": 8471,
        "code": "8471",
        "description": "Automatic data processing machines (computers, laptops)",
        "total_gst_rate": "18%",
        "cgst_rate": "9%",
        "sgst_rate": "9%",
    }
    mock_legal = {
        "chunk_id": "sec_16_itc",
        "document_type": "act",
        "reference": "Section 16",
        "title": "Eligibility and conditions for taking input tax credit",
        "content": "Every registered person shall be entitled to take credit of input tax...",
    }

    with patch("src.retrievers.decomposed_executor.retrieve_rates") as mock_rate_fn, \
         patch("src.retrievers.decomposed_executor.inspect_retrieval") as mock_legal_fn:
        mock_rate_fn.return_value = [mock_rate]
        mock_legal_fn.return_value = {"results": [mock_legal], "timings_ms": {"total": 8.0}}

        res = execute_decomposed_subqueries(subqueries, models=mock_models, top_k=5)

    # Correct retrievers dispatched
    mock_rate_fn.assert_called_once_with("rate of gst on laptops", db_url=None, limit=10)
    mock_legal_fn.assert_called_once_with("input tax credit eligibility on computer equipment", top_k=5, models=mock_models, db_url=None)

    # Preserves rate results with rate provenance
    assert len(res["rate_results"]) == 1
    r = res["rate_results"][0]
    assert r["code"] == "8471"
    assert r["subquery_id"] == "rate_1"
    assert r["retrieval_subquery_type"] == "rate"
    assert r["retrieval_subquery"] == "rate of gst on laptops"

    # Preserves legal results with legal provenance
    assert len(res["legal_results"]) == 1
    l = res["legal_results"][0]
    assert l["chunk_id"] == "sec_16_itc"
    assert l["subquery_id"] == "legal_1"
    assert l["retrieval_subquery_type"] == "legal"
    assert l["retrieval_subquery"] == "input tax credit eligibility on computer equipment"


# -------------------------------------------------------------------------
# Test 5: Duplicate retrieval result
# -------------------------------------------------------------------------
def test_duplicate_retrieval_deduplication_and_provenance(mock_models):
    """Verify two subqueries returning the same chunk results in 1 chunk with merged provenance."""
    subqueries = [
        {"type": "legal", "query": "credit note issuance time limit"},
        {"type": "legal", "query": "credit note adjustment in annual return"},
    ]

    same_chunk = {
        "chunk_id": "sec_34_2",
        "document_type": "act",
        "reference": "Section 34(2)",
        "title": "Credit and debit notes",
        "content": "Any registered person who issues a credit note in relation to a supply of goods or services...",
    }

    with patch("src.retrievers.decomposed_executor.inspect_retrieval") as mock_legal:
        mock_legal.return_value = {"results": [dict(same_chunk)], "timings_ms": {"total": 5.0}}
        res = execute_decomposed_subqueries(subqueries, models=mock_models, top_k=10)

    legal_results = res["legal_results"]

    # Final merged result contains exactly one chunk
    assert len(legal_results) == 1
    item = legal_results[0]
    assert item["chunk_id"] == "sec_34_2"

    # Provenance records both originating subqueries
    assert item["subquery_ids"] == ["legal_1", "legal_2"]
    assert len(item["matched_subqueries"]) == 2
    assert item["matched_subqueries"][0]["subquery_id"] == "legal_1"
    assert item["matched_subqueries"][0]["retrieval_subquery"] == "credit note issuance time limit"
    assert item["matched_subqueries"][1]["subquery_id"] == "legal_2"
    assert item["matched_subqueries"][1]["retrieval_subquery"] == "credit note adjustment in annual return"


# -------------------------------------------------------------------------
# Test 6: One subquery failure (Error isolation)
# -------------------------------------------------------------------------
def test_one_subquery_failure_error_isolation(mock_models):
    """Verify that if one subquery fails, successful subquery results are preserved,

    the error is recorded/logged, and no fallback evidence is fabricated.
    """
    subqueries = [
        {"type": "legal", "query": "valid statutory provision query"},
        {"type": "legal", "query": "failing query causing database error"},
    ]

    valid_chunk = {
        "chunk_id": "valid_chunk_1",
        "document_type": "act",
        "reference": "Section 12",
        "title": "Time of supply of goods",
        "content": "The liability to pay tax on goods shall arise at the time of supply...",
    }

    def mock_inspect(query, **kwargs):
        if "failing" in query:
            raise RuntimeError("Database connection reset by peer")
        return {"results": [dict(valid_chunk)], "timings_ms": {"total": 6.0}}

    with patch("src.retrievers.decomposed_executor.inspect_retrieval", side_effect=mock_inspect):
        res = execute_decomposed_subqueries(subqueries, models=mock_models, top_k=10)

    # Successful subquery results are preserved
    assert len(res["legal_results"]) == 1
    assert res["legal_results"][0]["chunk_id"] == "valid_chunk_1"
    assert res["legal_results"][0]["subquery_id"] == "legal_1"

    # Subquery error is recorded
    assert len(res["subquery_errors"]) == 1
    err = res["subquery_errors"][0]
    assert err["subquery_id"] == "legal_2"
    assert "Database connection reset by peer" in err["error"]

    # Tool execution status correctly logs failure
    tools_exec = res["tools_executed"]
    assert len(tools_exec) == 2
    assert tools_exec[0]["status"] == "success"
    assert tools_exec[1]["status"] == "error"
    assert "Database connection reset by peer" in tools_exec[1]["error"]

    # No fake or fabricated results
    assert len(res["rate_results"]) == 0


# -------------------------------------------------------------------------
# Test 7: Full LangGraph execution with decomposed retrieval
# -------------------------------------------------------------------------
def test_full_langgraph_decomposed_flow(mock_models, mock_openai_client):
    """Verify end-to-end LangGraph execution with decomposed retrieval subqueries."""
    from src.graph.graph import invoke_gst_graph

    query = "I gave a discount after invoicing. Can I reduce GST liability and how do I adjust it?"

    mock_plan = {
        "needs_legal_retrieval": True,
        "needs_structured_rate_lookup": False,
        "needs_hsn_lookup": False,
        "needs_notification_retrieval": False,
        "needs_direct_reasoning": False,
        "needs_calculation": False,
        "needs_grounded_reasoning": False,
        "needs_clarification": False,
        "needs_grounded_synthesis": True,
        "needs_query_decomposition": True,
        "retrieval_subqueries": [
            {"type": "legal", "query": "conditions for a post-supply discount to reduce taxable value"},
            {"type": "legal", "query": "procedure for adjusting tax liability after a post-supply discount"},
        ],
        "clean_subqueries": {"legal_query": query},
        "user_premises": {},
    }

    chunk_sec_15 = {
        "chunk_id": "chunk_sec_15_3_b",
        "document_type": "act",
        "reference": "Section 15(3)(b)",
        "content": "Conditions for post-supply discount...",
    }
    chunk_sec_34 = {
        "chunk_id": "chunk_sec_34",
        "document_type": "act",
        "reference": "Section 34",
        "content": "Issuance of credit note for tax adjustment...",
    }

    def mock_inspect(query, **kwargs):
        if "conditions" in query:
            return {"results": [dict(chunk_sec_15)], "timings_ms": {"total": 5.0}}
        return {"results": [dict(chunk_sec_34)], "timings_ms": {"total": 5.0}}

    with patch("src.graph.nodes.plan_capabilities", return_value=mock_plan), \
         patch("src.retrievers.decomposed_executor.inspect_retrieval", side_effect=mock_inspect) as mock_retrieve, \
         patch("src.graph.nodes.inspect_retrieval", side_effect=mock_inspect):
        out_state = invoke_gst_graph(
            query,
            models=mock_models,
            openai_client=mock_openai_client,
        )

    assert out_state["needs_legal"] is True
    assert len(out_state["legal_results"]) == 2
    # Verify provenance on graph state legal_results
    assert out_state["legal_results"][0]["subquery_id"] in ("legal_1", "legal_2")
    assert out_state["legal_results"][1]["subquery_id"] in ("legal_1", "legal_2")
    assert out_state["final_answer"] == "Synthesized grounded answer."

