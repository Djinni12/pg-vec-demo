"""Unit and integration tests for the GST LangGraph architecture."""

from unittest.mock import MagicMock, patch
import pytest

from src.graph.graph import build_gst_graph, compile_gst_graph, invoke_gst_graph
from src.graph.routing import route_capabilities, route_post_retrieval
from src.graph.state import GSTGraphState
from src.retrieval_inspector import LoadedModels


@pytest.fixture
def mock_models():
    return LoadedModels(embedding_model=MagicMock(), reranker=MagicMock(), initialization_ms=5.0)


@pytest.fixture
def mock_openai_client():
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = "Synthesized grounded answer from mock LLM."
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


# -----------------------------------------------------------------------------
# Test 1: State Structure and TypedDict
# -----------------------------------------------------------------------------
def test_gst_graph_state_structure():
    state: GSTGraphState = {
        "user_query": "Test query",
        "needs_legal": True,
        "needs_rate": False,
        "needs_direct_reasoning": False,
        "clean_queries": {"legal_query": "Test legal query"},
        "user_premises": {"taxable_amount": 1000.0},
        "legal_results": [{"reference": "Section 29"}],
        "rate_results": [],
        "reasoning_result": None,
        "final_answer": "Test answer",
        "sources": [{"reference": "Section 29"}],
        "error": None,
    }
    assert state["user_query"] == "Test query"
    assert state["needs_legal"] is True
    assert len(state["legal_results"]) == 1
    assert len(state["sources"]) == 1


# -----------------------------------------------------------------------------
# Test 2: Conditional Routing Edge Logic
# -----------------------------------------------------------------------------
def test_routing_legal_only():
    state: GSTGraphState = {
        "user_query": "How do I cancel my GST registration?",
        "needs_legal": True,
        "needs_rate": False,
        "needs_direct_reasoning": False,
    }
    targets = route_capabilities(state)
    assert targets == ["legal_retrieval"]


def test_routing_rate_only():
    state: GSTGraphState = {
        "user_query": "What is the GST rate for paneer?",
        "needs_legal": False,
        "needs_rate": True,
        "needs_direct_reasoning": False,
    }
    targets = route_capabilities(state)
    assert targets == ["rate_lookup"]


def test_routing_direct_reasoning_only():
    state: GSTGraphState = {
        "user_query": "₹20,000 with 10% discount and 5% GST",
        "needs_legal": False,
        "needs_rate": False,
        "needs_direct_reasoning": True,
    }
    targets = route_capabilities(state)
    assert targets == ["direct_reasoning"]


def test_routing_mixed_legal_and_rate():
    state: GSTGraphState = {
        "user_query": "What is the GST rate for laptops and what law applies?",
        "needs_legal": True,
        "needs_rate": True,
        "needs_direct_reasoning": False,
    }
    targets = route_capabilities(state)
    assert "legal_retrieval" in targets
    assert "rate_lookup" in targets
    assert len(targets) == 2


def test_routing_all_three_capabilities():
    state: GSTGraphState = {
        "user_query": "I have ITC of 2500 and want to supply butter. What law and rate apply?",
        "needs_legal": True,
        "needs_rate": True,
        "needs_direct_reasoning": True,
    }
    # Initial capability routing executes retrievals first to fetch required rates/laws
    initial_targets = route_capabilities(state)
    assert set(initial_targets) == {"legal_retrieval", "rate_lookup"}

    # Post-retrieval conditional edge routes to grounded_reasoning with retrieved evidence
    post_target = route_post_retrieval(state)
    assert post_target == "grounded_reasoning"


def test_routing_fallback_to_synthesis():
    state: GSTGraphState = {
        "user_query": "Hello",
        "needs_legal": False,
        "needs_rate": False,
        "needs_direct_reasoning": False,
    }
    targets = route_capabilities(state)
    assert targets == ["synthesis"]


# -----------------------------------------------------------------------------
# Test 3: Graph Building and Compilation
# -----------------------------------------------------------------------------
def test_graph_builder_and_nodes():
    builder = build_gst_graph(use_llm_planner=False)
    graph = builder.compile()

    # Verify registered node names
    assert "planner" in graph.nodes
    assert "legal_retrieval" in graph.nodes
    assert "rate_lookup" in graph.nodes
    assert "grounded_reasoning" in graph.nodes
    assert "direct_reasoning" in graph.nodes
    assert "calculation" in graph.nodes
    assert "synthesis" in graph.nodes


# -----------------------------------------------------------------------------
# Test 4: Invocation - Legal Query
# -----------------------------------------------------------------------------
def test_invocation_legal_cancellation(mock_models, mock_openai_client):
    query = "How do I cancel my GST registration?"

    mock_chunk = {
        "document_type": "act",
        "reference": "Section 29",
        "title": "Cancellation or suspension of registration",
        "chunk_id": "act_s29",
        "content": "The proper officer may cancel the registration of a taxable person...",
    }

    with patch("src.graph.nodes.inspect_retrieval") as mock_inspect:
        mock_inspect.return_value = {
            "results": [mock_chunk],
            "timings_ms": {"total": 10.0},
        }

        output = invoke_gst_graph(
            query,
            models=mock_models,
            openai_client=mock_openai_client,
            use_llm_planner=False,
        )

    assert output["needs_legal"] is True
    assert output["needs_rate"] is False
    assert len(output["legal_results"]) == 1
    assert output["legal_results"][0]["reference"] == "Section 29"
    assert len(output["rate_results"]) == 0
    assert len(output["sources"]) >= 1
    assert any("Section 29" in s["reference"] for s in output["sources"])


# -----------------------------------------------------------------------------
# Test 5: Invocation - Rate Query (Paneer)
# -----------------------------------------------------------------------------
def test_invocation_rate_paneer(mock_models, mock_openai_client):
    query = "What is the GST rate for paneer?"

    mock_rate = {
        "code": "0406",
        "item_type": "goods",
        "description": "Chena or paneer, whether or not pre-packaged and labelled",
        "total_gst_rate": "0%",
        "source_rate": "Nil",
        "rate_category": "EXEMPTION",
        "notification_number": "02/2017-Central Tax (Rate)",
        "serial_no": "27",
    }

    with patch("src.graph.nodes.retrieve_rates", return_value=[mock_rate]) as mock_retrieve:
        output = invoke_gst_graph(
            query,
            models=mock_models,
            openai_client=mock_openai_client,
            use_llm_planner=False,
        )

    assert output["needs_rate"] is True
    assert output["needs_legal"] is False
    assert len(output["rate_results"]) == 1
    assert output["rate_results"][0]["code"] == "0406"
    assert len(output["sources"]) == 1
    assert "0406" in output["sources"][0]["reference"]


# -----------------------------------------------------------------------------
# Test 6: Invocation - Mixed Query (Laptops + Law)
# -----------------------------------------------------------------------------
def test_invocation_mixed_laptop_and_law(mock_models, mock_openai_client):
    query = "What is the GST rate for laptops and what law applies?"

    mock_rate = {
        "code": "8471",
        "item_type": "goods",
        "description": "Automatic data processing machines and units thereof",
        "total_gst_rate": "18%",
        "cgst_rate": "9%",
        "sgst_rate": "9%",
        "rate_category": "CGST",
        "source_rate": "9%",
        "notification_number": "01/2017-Central Tax (Rate)",
        "serial_no": "360",
    }
    mock_chunk = {
        "document_type": "rule",
        "reference": "Rule 27",
        "title": "Value of supply of goods or services where consideration is not wholly in money",
        "chunk_id": "rule_r27",
        "content": "Where a laptop is supplied for forty thousand rupees along with barter of printer...",
    }

    with patch("src.graph.nodes.retrieve_rates", return_value=[mock_rate]), \
         patch("src.graph.nodes.inspect_retrieval") as mock_inspect:
        mock_inspect.return_value = {
            "results": [mock_chunk],
            "timings_ms": {"total": 15.0},
        }

        output = invoke_gst_graph(
            query,
            models=mock_models,
            openai_client=mock_openai_client,
            use_llm_planner=False,
        )

    # Verify BOTH capabilities ran and merged their state
    assert output["needs_rate"] is True
    assert output["needs_legal"] is True
    assert len(output["rate_results"]) == 1
    assert output["rate_results"][0]["code"] == "8471"
    assert len(output["legal_results"]) == 1
    assert output["legal_results"][0]["reference"] == "Rule 27"

    # Verify combined sources preserved
    refs = [s["reference"] for s in output["sources"]]
    assert any("8471" in r for r in refs)
    assert any("Rule 27" in r for r in refs)


# -----------------------------------------------------------------------------
# Test 7: Invocation - Direct Reasoning / Arithmetic Calculation (Flow C)
# -----------------------------------------------------------------------------
def test_invocation_direct_reasoning_discount_and_tax(mock_models, mock_openai_client):
    query = "Assume GST is 12%. Calculate GST on ₹80,000."

    with patch("src.graph.nodes.retrieve_rates") as mock_retrieve, \
         patch("src.graph.nodes.inspect_retrieval") as mock_inspect:
        output = invoke_gst_graph(
            query,
            models=mock_models,
            openai_client=mock_openai_client,
            use_llm_planner=False,
        )

    # Verify neither rate nor legal was retrieved, but reasoning and calculation ran
    mock_retrieve.assert_not_called()
    mock_inspect.assert_not_called()
    assert output["needs_direct_reasoning"] is True
    assert output["needs_calculation"] is True
    assert output["reasoning_result"] is not None
    assert "80,000" in output["reasoning_result"]
    assert output["calculation_result"] is not None
    assert output["calculation_result"]["status"] == "success"
    assert output["calculation_result"]["result_value"] == 9600.0


# -----------------------------------------------------------------------------
# Test 8: Invocation - Dependent Mixed Flow (Flow D: Butter + Interstate + Credit)
# -----------------------------------------------------------------------------
def test_invocation_flow_d_dependent_butter_itc(mock_models, mock_openai_client):
    query = (
        "I have input tax credit of CGST 2500, SGST 2500, IGST 2500. "
        "I am supplying butter interstate. What maximum value can I supply "
        "without making an additional GST cash payment?"
    )

    mock_rate = {
        "code": "0405",
        "item_type": "goods",
        "description": "Butter and other fats and oils derived from milk",
        "total_gst_rate": "5%",
        "cgst_rate": "2.5%",
        "sgst_rate": "2.5%",
        "rate_category": "CGST",
        "source_rate": "2.5%",
        "notification_number": "01/2017-Central Tax (Rate)",
        "serial_no": "8",
    }
    mock_chunk = {
        "document_type": "act",
        "reference": "Section 49",
        "title": "Payment of tax, interest, penalty and other amounts",
        "chunk_id": "act_s49",
        "content": "The input tax credit on account of integrated tax shall first be utilised towards payment of integrated tax...",
    }

    with patch("src.graph.nodes.retrieve_rates", return_value=[mock_rate]), \
         patch("src.graph.nodes.inspect_retrieval") as mock_inspect:
        mock_inspect.return_value = {
            "results": [mock_chunk],
            "timings_ms": {"total": 12.0},
        }

        output = invoke_gst_graph(
            query,
            models=mock_models,
            openai_client=mock_openai_client,
            use_llm_planner=False,
        )

    # 1. Verify capability flags
    assert output["needs_legal"] is True
    assert output["needs_rate"] is True
    assert output["needs_grounded_reasoning"] is True
    assert output["needs_calculation"] is True

    # 2. Verify retrieval results populated
    assert len(output["rate_results"]) == 1
    assert output["rate_results"][0]["code"] == "0405"
    assert len(output["legal_results"]) == 1
    assert output["legal_results"][0]["reference"] == "Section 49"

    # 3. Verify grounded reasoning output (statutory grounding, no manual math)
    assert output["reasoning_result"] is not None
    assert "Section 49" in output["reasoning_result"] or "Rule 88A" in output["reasoning_result"]
    assert "7,500.00" in output["reasoning_result"]

    # 4. Verify structured calculation inputs
    assert output["calculation_inputs"] is not None
    assert output["calculation_inputs"]["operation"] == "max_taxable_value_from_credit"
    assert output["calculation_inputs"]["available_eligible_credit"] == 7500.0
    assert output["calculation_inputs"]["tax_rate_pct"] == 5.0

    # 5. Verify deterministic calculation executed
    assert output["calculation_result"] is not None
    assert output["calculation_result"]["status"] == "success"
    assert output["calculation_result"]["result_value"] == 150000.0
    assert "150,000.00" in output["calculation_result"]["steps"][-1]
