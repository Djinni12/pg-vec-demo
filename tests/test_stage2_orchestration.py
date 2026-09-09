"""Tests for Stage 2 Multi-Capability Orchestration.

Verifies:
1. Critical regression query: Butter + ITC 2500 each interstate -> rate 5% + Rule 88A + Calculator -> ₹1,50,000.
2. Knowledge-base gap: Insufficient legal context stops calculation and reports gap.
3. Observability block completeness: PLAN, TOOLS EXECUTED, RETRIEVED EVIDENCE,
   STRUCTURED LEGAL FINDINGS, CALCULATION INPUTS, CALCULATION RESULT, SOURCES USED.
4. Calculator purity: Arithmetic only, no statutory decisions.
5. Direct reasoning calculations: Assumed rate and discount + tax.
6. Streaming flow observability.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.routers.query_router import plan_capabilities
from src.tools.calculator import (
    CalculationInputs,
    execute_calculator,
    calculate_max_taxable_value_from_credit,
    calculate_tax_on_value,
    calculate_discount_and_tax,
)
from src.generators.legal_interpreter import (
    interpret_legal_findings,
    interpret_legal_findings_deterministic,
    StructuredLegalFindings,
)
from src.generators.answer_generator import (
    run_gst_answer_flow,
    stream_gst_answer_flow,
    build_user_prompt,
)
from src.retrieval_inspector import LoadedModels


@pytest.fixture
def mock_models():
    return LoadedModels(embedding_model=MagicMock(), reranker=MagicMock(), initialization_ms=5.0)


@pytest.fixture
def mock_client():
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = "Based on Rule 88A, you can supply up to ₹1,50,000 of butter without additional cash GST."
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


# -------------------------------------------------------------------------
# 1. Deterministic Calculator Arithmetic Purity Tests
# -------------------------------------------------------------------------
def test_calculator_pure_arithmetic_max_taxable_value():
    """Calculator divides credit by rate; contains no legal rules."""
    res = calculate_max_taxable_value_from_credit(
        eligible_credit=7500.0,
        tax_rate_pct=5.0,
        credit_breakdown={"igst": 2500.0, "cgst": 2500.0, "sgst": 2500.0},
    )
    assert res.status == "success"
    assert res.result_value == 150000.0
    assert "₹7,500.00 / 0.05 = ₹150,000.00" in res.formula
    assert len(res.steps) >= 4


def test_calculator_invalid_rate():
    res = calculate_max_taxable_value_from_credit(eligible_credit=7500.0, tax_rate_pct=0.0)
    assert res.status == "error"
    assert "strictly greater than 0%" in res.error_message


def test_calculator_tax_on_value():
    res = calculate_tax_on_value(taxable_value=80000.0, tax_rate_pct=12.0)
    assert res.status == "success"
    assert res.result_value == 9600.0


def test_calculator_discount_and_tax():
    res = calculate_discount_and_tax(base_amount=20000.0, discount_pct=10.0, tax_rate_pct=5.0)
    assert res.status == "success"
    assert res.result_value == 18900.0


# -------------------------------------------------------------------------
# 2. Structured Legal Findings Interpreter Tests
# -------------------------------------------------------------------------
def test_legal_interpreter_rule88a_interstate_resolution():
    chunks = [
        {
            "document_type": "rule",
            "reference": "Rule 88A",
            "title": "Order of utilization of input tax credit",
            "chunk_id": "rule_r88a",
            "content": "Input tax credit on account of integrated tax shall first be utilised towards payment of integrated tax and the amount remaining, if any, may be utilised towards the payment of central tax and State tax...",
        }
    ]
    user_premises = {
        "supply_type": "interstate",
        "itc_balances": {"cgst": 2500.0, "sgst": 2500.0, "igst": 2500.0},
    }
    findings = interpret_legal_findings_deterministic(chunks, user_premises)
    assert findings.status == "resolved"
    assert findings.supply_type == "interstate"
    assert findings.output_tax_type == "IGST"
    assert findings.usable_credit_ledgers == ["igst", "cgst", "sgst"]
    assert findings.total_usable_credit == 7500.0
    assert len(findings.utilization_constraints) == 2
    assert len(findings.legal_evidence) == 1
    assert findings.legal_evidence[0].reference == "Rule 88A"


def test_legal_interpreter_knowledge_base_gap():
    """When retrieved chunks do not contain Rule 88A or Section 49, findings are marked unresolved."""
    irrelevant_chunks = [
        {
            "document_type": "act",
            "reference": "Section 10",
            "title": "Composition levy",
            "chunk_id": "act_s10",
            "content": "Notwithstanding anything to the contrary contained in this Act but subject to the provisions of sub-sections (3) and (4) of section 9...",
        }
    ]
    user_premises = {
        "supply_type": "interstate",
        "itc_balances": {"cgst": 2500.0, "sgst": 2500.0, "igst": 2500.0},
    }
    findings = interpret_legal_findings_deterministic(irrelevant_chunks, user_premises)
    assert findings.status == "unresolved"
    assert findings.unresolved_reason is not None
    assert "knowledge-base gap" in findings.unresolved_reason.lower()


# -------------------------------------------------------------------------
# 3. Critical Regression Test: Butter + ITC 2500 each interstate
# -------------------------------------------------------------------------
def test_stage2_critical_regression_butter_itc_interstate(mock_models, mock_client):
    query = (
        "I have GST credit of CGST 2500, SGST 2500 and IGST 2500. "
        "I want to send butter from one state to another. "
        "What taxable value can I supply without making an additional GST cash payment?"
    )
    mock_rate_item = {
        "code": "0405",
        "item_type": "goods",
        "description": "Butter and other fats and oils derived from milk, including ghee; dairy spreads",
        "total_gst_rate": "5%",
        "cgst_rate": "2.5%",
        "sgst_rate": "2.5%",
        "igst_rate_pct": 5.0,
        "rate_category": "CGST",
        "source_rate": "2.5%",
        "notification_number": "01/2017-Central Tax (Rate)",
        "serial_no": "8",
    }
    mock_legal_chunk = {
        "document_type": "rule",
        "reference": "Rule 88A",
        "title": "Order of utilization of input tax credit",
        "chunk_id": "rule_r88a",
        "content": (
            "Input tax credit on account of integrated tax shall first be utilised towards "
            "payment of integrated tax, and the amount remaining, if any, may be utilised towards "
            "payment of central tax and State tax... and credit of central tax or State tax shall "
            "be utilised towards integrated tax only after credit of integrated tax has been utilised fully."
        ),
    }

    with patch("src.generators.answer_generator.retrieve_rates", return_value=[mock_rate_item]) as mock_rates, \
         patch("src.generators.answer_generator.inspect_retrieval") as mock_legal:
        mock_legal.return_value = {
            "results": [mock_legal_chunk],
            "timings_ms": {"total": 15.0, "dense": 5.0, "bm25": 5.0, "rrf": 2.0, "reranker": 3.0},
        }
        resp = run_gst_answer_flow(query, models=mock_models, openai_client=mock_client)

    mock_rates.assert_called_once_with("butter", db_url=None, limit=5)
    mock_legal.assert_called_once()

    # Verify Route
    assert resp["route"] == "mixed"

    # Verify Structured Legal Findings
    findings = resp["structured_legal_findings"]
    assert findings is not None
    assert findings["status"] == "resolved"
    assert findings["supply_type"] == "interstate"
    assert findings["output_tax_type"] == "IGST"
    assert findings["usable_credit_ledgers"] == ["igst", "cgst", "sgst"]
    assert findings["total_usable_credit"] == 7500.0

    # Verify Calculator Result
    calc = resp["calculation_result"]
    assert calc is not None
    assert calc["status"] == "success"
    assert calc["operation"] == "max_taxable_value_from_credit"
    assert calc["result_value"] == 150000.0
    assert "₹7,500.00 / 0.05 = ₹150,000.00" in calc["formula"]

    # Verify Tools Executed
    tools = [t["tool"] for t in resp["tools_executed"]]
    assert "retrieve_rates" in tools
    assert "inspect_retrieval" in tools
    assert "interpret_legal_findings" in tools
    assert "execute_calculator" in tools

    # Verify Observability Payload
    obs = resp["observability"]
    assert obs is not None
    assert obs["plan"]["needs_structured_rate_lookup"] is True
    assert obs["plan"]["needs_legal_retrieval"] is True
    assert obs["plan"]["needs_calculation"] is True
    assert obs["retrieved_evidence"]["rates_count"] == 1
    assert obs["retrieved_evidence"]["legal_chunks_count"] == 1
    assert obs["structured_legal_findings"]["status"] == "resolved"
    assert obs["calculation_result"]["result_value"] == 150000.0
    assert obs["knowledge_base_gap"] is None

    # Verify Prompt Grounding passed to LLM
    call_messages = mock_client.chat.completions.create.call_args[1]["messages"]
    user_prompt = call_messages[1]["content"]
    assert "STRUCTURED LEGAL FINDINGS" in user_prompt
    assert "Rule 88A" in user_prompt
    assert "CALCULATOR ARITHMETIC RESULT" in user_prompt
    assert "150,000" in user_prompt
    assert "₹7,500" in user_prompt


# -------------------------------------------------------------------------
# 4. Knowledge-Base Gap Test: Missing Rule 88A halts calculation
# -------------------------------------------------------------------------
def test_stage2_kb_gap_halts_calculation(mock_models, mock_client):
    query = (
        "I have GST credit of CGST 2500, SGST 2500 and IGST 2500. "
        "I want to send butter from one state to another. "
        "What taxable value can I supply without making an additional GST cash payment?"
    )
    mock_rate_item = {
        "code": "0405",
        "item_type": "goods",
        "description": "Butter",
        "total_gst_rate": "5%",
        "igst_rate_pct": 5.0,
    }
    # Legal retrieval returns irrelevant chunks (gap in knowledge base)
    mock_chunk = {
        "document_type": "act",
        "reference": "Section 1",
        "title": "Short title, extent and commencement",
        "content": "This Act may be called the Central Goods and Services Tax Act, 2017.",
    }

    with patch("src.generators.answer_generator.retrieve_rates", return_value=[mock_rate_item]), \
         patch("src.generators.answer_generator.inspect_retrieval") as mock_legal:
        mock_legal.return_value = {"results": [mock_chunk], "timings_ms": {}}
        resp = run_gst_answer_flow(query, models=mock_models, openai_client=mock_client)

    # Legal findings must be marked unresolved
    findings = resp["structured_legal_findings"]
    assert findings["status"] == "unresolved"
    assert resp["observability"]["knowledge_base_gap"] is not None

    # Calculator must NOT have produced a result
    assert resp["calculation_result"] is None

    # Calculator tool must be marked skipped
    calc_tool = next((t for t in resp["tools_executed"] if t["tool"] == "execute_calculator"), None)
    assert calc_tool is not None
    assert calc_tool["status"] == "skipped"
    assert calc_tool["reason"] == "unresolved_legal_findings"

    # User prompt must include knowledge-base gap notice
    call_messages = mock_client.chat.completions.create.call_args[1]["messages"]
    user_prompt = call_messages[1]["content"]
    assert "Knowledge-Base Gap Notice" in user_prompt
    assert "CALCULATOR ARITHMETIC RESULT" not in user_prompt


# -------------------------------------------------------------------------
# 5. Direct Reasoning with Calculator Tests
# -------------------------------------------------------------------------
def test_direct_reasoning_discount_and_tax_calculator(mock_models, mock_client):
    query = "₹20,000, 10% discount, then 5% GST. Final amount?"
    resp = run_gst_answer_flow(query, models=mock_models, openai_client=mock_client)

    assert resp["route"] == "direct"
    assert resp["calculation_result"] is not None
    assert resp["calculation_result"]["operation"] == "discount_and_tax"
    assert resp["calculation_result"]["result_value"] == 18900.0


def test_direct_reasoning_assumed_rate_calculator(mock_models, mock_client):
    query = "Assume GST is 12%. Calculate GST on ₹80,000."
    resp = run_gst_answer_flow(query, models=mock_models, openai_client=mock_client)

    assert resp["route"] == "direct"
    assert resp["calculation_result"] is not None
    assert resp["calculation_result"]["operation"] == "tax_on_value"
    assert resp["calculation_result"]["result_value"] == 9600.0


# -------------------------------------------------------------------------
# 6. Streaming Flow Observability Test
# -------------------------------------------------------------------------
def test_streaming_flow_observability(mock_models, mock_client):
    query = (
        "I have GST credit of CGST 2500, SGST 2500 and IGST 2500. "
        "I want to send butter from one state to another. "
        "What taxable value can I supply without making an additional GST cash payment?"
    )
    mock_rate_item = {
        "code": "0405",
        "item_type": "goods",
        "description": "Butter",
        "total_gst_rate": "5%",
        "igst_rate_pct": 5.0,
    }
    mock_legal_chunk = {
        "document_type": "rule",
        "reference": "Rule 88A",
        "title": "Order of utilization",
        "content": "credit of integrated tax shall first be utilised towards payment of integrated tax...",
    }

    with patch("src.generators.answer_generator.retrieve_rates", return_value=[mock_rate_item]), \
         patch("src.generators.answer_generator.inspect_retrieval") as mock_legal:
        mock_legal.return_value = {"results": [mock_legal_chunk], "timings_ms": {}}

        # Mock streaming chunks from openai
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="You can supply "))]
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content="up to ₹1,50,000."))]
        mock_client.chat.completions.create.return_value = [chunk1, chunk2]

        events = list(stream_gst_answer_flow(query, models=mock_models, openai_client=mock_client))

    meta_event = next(e for e in events if e.get("type") == "meta")
    done_event = next(e for e in events if e.get("type") == "done")

    assert meta_event["observability"] is not None
    assert meta_event["observability"]["calculation_result"]["result_value"] == 150000.0

    assert done_event["observability"] is not None
    assert done_event["observability"]["structured_legal_findings"]["status"] == "resolved"
    assert "₹1,50,000" in done_event["answer"]
