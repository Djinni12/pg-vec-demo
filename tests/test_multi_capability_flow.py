"""Comprehensive tests for multi-capability GST chatbot orchestration.

Validates the 11 key real-world scenarios:
1. Pure reasoning (English): ₹20,000, 10% discount, then 5% GST. Final amount?
2. Multilingual reasoning (Gujarati): કોઈ વસ્તુની કિંમત ₹20,000 છે...
3. Multilingual reasoning (Hindi): किसी वस्तु का मूल्य ₹20,000 है...
4. Pure retrieval (Butter): What is the HSN code for butter?
5. Retrieval + reasoning: Find applicable GST rate for butter and calculate GST on ₹50,000.
6. Legal retrieval: Order of utilization of IGST, CGST and SGST ITC (Rule 88A / Section 49).
7. Legal + reasoning: Credit balances 2500 each utilized against interstate supply.
8. Combined ITC + Butter (Legal + Rate + Calc): Credit 2500 each + interstate supply of butter.
9. Assumed rate: Assume GST is 12%. Calculate GST on ₹80,000.
10. Ambiguous query: I have ₹5,000 GST credit. How much can I sell? (Clarification)
11. Temporal reasoning: Notification chronology and amendments.
12. Exception reasoning: General rule vs exception.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.routers.query_router import plan_capabilities
from src.generators.answer_generator import (
    run_gst_answer_flow,
    DIRECT_REASONING_SYSTEM_PROMPT,
)
from src.retrieval_inspector import LoadedModels


@pytest.fixture
def mock_models():
    return LoadedModels(embedding_model=MagicMock(), reranker=MagicMock(), initialization_ms=10.0)


@pytest.fixture
def mock_client():
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = "Calculated amount: ₹18,900."
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


# -------------------------------------------------------------------------
# Scenario 1: Pure Reasoning (English)
# -------------------------------------------------------------------------
def test_pure_reasoning_english(mock_models, mock_client):
    query = "₹20,000, 10% discount, then 5% GST. Final amount?"
    plan = plan_capabilities(query)

    assert plan["needs_direct_reasoning"] is True
    assert plan["needs_calculation"] is True
    assert plan["needs_grounded_synthesis"] is True
    assert plan["needs_structured_rate_lookup"] is False
    assert plan["needs_legal_retrieval"] is False
    assert plan["needs_notification_retrieval"] is False
    assert plan["user_premises"]["taxable_amount"] == 20000.0
    assert plan["user_premises"]["discount_pct"] == 10.0

    with patch("src.generators.answer_generator.retrieve_rates") as mock_rates, \
         patch("src.generators.answer_generator.inspect_retrieval") as mock_legal:
        resp = run_gst_answer_flow(query, models=mock_models, openai_client=mock_client)

    mock_rates.assert_not_called()
    mock_legal.assert_not_called()
    assert resp["route"] == "direct"
    assert resp["plan"]["needs_calculation"] is True
    call_messages = mock_client.chat.completions.create.call_args[1]["messages"]
    assert call_messages[0]["content"] == DIRECT_REASONING_SYSTEM_PROMPT


# -------------------------------------------------------------------------
# Scenario 2 & 3: Multilingual Reasoning (Gujarati & Hindi)
# -------------------------------------------------------------------------
def test_multilingual_reasoning_gujarati(mock_models, mock_client):
    query = "કોઈ વસ્તુની કિંમત ₹20,000 છે. પહેલા કિંમત પર 10% ડિસ્કાઉન્ટ આપવામાં આવે છે અને પછી બાકી રકમ પર 5% GST લગાવવામાં આવે છે. અંતિમ રકમ કેટલી થશે?"
    plan = plan_capabilities(query)

    assert plan["needs_direct_reasoning"] is True
    assert plan["needs_calculation"] is True
    assert plan["needs_structured_rate_lookup"] is False
    assert plan["user_premises"]["taxable_amount"] == 20000.0
    assert plan["user_premises"]["discount_pct"] == 10.0

    with patch("src.generators.answer_generator.retrieve_rates") as mock_rates, \
         patch("src.generators.answer_generator.inspect_retrieval") as mock_legal:
        resp = run_gst_answer_flow(query, models=mock_models, openai_client=mock_client)

    mock_rates.assert_not_called()
    mock_legal.assert_not_called()
    assert resp["route"] == "direct"


def test_multilingual_reasoning_hindi(mock_models, mock_client):
    query = "किसी वस्तु का मूल्य ₹20,000 है। पहले 10% छूट दी जाती है और फिर शेष राशि पर 5% GST लगाया जाता है। अंतिम राशि क्या होगी?"
    plan = plan_capabilities(query)

    assert plan["needs_direct_reasoning"] is True
    assert plan["needs_calculation"] is True
    assert plan["needs_structured_rate_lookup"] is False
    assert plan["user_premises"]["taxable_amount"] == 20000.0
    assert plan["user_premises"]["discount_pct"] == 10.0

    with patch("src.generators.answer_generator.retrieve_rates") as mock_rates, \
         patch("src.generators.answer_generator.inspect_retrieval") as mock_legal:
        resp = run_gst_answer_flow(query, models=mock_models, openai_client=mock_client)

    mock_rates.assert_not_called()
    mock_legal.assert_not_called()
    assert resp["route"] == "direct"


# -------------------------------------------------------------------------
# Scenario 4: Pure Retrieval (Butter)
# -------------------------------------------------------------------------
def test_pure_retrieval_butter(mock_models, mock_client):
    query = "What is the HSN code for butter?"
    plan = plan_capabilities(query)

    assert plan["needs_structured_rate_lookup"] is True
    assert plan["needs_hsn_lookup"] is True
    assert plan["needs_direct_reasoning"] is False
    assert plan["clean_subqueries"]["rate_query"] == "butter"

    mock_rate_item = {
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

    with patch("src.generators.answer_generator.retrieve_rates", return_value=[mock_rate_item]) as mock_rates, \
         patch("src.generators.answer_generator.inspect_retrieval") as mock_legal:
        resp = run_gst_answer_flow(query, models=mock_models, openai_client=mock_client)

    mock_rates.assert_called_once_with("butter", db_url=None, limit=5)
    mock_legal.assert_not_called()
    assert resp["route"] == "rate"
    assert len(resp["rate_results"]) == 1
    assert resp["rate_results"][0]["code"] == "0405"


# -------------------------------------------------------------------------
# Scenario 5: Retrieval + Reasoning (Butter + Calculation)
# -------------------------------------------------------------------------
def test_retrieval_plus_reasoning_butter_calculation(mock_models, mock_client):
    query = "Find the applicable GST rate for butter and calculate GST on ₹50,000."
    plan = plan_capabilities(query)

    assert plan["needs_structured_rate_lookup"] is True
    assert plan["needs_hsn_lookup"] is True
    assert plan["needs_calculation"] is True
    assert plan["needs_direct_reasoning"] is True
    assert plan["clean_subqueries"]["rate_query"] == "butter"
    assert plan["user_premises"]["taxable_amount"] == 50000.0

    mock_rate_item = {
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

    with patch("src.generators.answer_generator.retrieve_rates", return_value=[mock_rate_item]) as mock_rates:
        resp = run_gst_answer_flow(query, models=mock_models, openai_client=mock_client)

    mock_rates.assert_called_once_with("butter", db_url=None, limit=5)
    assert resp["route"] == "rate"
    call_messages = mock_client.chat.completions.create.call_args[1]["messages"]
    user_prompt = call_messages[1]["content"]
    assert "0405" in user_prompt
    assert "50,000" in user_prompt or "50000" in user_prompt


# -------------------------------------------------------------------------
# Scenario 6: Legal Retrieval (ITC Order of Utilization)
# -------------------------------------------------------------------------
def test_legal_retrieval_itc_order(mock_models, mock_client):
    query = "What is the order of utilization of IGST, CGST and SGST ITC?"
    plan = plan_capabilities(query)

    assert plan["needs_legal_retrieval"] is True
    assert plan["clean_subqueries"]["legal_query"] is not None

    mock_chunk = {
        "document_type": "act",
        "reference": "Section 49",
        "title": "Payment of tax, interest, penalty and other amounts",
        "chunk_id": "act_s49",
        "content": "Input tax credit on account of integrated tax shall first be utilised towards payment of integrated tax...",
    }

    with patch("src.generators.answer_generator.inspect_retrieval") as mock_legal:
        mock_legal.return_value = {
            "results": [mock_chunk],
            "timings_ms": {"total": 15.0, "dense": 5.0, "bm25": 5.0, "rrf": 2.0, "reranker": 3.0},
        }
        resp = run_gst_answer_flow(query, models=mock_models, openai_client=mock_client)

    assert resp["route"] == "legal"
    assert len(resp["sources"]) >= 1
    assert resp["sources"][0]["reference"] == "Section 49"


# -------------------------------------------------------------------------
# Scenario 7: Legal + Reasoning (ITC Balances vs Interstate Supply)
# -------------------------------------------------------------------------
def test_legal_plus_reasoning_itc_balances(mock_models, mock_client):
    query = "I have CGST 2500, SGST 2500 and IGST 2500. How can these credits be utilized against an interstate supply?"
    plan = plan_capabilities(query)

    assert plan["needs_legal_retrieval"] is True
    assert plan["needs_direct_reasoning"] is True
    assert plan["user_premises"]["itc_balances"] == {"cgst": 2500.0, "sgst": 2500.0, "igst": 2500.0}
    assert plan["user_premises"]["supply_type"] == "interstate"

    mock_chunk = {
        "document_type": "rule",
        "reference": "Rule 88A",
        "title": "Order of utilization of input tax credit",
        "chunk_id": "rule_r88a",
        "content": "Input tax credit on account of integrated tax shall first be utilised towards payment of integrated tax...",
    }

    with patch("src.generators.answer_generator.inspect_retrieval") as mock_legal:
        mock_legal.return_value = {
            "results": [mock_chunk],
            "timings_ms": {"total": 12.0, "dense": 4.0, "bm25": 4.0, "rrf": 2.0, "reranker": 2.0},
        }
        resp = run_gst_answer_flow(query, models=mock_models, openai_client=mock_client)

    assert resp["route"] == "legal"
    call_messages = mock_client.chat.completions.create.call_args[1]["messages"]
    user_prompt = call_messages[1]["content"]
    assert "Rule 88A" in user_prompt
    assert "2,500" in user_prompt or "2500" in user_prompt
    assert "interstate" in user_prompt.lower()


# -------------------------------------------------------------------------
# Scenario 8: Combined ITC + Butter (Rate + Legal + Calculation + Reasoning)
# -------------------------------------------------------------------------
def test_combined_itc_plus_butter_multi_capability(mock_models, mock_client):
    query = "I have CGST credit 2500, SGST credit 2500 and IGST credit 2500. I want to make an interstate supply of butter. What taxable value can I supply without additional cash GST payment?"
    plan = plan_capabilities(query)

    assert plan["needs_structured_rate_lookup"] is True
    assert plan["needs_hsn_lookup"] is True
    assert plan["needs_legal_retrieval"] is True
    assert plan["needs_calculation"] is True
    assert plan["needs_direct_reasoning"] is True
    assert plan["clean_subqueries"]["rate_query"] == "butter"
    assert "utilization" in plan["clean_subqueries"]["legal_query"]
    assert plan["user_premises"]["itc_balances"] == {"cgst": 2500.0, "sgst": 2500.0, "igst": 2500.0}
    assert plan["user_premises"]["supply_type"] == "interstate"

    mock_rate_item = {
        "code": "0405",
        "item_type": "goods",
        "description": "Butter and other fats and oils derived from milk, including ghee; dairy spreads",
        "total_gst_rate": "5%",
        "cgst_rate": "2.5%",
        "sgst_rate": "2.5%",
        "rate_category": "CGST",
        "source_rate": "2.5%",
        "notification_number": "01/2017-Central Tax (Rate)",
        "serial_no": "8",
    }
    mock_legal_chunk = {
        "document_type": "act",
        "reference": "Section 49",
        "title": "Payment of tax, interest, penalty and other amounts",
        "chunk_id": "act_s49",
        "content": "Input tax credit on account of central tax or State tax shall be utilised towards payment of integrated tax only after integrated tax credit is exhausted.",
    }

    with patch("src.generators.answer_generator.retrieve_rates", return_value=[mock_rate_item]) as mock_rates, \
         patch("src.generators.answer_generator.inspect_retrieval") as mock_legal:
        mock_legal.return_value = {
            "results": [mock_legal_chunk],
            "timings_ms": {"total": 20.0, "dense": 8.0, "bm25": 6.0, "rrf": 3.0, "reranker": 3.0},
        }
        resp = run_gst_answer_flow(query, models=mock_models, openai_client=mock_client)

    mock_rates.assert_called_once_with("butter", db_url=None, limit=5)
    mock_legal.assert_called_once()
    assert resp["route"] == "mixed"

    refs = [s["reference"] for s in resp["sources"]]
    assert any("0405" in r for r in refs)
    assert any("Section 49" in r for r in refs)

    call_messages = mock_client.chat.completions.create.call_args[1]["messages"]
    user_prompt = call_messages[1]["content"]
    assert "0405" in user_prompt
    assert "Section 49" in user_prompt
    assert "2,500" in user_prompt or "2500" in user_prompt


# -------------------------------------------------------------------------
# Scenario 9: Assumed Rate (Hypothetical Input, No DB Hallucination)
# -------------------------------------------------------------------------
def test_assumed_rate_hypothetical_input(mock_models, mock_client):
    query = "Assume GST is 12%. Calculate GST on ₹80,000."
    plan = plan_capabilities(query)

    assert plan["needs_direct_reasoning"] is True
    assert plan["needs_calculation"] is True
    assert plan["needs_structured_rate_lookup"] is False
    assert plan["user_premises"]["assumed_rate"] == 12.0
    assert plan["user_premises"]["taxable_amount"] == 80000.0

    with patch("src.generators.answer_generator.retrieve_rates") as mock_rates, \
         patch("src.generators.answer_generator.inspect_retrieval") as mock_legal:
        resp = run_gst_answer_flow(query, models=mock_models, openai_client=mock_client)

    mock_rates.assert_not_called()
    mock_legal.assert_not_called()
    assert resp["route"] == "direct"
    call_messages = mock_client.chat.completions.create.call_args[1]["messages"]
    user_prompt = call_messages[1]["content"]
    assert "Assume GST is 12%" in user_prompt


# -------------------------------------------------------------------------
# Scenario 10: Ambiguous Query (Clarification Fast-Path)
# -------------------------------------------------------------------------
def test_ambiguous_query_requests_clarification(mock_models, mock_client):
    query = "I have ₹5,000 GST credit. How much can I sell?"
    plan = plan_capabilities(query)

    assert plan["needs_clarification"] is True

    with patch("src.generators.answer_generator.retrieve_rates") as mock_rates, \
         patch("src.generators.answer_generator.inspect_retrieval") as mock_legal:
        resp = run_gst_answer_flow(query, models=mock_models, openai_client=mock_client)

    mock_rates.assert_not_called()
    mock_legal.assert_not_called()
    mock_client.chat.completions.create.assert_not_called()
    assert resp["route"] == "clarification"
    assert "rate" in resp["answer"].lower() or "product" in resp["answer"].lower()


# -------------------------------------------------------------------------
# Scenario 11: Temporal Reasoning (Notifications & Amendments)
# -------------------------------------------------------------------------
def test_temporal_reasoning_notifications(mock_models, mock_client):
    query = "Notification A introduced a rate, Notification B amended it, and a corrigendum changed it. What applies now?"
    plan = plan_capabilities(query)

    assert plan["needs_notification_retrieval"] is True
    assert plan["needs_temporal_reasoning"] is True

    mock_notif_chunk = {
        "chunk_id": "notif_09_2025_c1",
        "notification_number": "09/2025-Central Tax (Rate)",
        "reference": "Notification No. 09/2025-Central Tax (Rate)",
        "title": "Notification 09/2025-Central Tax (Rate)",
        "content": "In exercise of powers conferred by sub-section (1) of section 9...",
        "score": 0.88,
        "reranker_score": 0.88,
    }

    with patch("src.generators.answer_generator.retrieve_notifications", return_value=[mock_notif_chunk]) as mock_notif, \
         patch("src.generators.answer_generator.inspect_retrieval") as mock_legal:
        mock_legal.return_value = {"results": [], "timings_ms": {}}
        resp = run_gst_answer_flow(query, models=mock_models, openai_client=mock_client)

    mock_notif.assert_called_once()
    assert resp["route"] == "legal"
    call_messages = mock_client.chat.completions.create.call_args[1]["messages"]
    user_prompt = call_messages[1]["content"]
    assert "09/2025" in user_prompt


# -------------------------------------------------------------------------
# Scenario 12: Exception Reasoning (Rule vs Exception)
# -------------------------------------------------------------------------
def test_exception_reasoning_legal_rules(mock_models, mock_client):
    query = "Rule 89 normally applies. Does exception for inverted duty structure apply here?"
    plan = plan_capabilities(query)

    assert plan["needs_legal_retrieval"] is True
    assert plan["needs_exception_reasoning"] is True

    mock_chunk = {
        "document_type": "rule",
        "reference": "Rule 89",
        "title": "Application for refund of tax, interest, penalty, fees or any other amount",
        "chunk_id": "rule_r89",
        "content": "Provided that in case of inverted duty structure...",
    }

    with patch("src.generators.answer_generator.inspect_retrieval") as mock_legal:
        mock_legal.return_value = {
            "results": [mock_chunk],
            "timings_ms": {"total": 10.0, "dense": 4.0, "bm25": 4.0, "rrf": 1.0, "reranker": 1.0},
        }
        resp = run_gst_answer_flow(query, models=mock_models, openai_client=mock_client)

    mock_legal.assert_called_once()
    assert resp["route"] == "legal"
    assert resp["plan"]["needs_exception_reasoning"] is True
