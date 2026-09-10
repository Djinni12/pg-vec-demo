"""Tests for vector-based supporting notification retrieval and LangGraph integration."""

import pytest
from unittest.mock import MagicMock, patch

from src.graph.graph import build_gst_graph, compile_gst_graph, invoke_gst_graph
from src.graph.nodes import notification_support_node, grounded_reasoning_node
from src.graph.routing import route_capabilities, route_post_retrieval, route_post_notification_support
from src.graph.state import GSTGraphState
from src.retrievers.notification_retriever import (
    retrieve_notifications,
    retrieve_supporting_notifications,
)


def test_notification_support_node_with_rate_results():
    """Verify notification_support_node builds semantic query and retrieves Gazette chunks."""
    state: GSTGraphState = {
        "user_query": "What is the GST rate on butter?",
        "clean_queries": {"rate_query": "butter"},
        "rate_results": [
            {
                "code": "0405",
                "hsn_code": "0405",
                "description": "Butter and other fats and oils derived from milk, including ghee; dairy spreads",
                "notification_no": "09/2025-Central Tax (Rate)",
                "schedule": "Schedule I – 2.5%",
                "serial_no": "7.",
            }
        ],
        "legal_results": [],
    }

    mock_notif_chunk = {
        "chunk_id": "chunk-09-2025-central_tax_rate-0007",
        "notification_number": "09/2025-Central Tax (Rate)",
        "reference": "Notification No. 09/2025-Central Tax (Rate) | Effective: 2025-09-22 | Schedule I | S. No. 7",
        "title": "Notification 09/2025-Central Tax (Rate)",
        "content": "7. 0405 Butter and other fats and oils derived from milk...",
        "score": 0.95,
        "source_metadata": {
            "notification_number": "09/2025-Central Tax (Rate)",
            "effective_date": "2025-09-22",
            "operation_type": None,
            "target_notification": None,
        },
    }

    with patch(
        "src.retrievers.notification_retriever.retrieve_supporting_notifications",
        return_value=[mock_notif_chunk],
    ) as mock_retrieve:
        out = notification_support_node(state)
        assert "notification_results" in out
        assert len(out["notification_results"]) == 1
        assert out["notification_results"][0]["chunk_id"] == "chunk-09-2025-central_tax_rate-0007"
        mock_retrieve.assert_called_once()
        args, kwargs = mock_retrieve.call_args
        assert "0405" in args[0]
        assert kwargs["support_metadata"]["hsn_code"] == "0405"
        assert kwargs["support_metadata"]["serial_no"] == "7."


def test_notification_support_node_with_legal_results():
    """Verify notification_support_node uses clean legal query for service/legal exemption lookups."""
    state: GSTGraphState = {
        "user_query": "જીવન વીમાની પોલિસી પર અત્યારે GST લાગે છે કે તેમાં કોઈ છૂટ આપવામાં આવી છે?",
        "clean_queries": {"legal_query": "GST exemption on life insurance services"},
        "rate_results": [],
        "legal_results": [
            {
                "chunk_id": "act-cgst-sec-11",
                "reference": "Section 11",
                "title": "Power to grant exemption from tax",
                "content": "Where the Government is satisfied...",
            }
        ],
    }

    mock_notif_chunk = {
        "chunk_id": "chunk-16-2025-central_tax_rate-0002",
        "notification_number": "16/2025-Central Tax (Rate)",
        "reference": "Notification No. 16/2025-Central Tax (Rate) | Amending Notification No. 12/2017-Central Tax (Rate)",
        "title": "Notification 16/2025-Central Tax (Rate) (Amends 12/2017)",
        "content": "Services of life insurance business provided by an insurer to an individual...",
        "score": 0.60,
    }

    with patch(
        "src.retrievers.notification_retriever.retrieve_supporting_notifications",
        return_value=[mock_notif_chunk],
    ) as mock_retrieve:
        out = notification_support_node(state)
        assert "notification_results" in out
        assert len(out["notification_results"]) == 1
        assert out["notification_results"][0]["chunk_id"] == "chunk-16-2025-central_tax_rate-0002"
        mock_retrieve.assert_called_once()
        args, _ = mock_retrieve.call_args
        assert args[0] == "GST exemption on life insurance services"


def test_grounded_reasoning_evaluates_notification_amendment():
    """Verify grounded_reasoning_node reconciles amending notifications per Source Interpretation Rules."""
    state: GSTGraphState = {
        "user_query": "What applies to this rate?",
        "user_premises": {},
        "rate_results": [
            {
                "code": "0405",
                "description": "Butter",
                "total_gst_rate": "5%",
                "cgst_rate": "2.5%",
                "sgst_rate": "2.5%",
                "notification_no": "09/2025-Central Tax (Rate)",
            }
        ],
        "legal_results": [],
        "notification_results": [
            {
                "chunk_id": "chunk-19-2025",
                "reference": "Notification No. 19/2025-Central Tax (Rate) | Effective: 2025-10-01",
                "source_metadata": {
                    "notification_number": "19/2025-Central Tax (Rate)",
                    "target_notification": "09/2025-Central Tax (Rate)",
                    "operation_type": "SUBSTITUTE",
                    "effective_date": "2025-10-01",
                },
            }
        ],
    }

    out = grounded_reasoning_node(state)
    reasoning = out["reasoning_result"]
    assert "19/2025" in reasoning
    assert "SUBSTITUTE" in reasoning
    assert "09/2025" in reasoning


def test_graph_routing_with_notification_support():
    """Verify graph builder registers notification_support and correctly connects edges."""
    builder = build_gst_graph(use_llm_planner=False)
    assert "notification_support" in builder.nodes

    # Test route_capabilities when only notification is requested
    state_notif: GSTGraphState = {
        "user_query": "Explain Notification 09/2025",
        "needs_legal": False,
        "needs_rate": False,
        "needs_notification": True,
        "needs_direct_reasoning": False,
    }
    targets = route_capabilities(state_notif)
    assert targets == ["notification_support"]

    # Test route_post_notification_support
    state_reasoning: GSTGraphState = {
        "needs_grounded_reasoning": True,
    }
    assert route_post_notification_support(state_reasoning) == "grounded_reasoning"


def test_full_graph_invocation_with_supporting_notifications():
    """End-to-end execution of graph verifying notification_results populated in state."""
    mock_notif_chunk = {
        "chunk_id": "chunk-09-2025-central_tax_rate-0007",
        "reference": "Notification No. 09/2025-Central Tax (Rate) | Effective: 2025-09-22 | S. No. 7",
        "title": "Notification 09/2025-Central Tax (Rate)",
        "content": "Butter attracts 2.5% CGST under S. No. 7",
        "score": 0.95,
        "source_metadata": {
            "notification_number": "09/2025-Central Tax (Rate)",
            "effective_date": "2025-09-22",
            "operation_type": None,
        },
    }

    with patch(
        "src.retrievers.notification_retriever.retrieve_supporting_notifications",
        return_value=[mock_notif_chunk],
    ):
        res = invoke_gst_graph("What is the GST rate on butter?", use_llm_planner=False)
        assert "notification_results" in res
        assert len(res["notification_results"]) == 1
        assert res["notification_results"][0]["chunk_id"] == "chunk-09-2025-central_tax_rate-0007"
        assert res["final_answer"] != ""
