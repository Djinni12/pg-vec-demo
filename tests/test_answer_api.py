from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app import app
from src.generators.answer_generator import GenerationError
from src.retrieval_inspector import LoadedModels


@pytest.fixture
def client():
    app.state.models = LoadedModels(embedding_model=MagicMock(), reranker=MagicMock(), initialization_ms=50.0)
    return TestClient(app)


def test_chat_endpoint_success(client):
    mock_response = {
        "query": "How to cancel GST registration?",
        "answer": "Under Section 29, registration can be cancelled by the proper officer.\n\nSources:\n- Section 29",
        "sources": [
            {
                "rank": 1,
                "document_type": "act",
                "reference": "Section 29",
                "title": "Cancellation of registration",
                "chunk_id": "act_s29",
                "content": "Full section text...",
                "snippet": "Snippet text...",
                "reranker_score": 0.92,
            }
        ],
        "sources_used": [
            {
                "rank": 1,
                "document_type": "act",
                "reference": "Section 29",
                "title": "Cancellation of registration",
                "chunk_id": "act_s29",
                "content": "Full section text...",
                "snippet": "Snippet text...",
                "reranker_score": 0.92,
            }
        ],
        "retrieval_timing": 120.4,
        "generation_timing": 450.2,
        "total_timing": 570.6,
        "model_used": "gemma-4-31b-it",
        "model": "gemma-4-31b-it",
        "timings_ms": {
            "retrieval": 120.4,
            "generation": 450.2,
            "total": 570.6,
            "dense": 40.0,
            "bm25": 20.0,
            "rrf": 10.4,
            "reranker": 50.0,
        },
        "retrieval_debug": {
            "results": [],
            "dense_results": [],
            "bm25_results": [],
            "hybrid_results": [],
            "metadata": {"embedding_model": "BAAI/bge-m3"},
        },
    }

    with patch("app.run_gst_answer_flow", return_value=mock_response) as mock_flow:
        resp = client.post("/chat", json={"query": "How to cancel GST registration?", "top_k": 5})

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"].startswith("Under Section 29")
    assert len(data["sources"]) == 1
    assert data["sources"][0]["reference"] == "Section 29"
    assert data["retrieval_timing"] == 120.4
    assert data["generation_timing"] == 450.2
    assert data["total_timing"] == 570.6
    assert data["model_used"] == "gemma-4-31b-it"
    assert "retrieval_debug" in data

    mock_flow.assert_called_once_with(
        "How to cancel GST registration?",
        top_k=5,
        models=app.state.models,
        openai_model=None,
    )


def test_chat_endpoint_empty_query(client):
    resp = client.post("/chat", json={"query": "", "top_k": 5})
    assert resp.status_code == 422


def test_chat_endpoint_generation_error(client):
    with patch("app.run_gst_answer_flow", side_effect=GenerationError("OpenAI API failure")):
        resp = client.post("/chat", json={"query": "Valid query", "top_k": 5})

    assert resp.status_code == 502
    assert "OpenAI API failure" in resp.json()["detail"]


def test_search_endpoint_preserved(client):
    mock_retrieval = {
        "query": "registration",
        "results": [{"rank": 1, "chunk_id": "c1", "content": "text"}],
        "dense_results": [],
        "bm25_results": [],
        "hybrid_results": [],
        "timings_ms": {"total": 50.0},
        "metadata": {},
        "models": {},
        "config": {},
    }

    with patch("app.inspect_retrieval", return_value=mock_retrieval) as mock_inspect:
        resp = client.post("/search", json={"query": "registration", "top_k": 5})

    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "registration"
    assert len(data["results"]) == 1
    mock_inspect.assert_called_once()


def test_chat_stream_endpoint_success(client):
    mock_events = [
        {"type": "meta", "query": "test", "sources": [], "retrieval_timing": 10.0, "model_used": "test-model"},
        {"type": "token", "delta": "Hello "},
        {"type": "token", "delta": "GST!"},
        {"type": "done", "answer": "Hello GST!", "generation_timing": 50.0, "total_timing": 60.0},
    ]

    with patch("app.stream_gst_answer_flow", return_value=iter(mock_events)):
        resp = client.post("/chat/stream", json={"query": "test", "top_k": 5})

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    content = resp.text
    assert "data: " in content
    assert "Hello GST!" in content


def test_rates_endpoint_success(client):
    mock_rates = [
        {
            "item_type": "goods",
            "code": "8471",
            "description": "Automatic data processing machines",
            "formatted_rate": "18% IGST",
        }
    ]
    with patch("src.retrievers.rate_retriever.retrieve_rates", return_value=mock_rates):
        resp = client.post("/rates", json={"query": "8471", "limit": 5})

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["code"] == "8471"


def test_chat_endpoint_rate_response(client):
    mock_response = {
        "query": "What is the GST rate on chocolate?",
        "route": "rate",
        "answer": "Chocolate attracts 18% IGST under HSN 1806.",
        "rate_results": [
            {
                "item_type": "goods",
                "code": "1806",
                "description": "Chocolate",
                "formatted_rate": "18% IGST (9% CGST + 9% SGST)",
                "cgst_rate_pct": 9.0,
                "sgst_utgst_rate_pct": 9.0,
                "igst_rate_pct": 18.0,
                "compensation_cess": None,
                "condition": None,
            }
        ],
        "sources": [],
        "sources_used": [],
        "retrieval_timing": 12.5,
        "generation_timing": 250.0,
        "total_timing": 262.5,
        "model_used": "test-model",
        "model": "test-model",
        "timings_ms": {
            "retrieval": 12.5,
            "generation": 250.0,
            "total": 262.5,
            "rate_lookup": 12.5,
            "dense": 0.0,
            "bm25": 0.0,
            "rrf": 0.0,
            "reranker": 0.0,
        },
        "retrieval_debug": {},
    }

    with patch("app.run_gst_answer_flow", return_value=mock_response):
        resp = client.post("/chat", json={"query": "What is the GST rate on chocolate?"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["route"] == "rate"
    assert len(data["rate_results"]) == 1
    assert data["rate_results"][0]["code"] == "1806"
    assert data["timings_ms"]["rate_lookup"] == 12.5


def test_chat_endpoint_calls_run_graph_chat(client):
    with patch("app.run_graph_chat") as mock_graph:
        mock_graph.return_value = {
            "query": "What is GST on paneer?",
            "answer": "Paneer attracts 0% or 5% GST.",
            "sources": [],
            "rate_results": [],
            "retrieval_timing": 10.0,
            "generation_timing": 20.0,
            "total_timing": 30.0,
            "model_used": "test-model",
            "model": "test-model",
            "timings_ms": {},
            "retrieval_debug": {},
        }
        resp = client.post("/chat", json={"query": "What is GST on paneer?"})
        assert resp.status_code == 200
        mock_graph.assert_called_once()

