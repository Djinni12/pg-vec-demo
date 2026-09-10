from unittest.mock import MagicMock, patch
import os
import pytest
import openai

@pytest.fixture(autouse=True)
def disable_llm_planner_in_unit_tests(monkeypatch):
    monkeypatch.setenv("USE_LLM_PLANNER", "false")

from src.generators.answer_generator import (
    DIRECT_REASONING_SYSTEM_PROMPT,
    GenerationError,
    SYSTEM_PROMPT,
    build_direct_full_prompt,
    build_full_prompt,
    build_user_prompt,
    extract_sources,
    format_chunk,
    format_context,
    generate_answer,
    get_configured_model,
    run_gst_answer_flow,
    strip_reasoning,
)
from src.retrieval_inspector import LoadedModels


def test_format_chunk_includes_all_required_metadata():
    chunk = {
        "document_type": "act",
        "reference": "Section 29",
        "title": "Cancellation or suspension of registration",
        "chunk_id": "cgst_act_s29_c1",
        "content": "The proper officer may cancel registration of a taxable person.",
    }
    formatted = format_chunk(chunk, 1)

    assert "--- Source 1 ---" in formatted
    assert "Document Type: act" in formatted
    assert "Reference: Section 29" in formatted
    assert "Title: Cancellation or suspension of registration" in formatted
    assert "Chunk ID: cgst_act_s29_c1" in formatted
    assert "The proper officer may cancel registration of a taxable person." in formatted


def test_format_context_empty_and_multiple():
    assert format_context([]) == "No retrieved context available."

    chunks = [
        {
            "document_type": "act",
            "reference": "Section 29",
            "title": "Cancellation",
            "chunk_id": "c1",
            "content": "Content 1",
        },
        {
            "document_type": "rule",
            "reference": "Rule 22",
            "title": "Cancellation procedure",
            "chunk_id": "c2",
            "content": "Content 2",
        },
    ]
    formatted = format_context(chunks)
    assert "--- Source 1 ---" in formatted
    assert "--- Source 2 ---" in formatted
    assert "Rule 22" in formatted


def test_build_full_prompt_structure():
    chunks = [
        {
            "document_type": "form",
            "reference": "GST REG-16",
            "title": "Application for Cancellation",
            "chunk_id": "form_reg16",
            "content": "Form to apply for cancellation.",
        }
    ]
    query = "Which form is used to cancel GST registration?"
    prompt = build_full_prompt(query, chunks)

    assert prompt.startswith("SYSTEM PROMPT\n\n")
    assert "You are a GST legal and tax information assistant." in prompt
    assert "RETRIEVED CONTEXT\n\n" in prompt
    assert "GST REG-16" in prompt
    assert "USER QUESTION\n\nWhich form is used to cancel GST registration?" in prompt
    assert prompt.endswith("ANSWER")


def test_strip_reasoning_removes_internal_thought_tags():
    raw_with_thought = (
        "<thought>Thinking about Section 29 and Rule 22... We need to be careful.</thought>"
        "GST registration can be cancelled under Section 29 using Form GST REG-16."
    )
    cleaned = strip_reasoning(raw_with_thought)
    assert cleaned == "GST registration can be cancelled under Section 29 using Form GST REG-16."
    assert "thought" not in cleaned
    assert "Thinking about" not in cleaned


def test_strip_reasoning_removes_thinking_tags_case_insensitive():
    raw = "<THINKING>Internal logic</THINKING>Proper officer may cancel registration."
    assert strip_reasoning(raw) == "Proper officer may cancel registration."


def test_extract_sources():
    chunks = [
        {
            "rank": 1,
            "document_type": "act",
            "reference": "Section 29",
            "title": "Cancellation",
            "chunk_id": "act_s29",
            "content": "Full text of section 29...",
            "reranker_score": 0.95,
        }
    ]
    sources = extract_sources(chunks)
    assert len(sources) == 1
    assert sources[0]["document_type"] == "act"
    assert sources[0]["reference"] == "Section 29"
    assert sources[0]["title"] == "Cancellation"
    assert sources[0]["chunk_id"] == "act_s29"
    assert sources[0]["content"] == "Full text of section 29..."
    assert sources[0]["reranker_score"] == 0.95


def test_generate_answer_with_mocked_openai():
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = (
        "<thought>Analyzing context...</thought>\n"
        "Under Section 29 of the CGST Act, registration may be cancelled by the proper officer.\n\n"
        "Sources:\n- Section 29: Cancellation of registration"
    )
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion

    chunks = [
        {
            "rank": 1,
            "document_type": "act",
            "reference": "Section 29",
            "title": "Cancellation",
            "chunk_id": "c1",
            "content": "Section 29 details.",
            "reranker_score": 0.88,
        }
    ]

    result = generate_answer(
        query="How is GST registration cancelled?",
        chunks=chunks,
        model="test-gst-model",
        client=mock_client,
    )

    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "test-gst-model"
    assert call_kwargs["messages"][0]["role"] == "system"
    assert "GST legal and tax information assistant" in call_kwargs["messages"][0]["content"]
    assert "Section 29 details." in call_kwargs["messages"][1]["content"]

    assert "Analyzing context" not in result["answer"]
    assert "Under Section 29 of the CGST Act" in result["answer"]
    assert result["generation_timing"] >= 0
    assert result["model_used"] == "test-gst-model"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["chunk_id"] == "c1"


def test_generate_answer_direct_reasoning_uses_direct_prompt():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "5% of 100000 is 5000."
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    result = generate_answer(
        query="What is 5% of 100000?",
        model="test-gst-model",
        client=mock_client,
        direct_reasoning=True,
    )

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["messages"][0]["content"] == DIRECT_REASONING_SYSTEM_PROMPT
    assert "RETRIEVED CONTEXT" not in call_kwargs["messages"][1]["content"]
    assert result["answer"] == "5% of 100000 is 5000."
    assert result["sources"] == []


def test_generate_answer_empty_query_raises():
    with pytest.raises(ValueError, match="query cannot be empty"):
        generate_answer("   ", chunks=[])


def test_generate_answer_handles_api_errors():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = openai.RateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(status_code=429),
        body=None,
    )

    with pytest.raises(GenerationError, match="OpenAI rate limit exceeded"):
        generate_answer("query", chunks=[], client=mock_client)


def test_run_gst_answer_flow_orchestration():
    mock_models = LoadedModels(embedding_model=object(), reranker=object(), initialization_ms=10.0)
    mock_retrieval_data = {
        "query": "cancellation procedure",
        "results": [
            {
                "rank": 1,
                "document_type": "act",
                "reference": "Section 29",
                "title": "Cancellation",
                "chunk_id": "act_s29",
                "content": "Section 29 text",
                "reranker_score": 0.95,
            }
        ],
        "dense_results": [],
        "bm25_results": [],
        "hybrid_results": [],
        "timings_ms": {
            "total": 45.5,
            "dense": 15.0,
            "bm25": 10.0,
            "rrf": 5.0,
            "reranker": 15.5,
        },
        "metadata": {"embedding_model": "BAAI/bge-m3"},
        "models": {"embedding": "BAAI/bge-m3"},
        "config": {"final_top_k": 1},
    }

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Cancellation follows Section 29."
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    with patch(
        "src.generators.answer_generator.inspect_retrieval",
        return_value=mock_retrieval_data,
    ) as mock_inspect:
        res = run_gst_answer_flow(
            query="cancellation procedure",
            top_k=1,
            models=mock_models,
            openai_model="gemma-4-31b-it",
            openai_client=mock_client,
        )

    mock_inspect.assert_called_once_with(
        "cancellation procedure",
        top_k=1,
        models=mock_models,
        db_url=None,
    )
    assert res["answer"] == "Cancellation follows Section 29."
    assert res["retrieval_timing"] == 45.5
    assert res["generation_timing"] >= 0
    assert res["total_timing"] >= 45.5
    assert res["model_used"] == "gemma-4-31b-it"
    assert len(res["sources"]) == 1
    assert res["sources"][0]["reference"] == "Section 29"
    assert "retrieval_debug" in res
    assert res["retrieval_debug"]["metadata"]["embedding_model"] == "BAAI/bge-m3"


def test_run_gst_answer_flow_direct_percentage_skips_all_retrieval():
    mock_models = LoadedModels(embedding_model=object(), reranker=object(), initialization_ms=10.0)
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "5% of 100000 is 5000."
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    with patch("src.generators.answer_generator.retrieve_rates") as mock_ret_rates, \
         patch("src.generators.answer_generator.inspect_retrieval") as mock_ret_legal:
        res = run_gst_answer_flow(
            query="What is 5% of 100000?",
            top_k=2,
            models=mock_models,
            openai_client=mock_client,
        )

    mock_ret_rates.assert_not_called()
    mock_ret_legal.assert_not_called()
    assert res["route"] == "direct"
    assert res["answer"] == "5% of 100000 is 5000."
    assert res["sources"] == []
    assert res["retrieval_timing"] == 0.0


def test_run_gst_answer_flow_direct_taxable_value_skips_all_retrieval():
    mock_models = LoadedModels(embedding_model=object(), reranker=object(), initialization_ms=10.0)
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Taxable value = 5000 / 0.05 = 100000."
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    with patch("src.generators.answer_generator.retrieve_rates") as mock_ret_rates, \
         patch("src.generators.answer_generator.inspect_retrieval") as mock_ret_legal:
        res = run_gst_answer_flow(
            query="If GST amount is 5000 at 5%, what is taxable value?",
            top_k=2,
            models=mock_models,
            openai_client=mock_client,
        )

    mock_ret_rates.assert_not_called()
    mock_ret_legal.assert_not_called()
    assert res["route"] == "direct"
    assert "100000" in res["answer"]


def test_run_gst_answer_flow_gujarati_direct_discount_gst_skips_all_retrieval():
    mock_models = LoadedModels(embedding_model=object(), reranker=object(), initialization_ms=10.0)
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "અંતિમ રકમ ₹18,900 થશે."
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    query = (
        "કોઈ વસ્તુની કિંમત ₹20,000 છે. પહેલા કિંમત પર 10% ડિસ્કાઉન્ટ આપવામાં "
        "આવે છે અને પછી બાકી રકમ પર 5% GST લગાવવામાં આવે છે. અંતિમ રકમ કેટલી થશે?"
    )
    with patch("src.generators.answer_generator.retrieve_rates") as mock_ret_rates, \
         patch("src.generators.answer_generator.inspect_retrieval") as mock_ret_legal:
        res = run_gst_answer_flow(
            query=query,
            top_k=2,
            models=mock_models,
            openai_client=mock_client,
        )

    mock_ret_rates.assert_not_called()
    mock_ret_legal.assert_not_called()
    assert res["route"] == "direct"
    assert "₹18,900" in res["answer"]


def test_streaming_thought_filter():
    from src.generators.answer_generator import StreamingThoughtFilter

    tf = StreamingThoughtFilter()
    assert tf.process("Hello ") == "Hello "
    assert tf.process("<thought>internal reasoning</thought>World") == "World"
    assert tf.flush() == ""

    # Test splitting tags across chunks
    tf2 = StreamingThoughtFilter()
    out1 = tf2.process("Start: <tho")
    out2 = tf2.process("ught>skip me</thought")
    out3 = tf2.process(">After")
    trailing = tf2.flush()
    combined = out1 + out2 + out3 + trailing
    assert "skip me" not in combined
    assert "Start: After" in combined


def test_stream_answer_mocked():
    from src.generators.answer_generator import stream_answer

    mock_client = MagicMock()
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock(delta=MagicMock(content="Hello "))]
    chunk2 = MagicMock()
    chunk2.choices = [MagicMock(delta=MagicMock(content="GST!"))]
    mock_client.chat.completions.create.return_value = iter([chunk1, chunk2])

    chunks = [{"document_type": "act", "reference": "S1", "title": "T1", "chunk_id": "c1", "content": "C1"}]
    events = list(stream_answer("test query", chunks=chunks, client=mock_client, model="test-model"))

    assert len(events) == 2
    assert events[0] == ("token", "Hello ")
    assert events[1] == ("token", "GST!")


def test_stream_gst_answer_flow():
    from src.generators.answer_generator import stream_gst_answer_flow

    mock_models = LoadedModels(embedding_model=object(), reranker=object(), initialization_ms=10.0)
    mock_retrieval_data = {
        "query": "cancellation procedure",
        "results": [
            {
                "rank": 1,
                "document_type": "act",
                "reference": "Section 29",
                "title": "Cancellation",
                "chunk_id": "act_s29",
                "content": "Section 29 text",
                "reranker_score": 0.95,
            }
        ],
        "dense_results": [],
        "bm25_results": [],
        "hybrid_results": [],
        "timings_ms": {
            "total": 30.0,
            "dense": 10.0,
            "bm25": 10.0,
            "rrf": 5.0,
            "reranker": 5.0,
        },
        "metadata": {},
        "models": {},
        "config": {},
    }

    mock_client = MagicMock()
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock(delta=MagicMock(content="Registration "))]
    chunk2 = MagicMock()
    chunk2.choices = [MagicMock(delta=MagicMock(content="cancelled."))]
    mock_client.chat.completions.create.return_value = iter([chunk1, chunk2])

    with patch(
        "src.generators.answer_generator.inspect_retrieval",
        return_value=mock_retrieval_data,
    ):
        events = list(
            stream_gst_answer_flow(
                query="cancellation procedure",
                top_k=1,
                models=mock_models,
                openai_client=mock_client,
            )
        )

    event_types = [e["type"] for e in events]
    assert event_types == ["meta", "token", "token", "done"]

    meta_event = events[0]
    assert meta_event["retrieval_timing"] == 30.0
    assert len(meta_event["sources"]) == 1

    done_event = events[-1]
    assert done_event["answer"] == "Registration cancelled."
    assert done_event["total_timing"] >= 30.0


def test_format_rate_item_and_context():
    from src.generators.answer_generator import format_rate_context, format_rate_item

    assert format_rate_context([]) == "No matching rate entries found in the structured tariff database."

    rate_item = {
        "item_type": "goods",
        "code": "8471",
        "description": "Automatic data processing machines",
        "formatted_rate": "18% IGST (9% CGST + 9% SGST)",
        "cgst_rate_pct": 9.0,
        "sgst_utgst_rate_pct": 9.0,
        "igst_rate_pct": 18.0,
        "compensation_cess": None,
        "condition": None,
        "effective_date": "22.09.2025",
        "source_reference": "GST rates2025.pdf (Page 14, S. No. 360)",
    }
    formatted = format_rate_item(rate_item, 1)
    assert "--- Rate Item 1 (Goods) ---" in formatted
    assert "Tariff / HSN Code: 8471" in formatted
    assert "18% IGST" in formatted
    assert "Automatic data processing machines" in formatted

    context_block = format_rate_context([rate_item])
    assert "--- Rate Item 1 (Goods) ---" in context_block


def test_format_rate_item_central_tax_and_notification_date():
    from src.generators.answer_generator import extract_rate_sources, format_rate_item

    rate_item = {
        "item_type": "goods",
        "code": "0405",
        "description": "Butter and other fats and oils derived from milk",
        "section_heading": "CGST rates on goods as on 22.09.2025",
        "rate_category": "CGST",
        "source_rate": "2.5%",
        "cgst_rate": "2.5%",
        "sgst_rate": "2.5%",
        "total_gst_rate": "5%",
        "formatted_rate": "Total GST: 5% (CGST: 2.5%, SGST: 2.5%)",
        "notification_no": "09/2025-Central Tax (Rate)",
        "notification_date": "17th September, 2025",
        "rate_as_on_date": "22.09.2025",
        "effective_date": None,
        "condition": None,
        "source_reference": "GST rates2025.pdf (09/2025-Central Tax (Rate))",
    }
    formatted = format_rate_item(rate_item, 1)
    assert "Section Heading: CGST rates on goods as on 22.09.2025" in formatted
    assert "Rate Category: CGST" in formatted
    assert "Total GST Rate: 5%" in formatted
    assert "Derived CGST Rate: 2.5%" in formatted
    assert "Derived SGST Rate: 2.5%" in formatted
    assert "Notification Date: 17th September, 2025" in formatted
    assert "Rate As On Date: 22.09.2025" in formatted
    assert "Effective Date:" not in formatted

    sources = extract_rate_sources([rate_item])
    assert len(sources) == 1
    src = sources[0]
    assert "Total GST: 5%" in src["content"]
    assert "Notification Date: 17th September, 2025" in src["content"]
    assert "Rate As On Date: 22.09.2025" in src["content"]
    assert "Effective Date:" not in src["content"]
    assert "Total GST: 5%" in src["snippet"]



def test_run_gst_answer_flow_rate_route_skips_legal_retrieval():
    mock_models = LoadedModels(embedding_model=object(), reranker=object(), initialization_ms=10.0)
    mock_rates = [
        {
            "item_type": "goods",
            "code": "1806",
            "description": "Chocolates",
            "formatted_rate": "18% IGST (9% CGST + 9% SGST)",
            "cgst_rate_pct": 9.0,
            "sgst_utgst_rate_pct": 9.0,
            "igst_rate_pct": 18.0,
            "compensation_cess": None,
            "condition": None,
            "effective_date": "22.09.2025",
            "source_reference": "GST rates2025.pdf",
        }
    ]

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Chocolates attract 18% IGST under HSN 1806."
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    with patch("src.generators.answer_generator.retrieve_rates", return_value=mock_rates) as mock_ret_rates, \
         patch("src.generators.answer_generator.inspect_retrieval") as mock_ret_legal:
        res = run_gst_answer_flow(
            query="What is the GST rate on chocolate?",
            top_k=2,
            models=mock_models,
            openai_client=mock_client,
        )

    # Critical requirement: Rate queries do not execute heavy rerankers / legal retrieval
    mock_ret_legal.assert_not_called()
    mock_ret_rates.assert_called_once()
    assert res["route"] == "rate"
    assert len(res["rate_results"]) == 1
    assert res["rate_results"][0]["code"] == "1806"
    assert "18% IGST" in res["answer"]
    assert len(res["sources"]) == 1
    assert res["sources"][0]["reference"] == "HSN 1806"


def test_product_rate_query_uses_rag_not_direct():
    mock_models = LoadedModels(embedding_model=object(), reranker=object(), initialization_ms=10.0)
    mock_rates = [
        {
            "item_type": "goods",
            "code": "9999",
            "description": "X",
            "formatted_rate": "18% IGST",
            "source_reference": "GST rates2025.pdf",
        }
    ]
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "X attracts 18% IGST."
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    with patch("src.generators.answer_generator.retrieve_rates", return_value=mock_rates) as mock_ret_rates, \
         patch("src.generators.answer_generator.inspect_retrieval") as mock_ret_legal:
        res = run_gst_answer_flow(
            query="What GST rate applies to X?",
            top_k=2,
            models=mock_models,
            openai_client=mock_client,
        )

    mock_ret_rates.assert_called_once()
    mock_ret_legal.assert_not_called()
    assert res["route"] == "rate"


def test_document_rate_calculation_uses_rate_retrieval():
    mock_models = LoadedModels(embedding_model=object(), reranker=object(), initialization_ms=10.0)
    mock_rates = [
        {
            "item_type": "goods",
            "code": "9999",
            "description": "Applicable taxable supply",
            "formatted_rate": "18% IGST",
            "source_reference": "GST rates2025.pdf",
        }
    ]
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Using the retrieved 18% GST rate, tax on 100000 is 18000."
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    query = "Using the applicable GST rate from the documents, calculate tax on 100000"
    with patch("src.generators.answer_generator.retrieve_rates", return_value=mock_rates) as mock_ret_rates, \
         patch("src.generators.answer_generator.inspect_retrieval") as mock_ret_legal:
        res = run_gst_answer_flow(
            query=query,
            top_k=2,
            models=mock_models,
            openai_client=mock_client,
        )

    mock_ret_rates.assert_called_once_with(query, db_url=None, limit=2)
    mock_ret_legal.assert_not_called()
    assert res["route"] == "rate"
    assert len(res["rate_results"]) == 1
    assert "18000" in res["answer"]


def test_run_gst_answer_flow_mixed_route_executes_both():
    mock_models = LoadedModels(embedding_model=object(), reranker=object(), initialization_ms=10.0)
    mock_rates = [
        {
            "item_type": "goods",
            "code": "64",
            "description": "Footwear",
            "formatted_rate": "18% IGST",
            "cgst_rate_pct": 9.0,
            "sgst_utgst_rate_pct": 9.0,
            "igst_rate_pct": 18.0,
            "compensation_cess": None,
            "condition": None,
            "effective_date": "22.09.2025",
            "source_reference": "GST rates2025.pdf",
        }
    ]
    mock_legal = {
        "results": [
            {
                "rank": 1,
                "document_type": "rule",
                "reference": "Rule 22",
                "title": "Cancellation",
                "chunk_id": "rule_22",
                "content": "Cancellation procedure details",
                "reranker_score": 0.91,
            }
        ],
        "dense_results": [],
        "bm25_results": [],
        "hybrid_results": [],
        "timings_ms": {"total": 40.0, "dense": 10.0, "bm25": 10.0, "rrf": 5.0, "reranker": 15.0},
        "metadata": {},
        "models": {},
        "config": {},
    }

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Footwear attracts 18% GST. Cancellation procedure under Rule 22 requires..."
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    query = "GST rate on footwear and procedure for cancellation of registration under Rule 22"
    with patch("src.generators.answer_generator.retrieve_rates", return_value=mock_rates) as mock_ret_rates, \
         patch("src.generators.answer_generator.inspect_retrieval", return_value=mock_legal) as mock_ret_legal:
        res = run_gst_answer_flow(
            query=query,
            top_k=2,
            models=mock_models,
            openai_client=mock_client,
        )

    mock_ret_rates.assert_called_once()
    mock_ret_legal.assert_called_once()
    assert res["route"] == "mixed"
    assert len(res["rate_results"]) == 1
    assert len(res["sources"]) == 2
    assert res["sources"][0]["reference"] == "HSN 64"
    assert res["sources"][1]["reference"] == "Rule 22"
    assert res["timings_ms"]["rate_lookup"] >= 0


def test_system_prompt_rate_response_format_guidelines():
    """Verify SYSTEM_PROMPT contains the required natural opening and structured details instructions."""
    from src.generators.answer_generator import SYSTEM_PROMPT

    assert "SIMPLE DIRECT ANSWER" in SYSTEM_PROMPT
    assert "FULL RATE / SOURCE DETAILS" in SYSTEM_PROMPT
    assert "exempt from GST, so no GST is charged (0%)" in SYSTEM_PROMPT
    assert "attracts [Total GST]% GST under HSN [Code]" in SYSTEM_PROMPT
    assert "NEVER start with database labels like \"Item Description:\"" in SYSTEM_PROMPT
    assert "Description: [Must ALWAYS show the full retrieved legal description verbatim" in SYSTEM_PROMPT
    assert "Keep compensation cess distinct from base GST" in SYSTEM_PROMPT


def test_direct_full_prompt_keeps_legal_grounding_boundary():
    prompt = build_direct_full_prompt("What is 5% of 100000?")
    assert "Answer directly using only the numbers" in prompt
    assert "Do not look up, invent, estimate, or assume any GST law" in prompt
    assert "RETRIEVED CONTEXT" not in prompt
