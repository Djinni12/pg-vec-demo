"""LLM answer generation module for the GST legal RAG pipeline."""

from __future__ import annotations

import os
import re
import time
from typing import Any

from dotenv import load_dotenv
import openai
from openai import OpenAI

from src.retrieval_inspector import DEFAULT_TOP_K, LoadedModels, inspect_retrieval
from src.retrievers.rate_retriever import retrieve_rates
from src.routers.query_router import RouteType, classify_query

# Ensure environment variables from .env are loaded
load_dotenv(override=True)

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a GST legal and tax information assistant.

Answer using ONLY the retrieved context.

Your answers may be read by taxpayers, accountants, tax professionals, lawyers, and other users.

LANGUAGE AND STYLE:
- Use clear, simple, professional English.
- Make the answer easy to understand without removing important legal details.
- Avoid unnecessarily complex legal wording.
- When a legal or technical term is necessary, use it but explain it in simple words.
- Do not replace precise legal terms when doing so would change their meaning.
- Keep relevant Sections, Rules, Forms, deadlines, conditions, exceptions, and requirements.
- Explain the practical meaning of a legal provision instead of merely repeating its wording.
- Prefer short sentences and well-organized headings.
- Use bullets or numbered steps when explaining a procedure.
- Start with a direct answer, then provide the supporting legal details.
- When answering tax rate questions, start with a direct, user-friendly statement of the rate and HSN/SAC code, followed by the CGST/SGST/IGST breakdown and any applicable cess or conditions.
- Do not make the answer less detailed merely to make it easier to read.

GROUNDING:
- Use only information supported by the retrieved context.
- Do not invent legal requirements, Sections, Rules, Forms, dates, rates, or procedures.
- Final GST rates MUST be quoted strictly from the provided structured rate records. Never invent, estimate, or assume a tax rate.
- If no rate record matches the requested product or service, clearly state that the rate is not found in the available rate database.
- Clearly distinguish related concepts such as cancellation, suspension, and revocation of cancellation.
- Include only information relevant to the user's question.
- If the context is insufficient, clearly say so.
- Do not mention retrieval, embeddings, BM25, RRF, reranking, chunks, or internal system details.

REFERENCES:
- Mention relevant Sections, Rules, Forms, HSN/SAC codes, and notification references where supported.
- Use legal references to support the explanation, not as a substitute for explaining it."""


class GenerationError(Exception):
    """Raised when an error occurs during answer generation."""


def get_configured_model(model: str | None = None) -> str:
    """Return the configured OpenAI model name."""
    if model and model.strip():
        return model.strip()
    env_model = os.environ.get("OPENAI_MODEL")
    if env_model and env_model.strip():
        return env_model.strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    if "generativelanguage" in base_url:
        return "models/gemini-3.5-flash-lite"
    return DEFAULT_OPENAI_MODEL


def get_openai_client(
    api_key: str | None = None,
    base_url: str | None = None,
) -> OpenAI:
    """Create and return an OpenAI client using environment variables or explicit parameters."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is not set. Please configure it in your environment or .env file.")

    url = base_url or os.environ.get("OPENAI_BASE_URL")
    kwargs: dict[str, Any] = {"api_key": key}
    if url:
        kwargs["base_url"] = url
    return OpenAI(**kwargs)


def format_chunk(chunk: dict[str, Any], index: int) -> str:
    """Format a single retrieved context chunk with required source metadata."""
    doc_type = chunk.get("document_type") or "unknown"
    ref = chunk.get("reference") or "No reference"
    title = chunk.get("title") or "Untitled"
    chunk_id = chunk.get("chunk_id") or f"chunk_{index}"
    content = chunk.get("content") or chunk.get("snippet") or ""

    return (
        f"--- Source {index} ---\n"
        f"Document Type: {doc_type}\n"
        f"Reference: {ref}\n"
        f"Title: {title}\n"
        f"Chunk ID: {chunk_id}\n"
        f"Content:\n"
        f"{content.strip()}"
    )


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Format final reranked chunks into the retrieved context block."""
    if not chunks:
        return "No retrieved context available."
    return "\n\n".join(format_chunk(chunk, i) for i, chunk in enumerate(chunks, 1))


def format_rate_item(item: dict[str, Any], index: int) -> str:
    """Format a single structured GST rate result item with all key tariff fields."""
    item_type = (item.get("item_type") or "Goods").capitalize()
    code = item.get("code") or "N/A"
    desc = item.get("description") or "No description"
    rate_str = item.get("formatted_rate") or "Rate not specified"
    cgst = f"{item['cgst_rate_pct']:g}%" if item.get("cgst_rate_pct") is not None else "N/A"
    sgst = f"{item['sgst_utgst_rate_pct']:g}%" if item.get("sgst_utgst_rate_pct") is not None else "N/A"
    igst = f"{item['igst_rate_pct']:g}%" if item.get("igst_rate_pct") is not None else "N/A"
    cess = item.get("compensation_cess") or "None"
    condition = item.get("condition") or "None"
    effective_date = item.get("effective_date") or "Not specified"
    schedule = item.get("schedule") or "Not specified"
    notif = item.get("notification_number") or ""
    source_ref = item.get("source_reference") or "GST rates2025.pdf"

    return (
        f"--- Rate Item {index} ({item_type}) ---\n"
        f"Tariff / HSN Code: {code}\n"
        f"Description: {desc}\n"
        f"GST Rate: {rate_str}\n"
        f"CGST: {cgst} | SGST/UTGST: {sgst} | IGST: {igst}\n"
        f"Schedule: {schedule}\n"
        f"Notification: {notif}\n"
        f"Compensation Cess: {cess}\n"
        f"Condition: {condition}\n"
        f"Effective Date: {effective_date}\n"
        f"Source Reference: {source_ref}"
    )


def format_rate_context(rates: list[dict[str, Any]]) -> str:
    """Format list of structured rate items into prompt context block."""
    if not rates:
        return "No matching rate entries found in the structured tariff database."
    return "\n\n".join(format_rate_item(r, i) for i, r in enumerate(rates, 1))


def build_user_prompt(
    query: str,
    context: str | None = None,
    *,
    legal_context: str | None = None,
    rate_context: str | None = None,
) -> str:
    """Build user prompt containing retrieved legal and/or rate context and question."""
    sections: list[str] = []
    if rate_context and rate_context.strip():
        sections.append(f"STRUCTURED GST RATE RECORDS\n\n{rate_context.strip()}")
    if legal_context and legal_context.strip():
        sections.append(f"RETRIEVED LEGAL CONTEXT\n\n{legal_context.strip()}")
    if not sections and context and context.strip():
        sections.append(context.strip())

    combined_context = "\n\n".join(sections) if sections else "No retrieved context available."
    return (
        f"RETRIEVED CONTEXT\n\n"
        f"{combined_context}\n\n"
        f"USER QUESTION\n\n"
        f"{query}\n\n"
        f"ANSWER"
    )


def build_full_prompt(
    query: str,
    chunks: list[dict[str, Any]] | None = None,
    rate_results: list[dict[str, Any]] | None = None,
    context: str | None = None,
) -> str:
    """Build the complete grounded prompt string."""
    legal_ctx = format_context(chunks) if chunks is not None else None
    rate_ctx = format_rate_context(rate_results) if rate_results is not None else None
    user_prompt = build_user_prompt(
        query,
        context=context,
        legal_context=legal_ctx,
        rate_context=rate_ctx,
    )
    return (
        f"SYSTEM PROMPT\n\n"
        f"{SYSTEM_PROMPT}\n\n"
        f"{user_prompt}"
    )


def strip_reasoning(text: str) -> str:
    """Strip chain-of-thought or internal reasoning tags to prevent leaking thought tokens."""
    if not text:
        return ""
    # Strip closed thought/thinking tags
    cleaned = re.sub(r"<(thought|thinking)>[\s\S]*?</\1>", "", text, flags=re.IGNORECASE)
    # Strip any unclosed thought/thinking tag that starts the response
    cleaned = re.sub(r"^<(thought|thinking)>[\s\S]*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def extract_sources(chunks: list[dict[str, Any]], start_rank: int = 1) -> list[dict[str, Any]]:
    """Extract source metadata items from final reranked chunks."""
    sources: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start_rank):
        content = chunk.get("content") or chunk.get("snippet") or ""
        sources.append(
            {
                "rank": chunk.get("rank", index),
                "document_type": chunk.get("document_type"),
                "reference": chunk.get("reference"),
                "title": chunk.get("title"),
                "chunk_id": chunk.get("chunk_id"),
                "content": content,
                "snippet": chunk.get("snippet") or _snippet(content),
                "reranker_score": chunk.get("reranker_score"),
            }
        )
    return sources


def extract_rate_sources(
    rate_results: list[dict[str, Any]], start_rank: int = 1
) -> list[dict[str, Any]]:
    """Extract source metadata items from structured rate results for unified citation display."""
    sources: list[dict[str, Any]] = []
    for index, item in enumerate(rate_results, start_rank):
        code = item.get("code") or "N/A"
        item_type = item.get("item_type") or "rate"
        prefix = "HSN" if item_type == "goods" else "SAC"
        desc = item.get("description") or ""
        rate_str = item.get("formatted_rate") or "Rate not specified"
        cess = f"\nCompensation Cess: {item['compensation_cess']}" if item.get("compensation_cess") else ""
        cond = f"\nCondition: {item['condition']}" if item.get("condition") else ""
        eff = f"\nEffective Date: {item['effective_date']}" if item.get("effective_date") else ""
        source_ref = item.get("source_reference") or "GST rates2025.pdf"

        content = (
            f"Tariff Code: {prefix} {code}\n"
            f"Applicable Rate: {rate_str}"
            f"{cess}"
            f"{cond}"
            f"{eff}\n"
            f"Description: {desc}\n"
            f"Source Reference: {source_ref}"
        )
        sources.append(
            {
                "rank": index,
                "document_type": item_type,
                "reference": f"{prefix} {code}",
                "title": _snippet(desc, 80),
                "chunk_id": f"rate_{item_type}_{code}_{index}",
                "content": content,
                "snippet": f"{rate_str} · {_snippet(desc, 120)}",
                "reranker_score": item.get("score"),
            }
        )
    return sources


def extract_combined_sources(
    chunks: list[dict[str, Any]] | None = None,
    rate_results: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Extract and combine sources from structured rates and legal chunks."""
    sources: list[dict[str, Any]] = []
    if rate_results:
        sources.extend(extract_rate_sources(rate_results, start_rank=1))
    if chunks:
        sources.extend(extract_sources(chunks, start_rank=len(sources) + 1))
    return sources


def _snippet(content: str, max_length: int = 280) -> str:
    """Generate a readable snippet from content text."""
    if len(content) <= max_length:
        return content
    truncate_at = content.rfind(" ", 0, max_length)
    if truncate_at == -1:
        truncate_at = max_length
    return content[:truncate_at] + "..."


def generate_answer(
    query: str,
    chunks: list[dict[str, Any]] | None = None,
    *,
    rate_results: list[dict[str, Any]] | None = None,
    model: str | None = None,
    client: OpenAI | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1500,
) -> dict[str, Any]:
    """Send grounded context to OpenAI and return the clean answer with timings and sources."""
    if not query.strip():
        raise ValueError("query cannot be empty")

    model_name = get_configured_model(model)
    client = client or get_openai_client()

    legal_ctx = format_context(chunks or []) if chunks else ""
    rate_ctx = format_rate_context(rate_results or []) if rate_results else ""
    user_prompt = build_user_prompt(
        query,
        legal_context=legal_ctx,
        rate_context=rate_ctx,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    gen_start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except openai.AuthenticationError as exc:
        raise GenerationError(f"OpenAI authentication failed: {exc}") from exc
    except openai.RateLimitError as exc:
        raise GenerationError(f"OpenAI rate limit exceeded: {exc}") from exc
    except openai.APIConnectionError as exc:
        raise GenerationError(f"OpenAI connection error: {exc}") from exc
    except openai.OpenAIError as exc:
        raise GenerationError(f"OpenAI API error: {exc}") from exc
    except Exception as exc:
        raise GenerationError(f"Generation failed: {exc}") from exc

    gen_ms = round((time.perf_counter() - gen_start) * 1000, 3)

    choice = response.choices[0] if response.choices else None
    raw_content = choice.message.content if choice and choice.message else ""
    answer = strip_reasoning(raw_content or "")

    sources = extract_combined_sources(chunks, rate_results)

    return {
        "answer": answer,
        "sources": sources,
        "rate_results": rate_results or [],
        "generation_timing": gen_ms,
        "model_used": model_name,
    }


DEFAULT_ANSWER_TOP_K = 5


def run_gst_answer_flow(
    query: str,
    top_k: int = DEFAULT_ANSWER_TOP_K,
    *,
    models: LoadedModels,
    openai_model: str | None = None,
    openai_client: OpenAI | None = None,
    db_url: str | None = None,
    route_override: str | None = None,
) -> dict[str, Any]:
    """Execute routed retrieval pipeline and generate a grounded answer from rate results and/or reranked chunks."""
    route = RouteType(route_override) if route_override else classify_query(query)

    rate_results: list[dict[str, Any]] = []
    final_chunks: list[dict[str, Any]] = []
    rate_timing = 0.0
    retrieval_data: dict[str, Any] = {
        "results": [],
        "dense_results": [],
        "bm25_results": [],
        "hybrid_results": [],
        "metadata": {},
        "models": {},
        "config": {},
        "timings_ms": {"total": 0.0, "dense": 0.0, "bm25": 0.0, "rrf": 0.0, "reranker": 0.0},
    }

    if route in (RouteType.RATE, RouteType.MIXED):
        rate_start = time.perf_counter()
        rate_results = retrieve_rates(query, db_url=db_url, limit=top_k)
        rate_timing = round((time.perf_counter() - rate_start) * 1000, 3)

    if route in (RouteType.LEGAL, RouteType.MIXED):
        retrieval_data = inspect_retrieval(query, top_k=top_k, models=models, db_url=db_url)
        final_chunks = retrieval_data.get("results") or []

    legal_timing = retrieval_data.get("timings_ms", {}).get("total", 0.0)
    retrieval_timing = round(rate_timing + legal_timing, 3)

    gen_data = generate_answer(
        query=query,
        chunks=final_chunks,
        rate_results=rate_results,
        model=openai_model,
        client=openai_client,
    )

    generation_timing = gen_data["generation_timing"]
    total_timing = round(retrieval_timing + generation_timing, 3)

    return {
        "query": query,
        "route": route.value,
        "answer": gen_data["answer"],
        "rate_results": rate_results,
        "sources": gen_data["sources"],
        "sources_used": gen_data["sources"],
        "retrieval_timing": retrieval_timing,
        "generation_timing": generation_timing,
        "total_timing": total_timing,
        "model_used": gen_data["model_used"],
        "model": gen_data["model_used"],
        "timings_ms": {
            "retrieval": retrieval_timing,
            "generation": generation_timing,
            "total": total_timing,
            "rate_lookup": rate_timing,
            "dense": retrieval_data["timings_ms"].get("dense", 0.0),
            "bm25": retrieval_data["timings_ms"].get("bm25", 0.0),
            "rrf": retrieval_data["timings_ms"].get("rrf", 0.0),
            "reranker": retrieval_data["timings_ms"].get("reranker", 0.0),
        },
        "retrieval_debug": {
            "results": retrieval_data.get("results", []),
            "dense_results": retrieval_data.get("dense_results", []),
            "bm25_results": retrieval_data.get("bm25_results", []),
            "hybrid_results": retrieval_data.get("hybrid_results", []),
            "metadata": retrieval_data.get("metadata", {}),
            "models": retrieval_data.get("models", {}),
            "config": retrieval_data.get("config", {}),
        },
    }


class StreamingThoughtFilter:
    """Filters out <thought>...</thought> or <thinking>...</thinking> during live token streaming."""

    def __init__(self):
        self.in_thought = False
        self.buffer = ""

    def process(self, delta: str) -> str:
        if not delta:
            return ""
        self.buffer += delta
        output: list[str] = []

        while self.buffer:
            if self.in_thought:
                lower = self.buffer.lower()
                m_end = re.search(r"</(thought|thinking)>", lower)
                if m_end:
                    self.buffer = self.buffer[m_end.end():]
                    self.in_thought = False
                else:
                    m_partial = re.search(r"</[a-zA-Z]*$", lower)
                    if m_partial:
                        self.buffer = self.buffer[m_partial.start():]
                    else:
                        self.buffer = ""
                    break
            else:
                lower = self.buffer.lower()
                m_start = re.search(r"<(thought|thinking)>", lower)
                if m_start:
                    output.append(self.buffer[:m_start.start()])
                    self.buffer = self.buffer[m_start.end():]
                    self.in_thought = True
                else:
                    m_partial = re.search(r"<[a-zA-Z]*$", self.buffer)
                    if m_partial:
                        safe_idx = m_partial.start()
                        output.append(self.buffer[:safe_idx])
                        self.buffer = self.buffer[safe_idx:]
                        break
                    else:
                        output.append(self.buffer)
                        self.buffer = ""
                        break

        return "".join(output)

    def flush(self) -> str:
        if self.in_thought:
            return ""
        output = self.buffer
        self.buffer = ""
        return output


def stream_answer(
    query: str,
    chunks: list[dict[str, Any]] | None = None,
    *,
    rate_results: list[dict[str, Any]] | None = None,
    model: str | None = None,
    client: OpenAI | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1500,
):
    """Stream response tokens from OpenAI, yielding (event_type, delta)."""
    if not query.strip():
        raise ValueError("query cannot be empty")

    model_name = get_configured_model(model)
    client = client or get_openai_client()

    legal_ctx = format_context(chunks or []) if chunks else ""
    rate_ctx = format_rate_context(rate_results or []) if rate_results else ""
    user_prompt = build_user_prompt(
        query,
        legal_context=legal_ctx,
        rate_context=rate_ctx,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
    except openai.AuthenticationError as exc:
        raise GenerationError(f"OpenAI authentication failed: {exc}") from exc
    except openai.RateLimitError as exc:
        raise GenerationError(f"OpenAI rate limit exceeded: {exc}") from exc
    except openai.APIConnectionError as exc:
        raise GenerationError(f"OpenAI connection error: {exc}") from exc
    except openai.OpenAIError as exc:
        raise GenerationError(f"OpenAI API error: {exc}") from exc
    except Exception as exc:
        raise GenerationError(f"Generation failed: {exc}") from exc

    thought_filter = StreamingThoughtFilter()
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            raw_delta = chunk.choices[0].delta.content
            clean_delta = thought_filter.process(raw_delta)
            if clean_delta:
                yield ("token", clean_delta)

    trailing = thought_filter.flush()
    if trailing:
        yield ("token", trailing)


def stream_gst_answer_flow(
    query: str,
    top_k: int = DEFAULT_ANSWER_TOP_K,
    *,
    models: LoadedModels,
    openai_model: str | None = None,
    openai_client: OpenAI | None = None,
    db_url: str | None = None,
    route_override: str | None = None,
):
    """Stream retrieval and generation events for real-time live answers."""
    route = RouteType(route_override) if route_override else classify_query(query)

    rate_results: list[dict[str, Any]] = []
    final_chunks: list[dict[str, Any]] = []
    rate_timing = 0.0
    retrieval_data: dict[str, Any] = {
        "results": [],
        "dense_results": [],
        "bm25_results": [],
        "hybrid_results": [],
        "metadata": {},
        "models": {},
        "config": {},
        "timings_ms": {"total": 0.0, "dense": 0.0, "bm25": 0.0, "rrf": 0.0, "reranker": 0.0},
    }

    if route in (RouteType.RATE, RouteType.MIXED):
        rate_start = time.perf_counter()
        rate_results = retrieve_rates(query, db_url=db_url, limit=top_k)
        rate_timing = round((time.perf_counter() - rate_start) * 1000, 3)

    if route in (RouteType.LEGAL, RouteType.MIXED):
        retrieval_data = inspect_retrieval(query, top_k=top_k, models=models, db_url=db_url)
        final_chunks = retrieval_data.get("results") or []

    legal_timing = retrieval_data.get("timings_ms", {}).get("total", 0.0)
    retrieval_timing = round(rate_timing + legal_timing, 3)

    sources = extract_combined_sources(final_chunks, rate_results)
    model_name = get_configured_model(openai_model)

    yield {
        "type": "meta",
        "query": query,
        "route": route.value,
        "rate_results": rate_results,
        "sources": sources,
        "sources_used": sources,
        "retrieval_timing": retrieval_timing,
        "model_used": model_name,
        "model": model_name,
        "retrieval_debug": {
            "results": retrieval_data.get("results", []),
            "dense_results": retrieval_data.get("dense_results", []),
            "bm25_results": retrieval_data.get("bm25_results", []),
            "hybrid_results": retrieval_data.get("hybrid_results", []),
            "metadata": retrieval_data.get("metadata", {}),
            "models": retrieval_data.get("models", {}),
            "config": retrieval_data.get("config", {}),
        },
    }

    # 2. Generation streaming
    gen_start = time.perf_counter()
    full_answer_parts: list[str] = []

    for _, delta in stream_answer(
        query=query,
        chunks=final_chunks,
        rate_results=rate_results,
        model=openai_model,
        client=openai_client,
    ):
        full_answer_parts.append(delta)
        yield {
            "type": "token",
            "delta": delta,
        }

    gen_ms = round((time.perf_counter() - gen_start) * 1000, 3)
    total_ms = round(retrieval_timing + gen_ms, 3)
    full_answer = "".join(full_answer_parts).strip()

    yield {
        "type": "done",
        "answer": full_answer,
        "route": route.value,
        "rate_results": rate_results,
        "generation_timing": gen_ms,
        "total_timing": total_ms,
        "timings_ms": {
            "retrieval": retrieval_timing,
            "generation": gen_ms,
            "total": total_ms,
            "rate_lookup": rate_timing,
            "dense": retrieval_data["timings_ms"].get("dense", 0.0),
            "bm25": retrieval_data["timings_ms"].get("bm25", 0.0),
            "rrf": retrieval_data["timings_ms"].get("rrf", 0.0),
            "reranker": retrieval_data["timings_ms"].get("reranker", 0.0),
        },
    }
