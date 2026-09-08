"""Structure-aware chunks for parsed GST Act sections."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


DEFAULT_SECTIONS_JSON = Path("data/acts/central_gst_act_2017_sections.json")
DEFAULT_CHUNKS_JSON = Path("data/acts/central_gst_act_2017_chunks.json")
DEFAULT_MAX_TOKENS = 450
DEFAULT_OVERLAP_TOKENS = 60


def load_sections(path=DEFAULT_SECTIONS_JSON):
    """Load parsed section dictionaries from JSON."""
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Expected section JSON root to be a list")
    return data


def count_tokens(tokenizer, text, words_per_batch=250):
    """Count tokens with a Hugging Face-style tokenizer."""
    words = (text or "").split()
    total = 0
    for start in range(0, len(words), words_per_batch):
        total += len(tokenizer.tokenize(" ".join(words[start:start + words_per_batch])))
    return total


def parent_context(section):
    """Return compact context prepended to child chunks."""
    number = section.get("section_number", "")
    title = section.get("section_title", "")
    return f"Section {number}. {title}".strip()


def chunk_metadata(section, subsection_numbers, strategy):
    """Build chunk metadata without dropping section-level provenance."""
    return {
        "section_number": section.get("section_number"),
        "section_title": section.get("section_title"),
        "subsection_numbers": subsection_numbers,
        "status": section.get("status"),
        "chapter": section.get("chapter"),
        "strategy": strategy,
    }


def make_chunk(section, text, subsection_numbers, strategy, tokenizer):
    """Create one chunk dictionary with parent context and token count."""
    context = parent_context(section)
    chunk_text = f"{context}\n\n{text.strip()}" if context else text.strip()
    return {
        "chunk_id": None,
        "text": chunk_text,
        "token_count": count_tokens(tokenizer, chunk_text),
        "metadata": chunk_metadata(section, subsection_numbers, strategy),
    }


def split_large_text(text, max_tokens, overlap_tokens, tokenizer):
    """Fallback split for oversized subsection text, using token-aware word windows."""
    words = text.split()
    if not words:
        return []

    parts = []
    current = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and count_tokens(tokenizer, candidate) > max_tokens:
            parts.append(" ".join(current))
            if overlap_tokens > 0:
                overlap = []
                for previous_word in reversed(current):
                    candidate_overlap = " ".join([previous_word, *overlap])
                    if overlap and count_tokens(tokenizer, candidate_overlap) > overlap_tokens:
                        break
                    overlap.insert(0, previous_word)
                current = overlap
            else:
                current = []
        current.append(word)

    if current:
        parts.append(" ".join(current))
    return parts


def emit_grouped_subsections(section, group, chunks, max_tokens, overlap_tokens, tokenizer):
    """Append chunks for one buffered group of complete subsections."""
    if not group:
        return

    if len(group) == 1:
        subsection = group[0]
        subsection_number = subsection.get("subsection_number")
        subsection_text = subsection.get("text", "")
        standalone = make_chunk(
            section,
            subsection_text,
            [subsection_number] if subsection_number else [],
            "subsection",
            tokenizer,
        )
        if standalone["token_count"] <= max_tokens:
            chunks.append(standalone)
            return

        context_tokens = count_tokens(tokenizer, parent_context(section))
        body_limit = max(1, max_tokens - context_tokens)
        for part_index, part in enumerate(split_large_text(subsection_text, body_limit, overlap_tokens, tokenizer), 1):
            chunk = make_chunk(
                section,
                part,
                [subsection_number] if subsection_number else [],
                "subsection_token_split",
                tokenizer,
            )
            chunk["metadata"]["split_index"] = part_index
            chunks.append(chunk)
        return

    text = "\n\n".join(subsection.get("text", "").strip() for subsection in group if subsection.get("text"))
    subsection_numbers = [subsection.get("subsection_number") for subsection in group if subsection.get("subsection_number")]
    chunks.append(make_chunk(section, text, subsection_numbers, "subsection_group", tokenizer))


def chunk_section(section, max_tokens, overlap_tokens, tokenizer):
    """Chunk one parsed Section without crossing Section boundaries."""
    content = section.get("content", "")
    whole = make_chunk(section, content, [], "section", tokenizer)
    if whole["token_count"] <= max_tokens:
        return [whole]

    subsections = section.get("subsections") or []
    if not subsections:
        body_limit = max(1, max_tokens - count_tokens(tokenizer, parent_context(section)))
        chunks = []
        for part_index, part in enumerate(split_large_text(content, body_limit, overlap_tokens, tokenizer), 1):
            chunk = make_chunk(section, part, [], "section_token_split", tokenizer)
            chunk["metadata"]["split_index"] = part_index
            chunks.append(chunk)
        return chunks

    chunks = []
    group = []
    for subsection in subsections:
        candidate = [*group, subsection]
        candidate_text = "\n\n".join(item.get("text", "").strip() for item in candidate if item.get("text"))
        candidate_chunk = make_chunk(
            section,
            candidate_text,
            [item.get("subsection_number") for item in candidate if item.get("subsection_number")],
            "subsection_group",
            tokenizer,
        )
        if group and candidate_chunk["token_count"] > max_tokens:
            emit_grouped_subsections(section, group, chunks, max_tokens, overlap_tokens, tokenizer)
            group = [subsection]
        else:
            group = candidate
    emit_grouped_subsections(section, group, chunks, max_tokens, overlap_tokens, tokenizer)
    return chunks


def chunk_sections(sections, max_tokens=DEFAULT_MAX_TOKENS, overlap_tokens=DEFAULT_OVERLAP_TOKENS, tokenizer=None):
    """Return structure-aware chunks for parsed GST Act sections."""
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be non-negative")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    chunks = []
    for section in sections:
        chunks.extend(chunk_section(section, max_tokens, overlap_tokens, tokenizer))

    for index, chunk in enumerate(chunks, 1):
        section_number = chunk["metadata"].get("section_number")
        chunk["chunk_id"] = f"cgst-{section_number}-{index:04d}"
    return chunks


def chunk_size_report(chunks):
    """Return min/avg/max token sizes and chunk counts by section."""
    counts = [chunk["token_count"] for chunk in chunks]
    by_section = {}
    for chunk in chunks:
        section_number = chunk["metadata"].get("section_number")
        by_section[section_number] = by_section.get(section_number, 0) + 1
    most_chunks = sorted(by_section.items(), key=lambda item: (-item[1], str(item[0])))[:10]
    return {
        "chunk_count": len(chunks),
        "min_tokens": min(counts) if counts else 0,
        "avg_tokens": round(mean(counts), 2) if counts else 0,
        "max_tokens": max(counts) if counts else 0,
        "sections_with_most_chunks": most_chunks,
    }
