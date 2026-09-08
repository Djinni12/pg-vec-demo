"""Structure-aware chunks for parsed CGST Rules."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from .act_chunker import count_tokens, split_large_text


DEFAULT_RULES_JSON = Path("data/rules/gst_rules_rules.json")
DEFAULT_RULE_CHUNKS_JSON = Path("data/rules/gst_rules_chunks.json")
DEFAULT_MAX_TOKENS = 450
DEFAULT_OVERLAP_TOKENS = 60


def load_rules(path=DEFAULT_RULES_JSON):
    """Load parsed rule dictionaries from JSON."""
    with Path(path).open(encoding="utf-8") as handle:
        rules = json.load(handle)
    if not isinstance(rules, list):
        raise ValueError("Expected rules JSON root to be a list")
    return rules


def parent_context(rule):
    """Return compact context prepended to rule chunks."""
    number = rule.get("rule_number", "")
    title = rule.get("rule_title", "")
    chapter = rule.get("chapter")
    chapter_title = rule.get("chapter_title")
    parts = []
    if chapter or chapter_title:
        parts.append(" ".join(part for part in (chapter, chapter_title) if part))
    parts.append(f"Rule {number}. {title}".strip())
    return "\n".join(parts)


def chunk_metadata(rule, subrule_numbers, strategy):
    """Build rule chunk metadata."""
    return {
        "rule_number": rule.get("rule_number"),
        "rule_title": rule.get("rule_title"),
        "chapter": rule.get("chapter"),
        "chapter_title": rule.get("chapter_title"),
        "subrule_numbers": subrule_numbers,
        "status": rule.get("status"),
        "chunk_strategy": strategy,
    }


def make_chunk(rule, text, subrule_numbers, strategy, tokenizer):
    """Create one rule chunk with parent context and token count."""
    context = parent_context(rule)
    chunk_text = f"{context}\n\n{text.strip()}" if context else text.strip()
    return {
        "chunk_id": None,
        "text": chunk_text,
        "token_count": count_tokens(tokenizer, chunk_text),
        "metadata": chunk_metadata(rule, subrule_numbers, strategy),
    }


def emit_grouped_subrules(rule, group, chunks, max_tokens, overlap_tokens, tokenizer):
    """Append chunks for a buffered group of complete sub-rules."""
    if not group:
        return

    if len(group) == 1:
        subrule = group[0]
        number = subrule.get("subrule_number")
        text = subrule.get("text", "")
        standalone = make_chunk(rule, text, [number] if number else [], "subrule", tokenizer)
        if standalone["token_count"] <= max_tokens:
            chunks.append(standalone)
            return

        body_limit = max(1, max_tokens - count_tokens(tokenizer, parent_context(rule)))
        for split_index, part in enumerate(split_large_text(text, body_limit, overlap_tokens, tokenizer), 1):
            chunk = make_chunk(rule, part, [number] if number else [], "subrule_token_split", tokenizer)
            chunk["metadata"]["split_index"] = split_index
            chunks.append(chunk)
        return

    text = "\n\n".join(subrule.get("text", "").strip() for subrule in group if subrule.get("text"))
    numbers = [subrule.get("subrule_number") for subrule in group if subrule.get("subrule_number")]
    chunks.append(make_chunk(rule, text, numbers, "subrule_group", tokenizer))


def chunk_rule(rule, max_tokens, overlap_tokens, tokenizer):
    """Chunk one parsed Rule without crossing Rule boundaries."""
    content = rule.get("content", "")
    whole = make_chunk(rule, content, [], "rule", tokenizer)
    if whole["token_count"] <= max_tokens:
        return [whole]

    subrules = rule.get("subrules") or []
    if not subrules:
        body_limit = max(1, max_tokens - count_tokens(tokenizer, parent_context(rule)))
        chunks = []
        for split_index, part in enumerate(split_large_text(content, body_limit, overlap_tokens, tokenizer), 1):
            chunk = make_chunk(rule, part, [], "rule_token_split", tokenizer)
            chunk["metadata"]["split_index"] = split_index
            chunks.append(chunk)
        return chunks

    chunks = []
    group = []
    for subrule in subrules:
        candidate = [*group, subrule]
        candidate_text = "\n\n".join(item.get("text", "").strip() for item in candidate if item.get("text"))
        candidate_numbers = [item.get("subrule_number") for item in candidate if item.get("subrule_number")]
        candidate_chunk = make_chunk(rule, candidate_text, candidate_numbers, "subrule_group", tokenizer)
        if group and candidate_chunk["token_count"] > max_tokens:
            emit_grouped_subrules(rule, group, chunks, max_tokens, overlap_tokens, tokenizer)
            group = [subrule]
        else:
            group = candidate
    emit_grouped_subrules(rule, group, chunks, max_tokens, overlap_tokens, tokenizer)
    return chunks


def chunk_rules(rules, max_tokens=DEFAULT_MAX_TOKENS, overlap_tokens=DEFAULT_OVERLAP_TOKENS, tokenizer=None):
    """Return structure-aware chunks for parsed CGST Rules."""
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
    for rule in rules:
        chunks.extend(chunk_rule(rule, max_tokens, overlap_tokens, tokenizer))

    for index, chunk in enumerate(chunks, 1):
        rule_number = chunk["metadata"].get("rule_number")
        chunk["chunk_id"] = f"cgst-rules-{rule_number}-{index:04d}"
    return chunks


def chunk_size_report(chunks):
    """Return min/avg/max token sizes and chunk counts by rule."""
    counts = [chunk["token_count"] for chunk in chunks]
    by_rule = {}
    for chunk in chunks:
        rule_number = chunk["metadata"].get("rule_number")
        by_rule[rule_number] = by_rule.get(rule_number, 0) + 1
    most_chunks = sorted(by_rule.items(), key=lambda item: (-item[1], str(item[0])))[:10]
    return {
        "chunk_count": len(chunks),
        "min_tokens": min(counts) if counts else 0,
        "avg_tokens": round(mean(counts), 2) if counts else 0,
        "max_tokens": max(counts) if counts else 0,
        "rules_with_most_chunks": most_chunks,
    }
