"""Structure-aware chunks for parsed GST Forms."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer

from act_chunker import count_tokens, split_large_text

DEFAULT_FORMS_JSON = Path("data/form/gst_forms_forms.json")
DEFAULT_FORM_CHUNKS_JSON = Path("data/form/gst_forms_chunks.json")
DEFAULT_MAX_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 60
TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_forms(path=DEFAULT_FORMS_JSON):
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Expected forms JSON root to be a list")
    return data


def form_context(form):
    number = form.get("form_number", "")
    title = form.get("form_title", "")
    return f"{number}. {title}".strip()


def chunk_metadata(form, strategy):
    return {
        "form_uid": form.get("form_uid"),
        "form_number": form.get("form_number"),
        "form_family": form.get("form_family"),
        "form_code": form.get("form_code"),
        "form_title": form.get("form_title"),
        "title": form.get("form_title"),
        "language": form.get("language"),
        "rule_references": form.get("rule_references") or [],
        "part_number": form.get("part_number"),
        "section_label": None,
        "page_start": form.get("page_start"),
        "page_end": form.get("page_end"),
        "source_start_page": form.get("source_start_page", form.get("page_start")),
        "source_end_page": form.get("source_end_page", form.get("page_end")),
        "chunk_strategy": strategy,
    }


def make_chunk(form, text, strategy, tokenizer):
    context = form_context(form)
    chunk_text = f"{context}\n\n{text.strip()}" if context else text.strip()
    return {
        "chunk_id": None,
        "text": chunk_text,
        "token_count": count_tokens(tokenizer, chunk_text),
        "metadata": chunk_metadata(form, strategy),
    }


LOGICAL_BLOCK_STARTS = (
    "भाग", "बाग", "Part", "अनुदेश", "अनुदेर", "Instruction", "कथन", "उपाबंध", "पाबंध",
    "घोषणा", "सत्यापन", "सत्माऩन", "सत्यापन", "9टQपण", "टिप्पण", "9टप्पण",
)


def is_logical_block_start(line):
    stripped = line.strip()
    if not stripped:
        return False
    if any(stripped.startswith(prefix) for prefix in LOGICAL_BLOCK_STARTS):
        return True
    if __import__("re").match(r"^\d{1,2}\s*[.)।:-]", stripped):
        return True
    return False


def split_logical_blocks(content):
    blocks = []
    current = []
    for line in content.splitlines():
        if current and is_logical_block_start(line):
            blocks.append("\n".join(current).strip())
            current = []
        current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return [block for block in blocks if block]


def emit_token_split(form, text, strategy, max_tokens, overlap_tokens, tokenizer):
    context_tokens = count_tokens(tokenizer, form_context(form))
    body_limit = max(1, max_tokens - context_tokens)
    chunks = []
    for part_index, part in enumerate(split_large_text(text, body_limit, overlap_tokens, tokenizer), 1):
        chunk = make_chunk(form, part, strategy, tokenizer)
        chunk["metadata"]["split_index"] = part_index
        chunks.append(chunk)
    return chunks


def chunk_form(form, max_tokens, overlap_tokens, tokenizer):
    content = (form.get("content") or "").strip()
    if not content:
        return []
    whole = make_chunk(form, content, "form", tokenizer)
    if whole["token_count"] <= max_tokens:
        return [whole]

    chunks = []
    group = []
    for block in split_logical_blocks(content):
        block_chunk = make_chunk(form, block, "form_logical_block", tokenizer)
        if block_chunk["token_count"] > max_tokens:
            if group:
                chunks.append(make_chunk(form, "\n\n".join(group), "form_logical_group", tokenizer))
                group = []
            chunks.extend(emit_token_split(form, block, "form_token_split", max_tokens, overlap_tokens, tokenizer))
            continue

        candidate = "\n\n".join([*group, block])
        if group and make_chunk(form, candidate, "form_logical_group", tokenizer)["token_count"] > max_tokens:
            chunks.append(make_chunk(form, "\n\n".join(group), "form_logical_group", tokenizer))
            group = [block]
        else:
            group.append(block)

    if group:
        chunks.append(make_chunk(form, "\n\n".join(group), "form_logical_group", tokenizer))
    return chunks


def build_form_chunks(forms, max_tokens=DEFAULT_MAX_TOKENS, overlap_tokens=DEFAULT_OVERLAP_TOKENS):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    chunks = []
    for form in forms:
        chunks.extend(chunk_form(form, max_tokens, overlap_tokens, tokenizer))
    for index, chunk in enumerate(chunks, 1):
        form_uid = str(chunk["metadata"].get("form_uid") or chunk["metadata"].get("form_number") or "form").lower().replace(" ", "-")
        chunk["chunk_id"] = f"{form_uid}-{index:04d}"
    return chunks


def chunk_report(chunks):
    token_counts = [chunk["token_count"] for chunk in chunks]
    by_form = {}
    for chunk in chunks:
        form_number = chunk["metadata"].get("form_number")
        by_form[form_number] = by_form.get(form_number, 0) + 1
    return {
        "chunk_count": len(chunks),
        "min_tokens": min(token_counts) if token_counts else 0,
        "avg_tokens": mean(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "forms_by_chunk_count": sorted(by_form.items(), key=lambda item: item[1], reverse=True),
    }


def save_chunks(chunks, path=DEFAULT_FORM_CHUNKS_JSON):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
