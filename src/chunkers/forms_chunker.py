"""Structure-aware chunks for parsed GST Forms."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


DEFAULT_FORMS_JSON = Path("data/forms/cgst_forms_parsed.json")
DEFAULT_FORM_CHUNKS_JSON = Path("data/forms/cgst_forms_chunks.json")
DEFAULT_MAX_TOKENS = 450
DEFAULT_OVERLAP_TOKENS = 60


def load_forms(path=DEFAULT_FORMS_JSON):
    """Load parsed form dictionaries from JSON."""
    with Path(path).open(encoding="utf-8") as handle:
        forms = json.load(handle)
    if not isinstance(forms, list):
        raise ValueError("Expected forms JSON root to be a list")
    return forms


def count_tokens(tokenizer, text, words_per_batch=250):
    """Count tokens with a Hugging Face-style tokenizer."""
    words = (text or "").split()
    total = 0
    for start in range(0, len(words), words_per_batch):
        total += len(tokenizer.tokenize(" ".join(words[start:start + words_per_batch])))
    return total


def parent_context(form):
    """Return compact context prepended to form chunks."""
    form_code = form.get("form_code", "")
    title = form.get("title_original", "")
    form_family = form.get("form_family", "")
    
    parts = []
    if form_family:
        parts.append(form_family)
    if form_code:
        parts.append(f"Form {form_code}")
    if title:
        parts.append(title)
    
    return "\n".join(parts)


def chunk_metadata(form, section_numbers, part_info, strategy):
    """Build form chunk metadata."""
    return {
        "document_type": "form",
        "form_code": form.get("form_code"),
        "form_family": form.get("form_family"),
        "form_number": form.get("form_number"),
        "language": form.get("language", "hi"),
        "rule_references": form.get("rule_references"),
        "section_numbers": section_numbers,
        "part_number": part_info.get("part_number") if part_info else None,
        "part_title": part_info.get("part_title") if part_info else None,
        "block_type": part_info.get("block_type") if part_info else None,
        "status": form.get("status"),
        "chunk_strategy": strategy,
        "source_pages": form.get("source_metadata", {}).get("pages"),
        "start_page": form.get("source_metadata", {}).get("start_page"),
        "end_page": form.get("source_metadata", {}).get("end_page"),
    }


def make_chunk(form, text, section_numbers, part_info, strategy, tokenizer):
    """Create one form chunk with parent context and token count."""
    context = parent_context(form)
    chunk_text = f"{context}\n\n{text.strip()}" if context else text.strip()
    return {
        "chunk_id": None,
        "text": chunk_text,
        "token_count": count_tokens(tokenizer, chunk_text),
        "metadata": chunk_metadata(form, section_numbers, part_info, strategy),
    }


def split_large_text(text, max_tokens, overlap_tokens, tokenizer):
    """Fallback split for oversized text, using token-aware word windows."""
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


def is_table_block(block):
    """Check if a block contains table content."""
    content = block.get("content_original", "")
    # Detect table-like patterns
    has_tab_separated = "\t" in content
    has_pipe_table = "|" in content and content.count("|") > 2
    has_numbered_rows = bool(__import__("re").search(r"(?m)^\s*\d+[.\)]\s+", content))
    return has_tab_separated or has_pipe_table or has_numbered_rows


def emit_grouped_blocks(form, group, chunks, max_tokens, overlap_tokens, tokenizer, part_info=None):
    """Append chunks for a buffered group of complete blocks/fields."""
    if not group:
        return

    if len(group) == 1:
        block = group[0]
        block_type = block.get("block_type", "field")
        text = block.get("content_original", "")
        
        standalone = make_chunk(
            form, 
            text, 
            [block.get("field_number")] if block.get("field_number") else [],
            part_info,
            "block" if block_type != "field" else "field",
            tokenizer,
        )
        
        if standalone["token_count"] <= max_tokens:
            chunks.append(standalone)
            return

        # Token-split the large block
        body_limit = max(1, max_tokens - count_tokens(tokenizer, parent_context(form)))
        for split_index, part in enumerate(split_large_text(text, body_limit, overlap_tokens, tokenizer), 1):
            chunk = make_chunk(
                form, 
                part, 
                [block.get("field_number")] if block.get("field_number") else [],
                part_info,
                "block_token_split",
                tokenizer,
            )
            chunk["metadata"]["split_index"] = split_index
            chunks.append(chunk)
        return

    # Group multiple blocks together
    text = "\n\n".join(
        block.get("content_original", "").strip() 
        for block in group 
        if block.get("content_original")
    )
    section_numbers = [
        block.get("field_number") 
        for block in group 
        if block.get("field_number")
    ]
    chunks.append(make_chunk(form, text, section_numbers, part_info, "block_group", tokenizer))


def chunk_by_parts(form, max_tokens, overlap_tokens, tokenizer):
    """Chunk form by Part/Section divisions."""
    chunks = []
    parts = form.get("parts", [])
    blocks = form.get("blocks", [])
    
    if not parts:
        # No explicit parts, chunk by blocks
        return chunk_by_blocks(form, max_tokens, overlap_tokens, tokenizer)
    
    # Group blocks by part
    for part in parts:
        part_start = part.get("start", 0)
        part_end = part.get("end", len(form.get("content_original", "")))
        
        # Find blocks within this part
        part_blocks = [
            block for block in blocks
            if block.get("start_offset", 0) >= part_start 
            and block.get("start_offset", 0) < part_end
        ]
        
        if not part_blocks:
            # Extract text directly from content for this part
            content = form.get("content_original", "")
            part_text = content[part_start:part_end].strip()
            if part_text:
                chunk = make_chunk(
                    form,
                    part_text,
                    [],
                    {"part_number": part.get("part_number"), "part_title": part.get("part_title"), "block_type": "part"},
                    "part",
                    tokenizer,
                )
                if chunk["token_count"] <= max_tokens:
                    chunks.append(chunk)
                else:
                    # Split large part
                    body_limit = max(1, max_tokens - count_tokens(tokenizer, parent_context(form)))
                    for split_index, part_split in enumerate(
                        split_large_text(part_text, body_limit, overlap_tokens, tokenizer), 1
                    ):
                        split_chunk = make_chunk(
                            form,
                            part_split,
                            [],
                            {"part_number": part.get("part_number"), "part_title": part.get("part_title"), "block_type": "part"},
                            "part_token_split",
                            tokenizer,
                        )
                        split_chunk["metadata"]["split_index"] = split_index
                        chunks.append(split_chunk)
        else:
            # Chunk the part's blocks
            part_info = {"part_number": part.get("part_number"), "part_title": part.get("part_title"), "block_type": "part"}
            group = []
            for block in part_blocks:
                candidate = [*group, block]
                candidate_text = "\n\n".join(
                    item.get("content_original", "").strip() 
                    for item in candidate 
                    if item.get("content_original")
                )
                candidate_chunk = make_chunk(
                    form,
                    candidate_text,
                    [item.get("field_number") for item in candidate if item.get("field_number")],
                    part_info,
                    "block_group",
                    tokenizer,
                )
                if group and candidate_chunk["token_count"] > max_tokens:
                    emit_grouped_blocks(form, group, chunks, max_tokens, overlap_tokens, tokenizer, part_info)
                    group = [block]
                else:
                    group = candidate
            emit_grouped_blocks(form, group, chunks, max_tokens, overlap_tokens, tokenizer, part_info)
    
    return chunks


def chunk_by_blocks(form, max_tokens, overlap_tokens, tokenizer):
    """Chunk form by logical blocks (main_form, instructions, verification, attachments, fields)."""
    chunks = []
    blocks = form.get("blocks", [])
    
    if not blocks:
        # Fall back to whole-form chunking
        return chunk_form_whole(form, max_tokens, overlap_tokens, tokenizer)
    
    # Check if entire form fits
    full_content = form.get("content_original", "")
    whole = make_chunk(form, full_content, [], None, "form", tokenizer)
    if whole["token_count"] <= max_tokens:
        return [whole]
    
    # Group blocks strategically
    main_blocks = []
    instruction_blocks = []
    verification_blocks = []
    attachment_blocks = []
    field_blocks = []
    
    for block in blocks:
        block_type = block.get("block_type", "")
        if block_type == "main_form":
            main_blocks.append(block)
        elif block_type == "instructions":
            instruction_blocks.append(block)
        elif block_type == "verification":
            verification_blocks.append(block)
        elif block_type == "attachments":
            attachment_blocks.append(block)
        else:
            field_blocks.append(block)
    
    # Process each category
    for block_group, block_type in [
        (main_blocks, "main_form"),
        (instruction_blocks, "instructions"),
        (verification_blocks, "verification"),
        (attachment_blocks, "attachments"),
        (field_blocks, "field"),
    ]:
        if not block_group:
            continue
        
        part_info = {"block_type": block_type}
        group = []
        for block in block_group:
            candidate = [*group, block]
            candidate_text = "\n\n".join(
                item.get("content_original", "").strip() 
                for item in candidate 
                if item.get("content_original")
            )
            candidate_chunk = make_chunk(
                form,
                candidate_text,
                [item.get("field_number") for item in candidate if item.get("field_number")],
                part_info,
                "block_group",
                tokenizer,
            )
            if group and candidate_chunk["token_count"] > max_tokens:
                emit_grouped_blocks(form, group, chunks, max_tokens, overlap_tokens, tokenizer, part_info)
                group = [block]
            else:
                group = candidate
        emit_grouped_blocks(form, group, chunks, max_tokens, overlap_tokens, tokenizer, part_info)
    
    return chunks


def chunk_form_whole(form, max_tokens, overlap_tokens, tokenizer):
    """Chunk entire form as single unit or split if too large."""
    chunks = []
    content = form.get("content_original", "")
    
    whole = make_chunk(form, content, [], None, "form", tokenizer)
    if whole["token_count"] <= max_tokens:
        return [whole]
    
    # Form is too large, split by token windows
    body_limit = max(1, max_tokens - count_tokens(tokenizer, parent_context(form)))
    for split_index, part in enumerate(split_large_text(content, body_limit, overlap_tokens, tokenizer), 1):
        chunk = make_chunk(form, part, [], None, "form_token_split", tokenizer)
        chunk["metadata"]["split_index"] = split_index
        chunks.append(chunk)
    
    return chunks


def chunk_form(form, max_tokens, overlap_tokens, tokenizer):
    """Chunk one parsed Form without crossing Form boundaries."""
    # Strategy selection:
    # 1. Small form -> keep whole
    # 2. Has parts -> split by part
    # 3. Has blocks -> split by block type
    # 4. Large without structure -> token-split
    
    content = form.get("content_original", "")
    whole = make_chunk(form, content, [], None, "form", tokenizer)
    
    if whole["token_count"] <= max_tokens:
        return [whole]
    
    parts = form.get("parts", [])
    blocks = form.get("blocks", [])
    
    if parts:
        return chunk_by_parts(form, max_tokens, overlap_tokens, tokenizer)
    
    if blocks:
        return chunk_by_blocks(form, max_tokens, overlap_tokens, tokenizer)
    
    # Fallback: token-split the whole form
    return chunk_form_whole(form, max_tokens, overlap_tokens, tokenizer)


def chunk_forms(forms, max_tokens=DEFAULT_MAX_TOKENS, overlap_tokens=DEFAULT_OVERLAP_TOKENS, tokenizer=None):
    """Return structure-aware chunks for parsed GST Forms."""
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
    for form in forms:
        form_chunks = chunk_form(form, max_tokens, overlap_tokens, tokenizer)
        chunks.extend(form_chunks)

    # Assign chunk IDs
    for index, chunk in enumerate(chunks, 1):
        form_code = chunk["metadata"].get("form_code", "UNKNOWN")
        chunk["chunk_id"] = f"cgst-form-{form_code}-{index:04d}"
    
    return chunks


def chunk_size_report(chunks):
    """Return min/avg/max token sizes and chunk counts by form."""
    counts = [chunk["token_count"] for chunk in chunks]
    by_form = {}
    for chunk in chunks:
        form_code = chunk["metadata"].get("form_code")
        by_form[form_code] = by_form.get(form_code, 0) + 1
    most_chunks = sorted(by_form.items(), key=lambda item: (-item[1], str(item[0])))[:10]
    return {
        "chunk_count": len(chunks),
        "min_tokens": min(counts) if counts else 0,
        "avg_tokens": round(mean(counts), 2) if counts else 0,
        "max_tokens": max(counts) if counts else 0,
        "forms_with_most_chunks": most_chunks,
    }


def validate_chunks(forms, chunks):
    """Validate chunks against original forms."""
    issues = {
        "missing_forms": [],
        "cross_form_chunks": [],
        "oversized_chunks": [],
        "broken_table_blocks": [],
        "content_loss": [],
        "empty_chunks": [],
        "PASS": True,
    }
    
    form_codes = {form.get("form_code") for form in forms if form.get("form_code")}
    chunked_forms = set()
    
    max_tokens_seen = 0
    
    for chunk in chunks:
        form_code = chunk["metadata"].get("form_code")
        chunked_forms.add(form_code)
        
        # Check for empty chunks
        if not chunk.get("text", "").strip():
            issues["empty_chunks"].append(chunk.get("chunk_id"))
            issues["PASS"] = False
        
        # Track max token size
        token_count = chunk.get("token_count", 0)
        if token_count > max_tokens_seen:
            max_tokens_seen = token_count
        
        # Check for cross-form contamination (chunk mentioning multiple form codes)
        text = chunk.get("text", "")
        mentioned_forms = set()
        for fc in form_codes:
            if fc and fc in text:
                mentioned_forms.add(fc)
        if len(mentioned_forms) > 1:
            # This might be legitimate if forms reference each other
            issues["cross_form_chunks"].append({
                "chunk_id": chunk.get("chunk_id"),
                "mentioned_forms": list(mentioned_forms),
            })
    
    # Check for missing forms
    missing = form_codes - chunked_forms
    if missing:
        issues["missing_forms"] = list(missing)
        issues["PASS"] = False
    
    # Check for content loss
    for form in forms:
        form_code = form.get("form_code")
        original_content = form.get("content_original", "")
        
        # Get all chunks for this form
        form_chunks = [c for c in chunks if c["metadata"].get("form_code") == form_code]
        
        if not form_chunks and original_content.strip():
            issues["content_loss"].append({
                "form_code": form_code,
                "reason": "no chunks generated",
            })
            issues["PASS"] = False
            continue
        
        # Reconstruct content from chunks (without parent context)
        reconstructed = []
        for chunk in form_chunks:
            text = chunk.get("text", "")
            # Remove parent context prefix
            context = parent_context(form)
            if text.startswith(context):
                text = text[len(context):].strip()
                if text.startswith("\n\n"):
                    text = text[2:]
            reconstructed.append(text)
        
        reconstructed_text = "\n\n".join(reconstructed)
        
        # Simple check: original keywords should appear in chunks
        original_words = set(original_content.split())
        reconstructed_words = set(reconstructed_text.split())
        
        # Allow some loss due to normalization, but flag significant loss
        if len(original_words) > 0:
            overlap_ratio = len(original_words & reconstructed_words) / len(original_words)
            if overlap_ratio < 0.8:  # Less than 80% word overlap
                issues["content_loss"].append({
                    "form_code": form_code,
                    "reason": f"word overlap ratio {overlap_ratio:.2f} < 0.80",
                })
                issues["PASS"] = False
    
    # Report statistics
    issues["statistics"] = {
        "total_forms": len(forms),
        "total_chunks": len(chunks),
        "chunks_per_form_avg": round(len(chunks) / len(forms), 2) if forms else 0,
        "max_tokens_in_chunk": max_tokens_seen,
    }
    
    return issues


def save_chunks(chunks, output_path=None):
    """Save chunks to JSON file."""
    if output_path is None:
        output_path = Path("data/forms/cgst_forms_chunks.json")
    
    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    return output_path


if __name__ == "__main__":
    import sys
    
    forms_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FORMS_JSON
    
    if not forms_path.exists():
        print(f"Error: Forms JSON not found at {forms_path}")
        print("Run gst_forms_parser.py first to generate parsed forms.")
        sys.exit(1)
    
    print(f"Loading forms from {forms_path}...")
    forms = load_forms(forms_path)
    print(f"Loaded {len(forms)} forms")
    
    print("\nChunking forms...")
    chunks = chunk_forms(forms)
    print(f"Generated {len(chunks)} chunks")
    
    # Report
    report = chunk_size_report(chunks)
    print(f"\nChunk Statistics:")
    print(f"  Total chunks: {report['chunk_count']}")
    print(f"  Min tokens: {report['min_tokens']}")
    print(f"  Avg tokens: {report['avg_tokens']}")
    print(f"  Max tokens: {report['max_tokens']}")
    
    # Validate
    print("\nValidating chunks...")
    validation = validate_chunks(forms, chunks)
    print(f"  PASS: {validation['PASS']}")
    print(f"  Missing forms: {len(validation['missing_forms'])}")
    print(f"  Content loss issues: {len(validation['content_loss'])}")
    print(f"  Empty chunks: {len(validation['empty_chunks'])}")
    print(f"  Cross-form chunks: {len(validation['cross_form_chunks'])}")
    
    # Save
    output_path = save_chunks(chunks)
    print(f"\nSaved chunks to {output_path}")
    
    # Save validation report
    report_path = Path("data/forms/cgst_forms_validation.json")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(validation, f, ensure_ascii=False, indent=2)
    print(f"Saved validation report to {report_path}")
