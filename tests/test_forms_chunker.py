"""Tests for GST Forms Chunker."""

import json
from pathlib import Path

import pytest


def test_chunker_module_imports():
    """Test that the forms chunker module can be imported."""
    from forms_chunker import (
        DEFAULT_FORMS_JSON,
        DEFAULT_FORM_CHUNKS_JSON,
        DEFAULT_MAX_TOKENS,
        load_forms,
        parent_context,
        chunk_metadata,
        make_chunk,
        chunk_form,
        chunk_forms,
        validate_chunks,
    )
    
    assert DEFAULT_FORMS_JSON is not None
    assert DEFAULT_MAX_TOKENS > 0


def test_parent_context():
    """Test parent context generation for form chunks."""
    from forms_chunker import parent_context
    
    form = {
        "form_code": "GST CMP-01",
        "form_family": "Composition Levy",
        "title_original": "Application for Composition Levy",
    }
    
    context = parent_context(form)
    assert "Composition Levy" in context
    assert "GST CMP-01" in context


def test_chunk_metadata():
    """Test chunk metadata structure."""
    from forms_chunker import chunk_metadata
    
    form = {
        "form_code": "GST REG-01",
        "form_family": "Registration",
        "form_number": "01",
        "language": "hi",
        "rule_references": [{"rule_number": "5"}],
        "source_metadata": {"pages": [1, 2], "start_page": 1, "end_page": 2},
    }
    
    metadata = chunk_metadata(form, ["1"], {"block_type": "main_form"}, "block")
    
    assert metadata["document_type"] == "form"
    assert metadata["form_code"] == "GST REG-01"
    assert metadata["language"] == "hi"
    assert metadata["chunk_strategy"] == "block"


def test_validate_chunks_structure():
    """Test validation returns proper structure."""
    from forms_chunker import validate_chunks
    
    forms = [{"form_code": "GST CMP-01", "content_original": "test content"}]
    chunks = [{
        "chunk_id": "cgst-form-GST CMP-01-0001",
        "text": "Composition Levy\n\nForm GST CMP-01\n\ntest content",
        "token_count": 10,
        "metadata": {"form_code": "GST CMP-01"},
    }]
    
    result = validate_chunks(forms, chunks)
    
    assert "PASS" in result
    assert "missing_forms" in result
    assert "content_loss" in result
    assert "statistics" in result


def test_empty_chunks_validation():
    """Test validation detects empty chunks."""
    from forms_chunker import validate_chunks
    
    forms = [{"form_code": "GST CMP-01", "content_original": "test content"}]
    chunks = [{
        "chunk_id": "cgst-form-GST CMP-01-0001",
        "text": "",
        "token_count": 0,
        "metadata": {"form_code": "GST CMP-01"},
    }]
    
    result = validate_chunks(forms, chunks)
    
    assert result["PASS"] == False
    assert len(result["empty_chunks"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
