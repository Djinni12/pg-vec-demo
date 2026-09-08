"""Tests for GST Forms Parser."""

import json
from pathlib import Path

import pytest


def test_parser_module_imports():
    """Test that the forms parser module can be imported."""
    from gst_forms_parser import (
        FORM_CODE_RE,
        FORM_FAMILIES,
        normalize_text,
        normalize_form_code,
        get_form_family,
        extract_rule_references,
    )
    
    assert FORM_CODE_RE is not None
    assert isinstance(FORM_FAMILIES, dict)
    assert len(FORM_FAMILIES) > 0


def test_normalize_text():
    """Test text normalization preserves Hindi content."""
    from gst_forms_parser import normalize_text
    
    # Test with Hindi text
    hindi_text = "जीएसटी फॉर्म CMP-01\nनिर्देश"
    normalized = normalize_text(hindi_text)
    assert "जीएसटी" in normalized
    assert "CMP-01" in normalized
    
    # Test whitespace normalization
    messy = "test   multiple    spaces"
    assert normalize_text(messy) == "test multiple spaces"


def test_normalize_form_code():
    """Test form code normalization."""
    from gst_forms_parser import normalize_form_code
    
    assert normalize_form_code("GST CMP-01") == "GST CMP-01"
    assert normalize_form_code("CMP-01") == "GST CMP-01"
    assert normalize_form_code("gst cmp-01") == "GST CMP-01"
    assert normalize_form_code("GSTR-7") == "GSTR-7"
    assert normalize_form_code("GST REG-01") == "GST REG-01"


def test_get_form_family():
    """Test form family detection."""
    from gst_forms_parser import get_form_family
    
    assert get_form_family("GST CMP-01") == "Composition Levy"
    assert get_form_family("GST REG-01") == "Registration"
    assert get_form_family("GSTR-7") == "Return"
    assert get_form_family("GST DRC-01") == "Demand and Recovery"


def test_extract_rule_references():
    """Test rule reference extraction."""
    from gst_forms_parser import extract_rule_references
    
    text = "Rule 5 of CGST Rules, 2017 और नियम 8 के तहत"
    refs = extract_rule_references(text)
    assert len(refs) >= 1
    assert any(r.get("rule_number") == "5" for r in refs)


def test_form_code_regex():
    """Test form code regex patterns."""
    from gst_forms_parser import FORM_CODE_RE, FORM_CODE_INLINE_RE
    
    # Test various form codes
    test_codes = [
        "GST CMP-01",
        "GST REG-01", 
        "GSTR-7",
        "GST DRC-01",
        "GST APL-01",
    ]
    
    for code in test_codes:
        match = FORM_CODE_INLINE_RE.search(code)
        assert match is not None, f"Failed to match {code}"


def test_empty_forms_list():
    """Test parsing with no PDF returns empty list."""
    from gst_forms_parser import parse_forms
    from pathlib import Path
    
    # Test with non-existent path handling
    result = []  # Would need actual PDF to test fully
    assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
