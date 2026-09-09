"""Tests for structured CSV-based GST rate retriever."""

import pytest
from src.retrievers.rate_retriever import (
    EXPECTED_ROW_COUNT,
    REQUIRED_FIELDS,
    exact_code_lookup,
    extract_query_codes,
    extract_search_phrase,
    load_rates_csv,
    normalize_hsn_codes,
    retrieve_rates,
    text_rate_search,
)


def test_csv_loads_exactly_1663_rows_and_all_columns():
    """Verify that exactly 1,663 rows load without silent skipping and schema is intact."""
    rows = load_rates_csv(force_reload=True)
    assert len(rows) == EXPECTED_ROW_COUNT == 1663

    sample = rows[0]
    for field in REQUIRED_FIELDS:
        assert field in sample, f"Missing required column: {field}"


def test_csv_escaped_quotes_smart_cards():
    """Verify escaped quotes like \\\"smart cards\\\" are parsed without row skipping."""
    rows = load_rates_csv()
    smart_cards = [r for r in rows if "smart cards" in r["description"]]
    assert len(smart_cards) == 1
    item = smart_cards[0]
    assert item["hsn_code"] == "8523"
    assert item["gst_rate"] == "9%"
    assert '"smart cards"' in item["description"]


def test_leading_zeros_preserved_in_hsn_codes():
    """Verify that HSN codes with leading zeros (e.g. 0402, 0101) are never stripped."""
    rows = load_rates_csv()
    hsn_0402 = next((r for r in rows if r["hsn_code"] == "0402"), None)
    assert hsn_0402 is not None
    assert hsn_0402["hsn_code"] == "0402"
    assert hsn_0402["hsn_code"].startswith("0")

    hsn_0101 = next((r for r in rows if "0101" in r["hsn_code"]), None)
    assert hsn_0101 is not None
    assert "0101" in hsn_0101["hsn_code"]


def test_rates_and_legal_notes_preserved_faithfully():
    """Verify rates are not modified or inferred, and conditions/footnotes/amendments are kept."""
    rows = load_rates_csv()
    rates = {r["gst_rate"] for r in rows}
    assert "2.5%" in rates
    assert "9%" in rates
    assert "Nil" in rates
    assert "14%" in rates

    # Check conditional Notification 02/2022 row has condition text
    cond_row = next((r for r in rows if r["notification_no"] == "02/2022-Central Tax (Rate)"), None)
    assert cond_row is not None
    assert cond_row["condition"] != ""
    assert "credit of input tax" in cond_row["condition"].lower()


def test_extract_query_codes():
    assert "0402" in extract_query_codes("GST rate for HSN 0402")
    assert "8518" in extract_query_codes("What is the GST rate for HSN 8518?")
    assert "8523" in extract_query_codes("rate of HSN 8523")
    assert "01012100" in extract_query_codes("Rate on 0101 21 00 live horses")
    assert extract_query_codes("What is the rate for milk?") == []


def test_extract_search_phrase():
    assert extract_search_phrase("GST rate for milk") == "milk"
    assert extract_search_phrase("What is the GST rate for milk?") == "milk"
    assert extract_search_phrase("GST on microphones") == "microphones"
    assert extract_search_phrase("What is the GST rate for CCTV cameras?") == "CCTV cameras"
    assert extract_search_phrase("tax rate on smart cards") == "smart cards"
    assert extract_search_phrase("What is the rate for smart cards?") == "smart cards"
    assert extract_search_phrase("GST rate for live horses") == "live horses"


def test_exact_query_hsn_0402():
    """Test 'GST rate for HSN 0402'."""
    hits = retrieve_rates("GST rate for HSN 0402", limit=5)
    assert len(hits) >= 1
    hit = hits[0]
    assert hit["hsn_code"] == "0402"
    assert "milk and cream" in hit["description"].lower()
    assert hit["gst_rate"] == "2.5%"
    assert hit["schedule"] == "Schedule I – 2.5%"
    assert hit["serial_no"] == "4."


def test_exact_query_hsn_8518():
    """Test 'What is the GST rate for HSN 8518?'."""
    hits = retrieve_rates("What is the GST rate for HSN 8518?", limit=5)
    assert len(hits) >= 1
    hit = hits[0]
    assert hit["hsn_code"] == "8518"
    assert "microphones" in hit["description"].lower()
    assert hit["gst_rate"] == "9%"
    assert hit["schedule"] == "Schedule II – 9%"
    assert hit["serial_no"] == "491."


def test_exact_query_hsn_8523():
    """Test 'rate of HSN 8523'."""
    hits = retrieve_rates("rate of HSN 8523", limit=5)
    assert len(hits) >= 1
    hit = hits[0]
    assert hit["hsn_code"] == "8523"
    assert "smart cards" in hit["description"]
    assert hit["gst_rate"] == "9%"
    assert hit["schedule"] == "Schedule II – 9%"
    assert hit["serial_no"] == "495."


def test_product_query_milk():
    """Test 'What is the GST rate for milk?'."""
    hits = retrieve_rates("What is the GST rate for milk?", limit=5)
    assert len(hits) >= 2
    # Should include both fresh milk (Nil) and concentrated milk/cream (2.5%)
    hsn_codes = [h["hsn_code"] for h in hits]
    assert any("0402" in code for code in hsn_codes)
    assert any("0401" in code for code in hsn_codes)


def test_product_query_microphones():
    """Test 'GST on microphones'."""
    hits = retrieve_rates("GST on microphones", limit=5)
    assert len(hits) >= 1
    hit = hits[0]
    assert hit["hsn_code"] == "8518"
    assert "microphones" in hit["description"].lower()
    assert hit["gst_rate"] == "9%"
    assert hit["schedule"] == "Schedule II – 9%"
    assert hit["serial_no"] == "491."


def test_product_query_cctv_cameras():
    """Test 'GST rate for CCTV cameras'."""
    hits = retrieve_rates("GST rate for CCTV cameras", limit=5)
    assert len(hits) >= 1
    hit = hits[0]
    assert hit["hsn_code"] == "8525"
    assert "cctv" in hit["description"].lower()
    assert hit["gst_rate"] == "9%"
    assert hit["schedule"] == "Schedule II – 9%"
    assert hit["serial_no"] == "497."


def test_product_query_smart_cards():
    """Test 'What is the rate for smart cards?'."""
    hits = retrieve_rates("What is the rate for smart cards?", limit=5)
    assert len(hits) >= 1
    hit = hits[0]
    assert hit["hsn_code"] == "8523"
    assert "smart cards" in hit["description"].lower()
    assert hit["gst_rate"] == "9%"
    assert hit["schedule"] == "Schedule II – 9%"
    assert hit["serial_no"] == "495."


def test_product_query_live_horses():
    """Test 'GST rate for live horses'."""
    hits = retrieve_rates("GST rate for live horses", limit=5)
    assert len(hits) >= 1
    hit = hits[0]
    assert "0101" in hit["hsn_code"]
    assert "live horses" in hit["description"].lower()
    assert hit["gst_rate"] == "2.5%"
    assert hit["schedule"] == "Schedule I – 2.5%"
    assert hit["serial_no"] == "1."


def test_nonexistent_or_ambiguous_product_no_fabrication():
    """Test a product that does NOT exist to ensure the system does not fabricate a rate."""
    hits1 = retrieve_rates("GST rate for kryptonite spaceships", limit=5)
    assert hits1 == []

    hits2 = retrieve_rates("What is the GST rate on magical unicorn feathers?", limit=5)
    assert hits2 == []


def test_retrieve_rates_empty_query():
    assert retrieve_rates("") == []
    assert retrieve_rates("   ") == []


def test_product_query_butter_central_tax_metadata():
    """Verify butter query retrieves HSN 0405 with accurate Central Tax rate_type, section and dates."""
    hits = retrieve_rates("What is the GST rate for butter?", limit=5)
    assert len(hits) >= 1
    hit = hits[0]
    assert hit["hsn_code"] == "0405"
    assert "butter" in hit["description"].lower()
    assert hit["section_heading"] == "CGST rates on goods as on 22.09.2025"
    assert hit["rate_category"] == "CGST"
    assert hit["source_rate"] == "2.5%"
    assert hit["cgst_rate"] == "2.5%"
    assert hit["sgst_rate"] == "2.5%"
    assert hit["total_gst_rate"] == "5%"
    assert hit["gst_rate"] == "2.5%"
    assert hit["rate_type"] == "Central Tax (CGST)"
    assert hit["notification_no"] == "09/2025-Central Tax (Rate)"
    assert hit["notification_date"] == "17th September, 2025"
    assert hit["rate_as_on_date"] == "22.09.2025"
    assert hit["source_page"] == "1"
    assert hit["effective_date"] is None


