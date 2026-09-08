"""Comprehensive test suite for the GST rates2025.pdf extraction and ingestion pipeline."""

from pathlib import Path
import pytest
import psycopg

from src.parsers.pdf_rate_parser import (
    DEFAULT_PDF_PATH,
    FALLBACK_PDF_PATH,
    normalize_digits,
    parse_pdf_rates,
    resolve_pdf_path,
    validate_extracted_rates,
)
from src.retrievers.rate_retriever import (
    database_url,
    exact_code_lookup,
    retrieve_rates,
    text_rate_search,
)


def test_resolve_pdf_path():
    p = resolve_pdf_path()
    assert p.exists()
    assert p.name == "GST rates2025.pdf"


def test_normalize_digits_variations():
    assert set(normalize_digits("0101 21 00, 0101 29")) == {"01", "0101", "01012100", "010129"}
    assert set(normalize_digits("0406")) == {"04", "0406"}
    assert set(normalize_digits("6815")) == {"68", "6815"}
    assert set(normalize_digits("2804 40 10")) == {"28", "2804", "28044010"}
    assert normalize_digits("Chapter 3") == ["03"]
    assert normalize_digits("") == []


@pytest.fixture(scope="module")
def parsed_records():
    return parse_pdf_rates()


def test_pdf_extraction_validation(parsed_records):
    """Verify that complete PDF extracts over 1400 rows with 0 validation errors."""
    records = parsed_records
    assert len(records) >= 1500

    report = validate_extracted_rates(records)
    assert report["is_valid"] is True, f"Validation errors: {report['errors']}"
    assert report["errors"] == []
    assert report["total_records"] >= 1500
    assert report["categories"]["goods"] > 1000
    assert report["categories"]["exempted_goods"] > 200
    assert report["categories"]["cess"] > 50


def test_pdf_multiple_entries_for_same_hsn(parsed_records):
    """Verify HSN 0406 and 6815 have multiple valid entries in the extracted dataset."""
    records = parsed_records

    # 0406: Cheese vs Chena/paneer
    entries_0406 = [r for r in records if "0406" in r["hsn_code"]]
    assert len(entries_0406) >= 2
    rates = {e["rate"] for e in entries_0406}
    assert "2.5%" in rates
    assert "Nil" in rates

    # 6815: Fly ash bricks (conditional 3% vs standard 6%)
    entries_6815 = [r for r in records if "6815" in r["hsn_code"] and "fly ash" in r["description"].lower()]
    assert len(entries_6815) >= 2
    rates_6815 = {e["rate"] for e in entries_6815}
    assert "3%" in rates_6815
    assert "6%" in rates_6815


def test_pdf_conditions_populated(parsed_records):
    """Verify Notification 02/2022 condition text is extracted and attached."""
    records = parsed_records
    cond_entries = [r for r in records if r.get("condition_number") == "1"]
    assert len(cond_entries) >= 1
    for ce in cond_entries:
        assert ce["condition_text"] is not None
        assert "credit of input tax" in ce["condition_text"].lower()


@pytest.fixture(scope="module")
def db_conn():
    try:
        conn = psycopg.connect(database_url())
        yield conn
        conn.close()
    except Exception as exc:
        pytest.skip(f"Database unavailable: {exc}")


def test_database_no_csv_fallback_or_tables(db_conn):
    """Verify active database does not contain obsolete CSV rate tables."""
    with db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name IN ('gst_goods_rates', 'gst_service_rates');
            """
        )
        assert cur.fetchall() == []

        cur.execute("SELECT COUNT(*) FROM gst_rates_2025;")
        assert cur.fetchone()[0] >= 1500


def test_all_five_examples_in_database(db_conn):
    """Verify paneer, 0406, toothpaste, medical oxygen, fly ash bricks all retrieve properly."""
    # 1. paneer
    hits = retrieve_rates("What is the GST rate on paneer?")
    assert len(hits) >= 2
    descriptions = [h["description"].lower() for h in hits]
    assert any("chena or paneer" in d for d in descriptions)

    # 2. 0406
    hits = retrieve_rates("HSN 0406")
    assert len(hits) >= 2
    assert any(h["rate"] == "2.5%" for h in hits)
    assert any(h["rate"].lower() in ("nil", "exempt") for h in hits)

    # 3. toothpaste
    hits = retrieve_rates("GST rate for toothpaste")
    assert len(hits) >= 1
    assert "toothpaste" in hits[0]["description"].lower()
    assert hits[0]["igst_rate_pct"] == 5.0

    # 4. medical oxygen
    hits = retrieve_rates("What is the rate on medical oxygen?")
    assert len(hits) >= 1
    assert "oxygen" in hits[0]["description"].lower()
    assert hits[0]["igst_rate_pct"] == 5.0

    # 5. fly ash bricks
    hits = retrieve_rates("rate on fly ash bricks")
    assert len(hits) >= 2
    assert any(h.get("condition_number") == "1" for h in hits)
