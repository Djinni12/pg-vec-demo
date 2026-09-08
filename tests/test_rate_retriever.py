"""Tests for structured GST rate retriever and 2025 PDF rate database."""

import pytest
import psycopg
from src.retrievers.rate_retriever import (
    database_url,
    exact_code_lookup,
    extract_query_codes,
    extract_search_phrase,
    retrieve_rates,
    text_rate_search,
)


def test_extract_query_codes():
    assert "8471" in extract_query_codes("What is the rate for HSN 8471?")
    assert "0406" in extract_query_codes("Rate for HSN: 0406")
    assert "6815" in extract_query_codes("Heading 6815 fly ash bricks")
    assert "28044010" in extract_query_codes("Rate on 2804 40 10 medical oxygen")
    assert extract_query_codes("What is the rate on paneer?") == []


def test_extract_search_phrase():
    assert extract_search_phrase("What is the GST rate on paneer?") == "paneer"
    assert extract_search_phrase("GST rate on toothpaste") == "toothpaste"
    assert extract_search_phrase("Rate for medical oxygen") == "medical oxygen"
    assert extract_search_phrase("What is the rate on fly ash bricks?") == "fly ash bricks"


@pytest.fixture(scope="module")
def db_conn():
    try:
        conn = psycopg.connect(database_url())
        yield conn
        conn.close()
    except Exception as exc:
        pytest.skip(f"Database connection unavailable: {exc}")


def test_no_old_csv_rate_tables_remain(db_conn):
    """Verify that old CSV tables are completely removed from active database."""
    with db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name IN ('gst_goods_rates', 'gst_service_rates');
            """
        )
        old_tables = [r[0] for r in cur.fetchall()]
        assert old_tables == [], f"Old CSV tables still found in database: {old_tables}"

        # Verify gst_rates_2025 is present and populated
        cur.execute("SELECT COUNT(*) FROM gst_rates_2025;")
        count = cur.fetchone()[0]
        assert count >= 1500, f"Expected >= 1500 rows in gst_rates_2025, found {count}"


def test_exact_code_lookup_0406_multiple_entries(db_conn):
    """HSN 0406 must return multiple entries: 5% cheese vs 0% chena/paneer."""
    results = exact_code_lookup(db_conn, "0406")
    assert len(results) >= 2

    # Check 5% cheese entry
    cheese_entry = next((r for r in results if "cheese" in r["description"].lower() and "other than" in r["description"].lower()), None)
    assert cheese_entry is not None
    assert cheese_entry["igst_rate_pct"] == 5.0
    assert cheese_entry["cgst_rate_pct"] == 2.5
    assert cheese_entry["schedule"] == "Schedule I – 2.5%"

    # Check 0% chena or paneer entry
    paneer_entry = next((r for r in results if r["description"].lower().startswith("chena or paneer")), None)
    assert paneer_entry is not None
    assert paneer_entry["igst_rate_pct"] == 0.0
    assert "Nil / Exempt" in paneer_entry["formatted_rate"]
    assert paneer_entry["schedule"] == "Exempted Goods Schedule"


def test_exact_code_lookup_6815_fly_ash_bricks_conditions(db_conn):
    """HSN 6815 must return conditional Notification 02/2022 and Notification 14/2025."""
    results = exact_code_lookup(db_conn, "6815")
    assert len(results) >= 2

    # Notification 02/2022 (Condition 1: input tax credit not taken)
    cond_entry = next((r for r in results if r["condition_number"] == "1"), None)
    assert cond_entry is not None
    assert cond_entry["cgst_rate_pct"] == 3.0
    assert cond_entry["igst_rate_pct"] == 6.0
    assert cond_entry["condition"] is not None
    assert "credit of input tax" in cond_entry["condition"].lower()

    # Notification 14/2025 (Standard 6% CGST / 12% IGST)
    std_entry = next((r for r in results if r["notification_number"] == "14/2025-Central Tax (Rate)"), None)
    assert std_entry is not None
    assert std_entry["cgst_rate_pct"] == 6.0
    assert std_entry["igst_rate_pct"] == 12.0


def test_exact_code_lookup_28044010_medical_oxygen(db_conn):
    """HSN 2804 40 10 must return Medical grade oxygen at 5% IGST."""
    results = exact_code_lookup(db_conn, "28044010")
    assert len(results) >= 1
    hit = results[0]
    assert "oxygen" in hit["description"].lower()
    assert hit["igst_rate_pct"] == 5.0
    assert hit["cgst_rate_pct"] == 2.5


def test_text_search_paneer(db_conn):
    """Fuzzy search for paneer returns both exempt chena/paneer and 5% cheese."""
    results = text_rate_search(db_conn, "paneer", limit=5)
    assert len(results) >= 2
    descriptions = [r["description"].lower() for r in results]
    assert any("chena or paneer" in d for d in descriptions)
    assert any("cheese, other than chena or paneer" in d for d in descriptions)


def test_text_search_toothpaste(db_conn):
    """Search for toothpaste returns HSN 3306 Toothpaste at 5% IGST."""
    results = text_rate_search(db_conn, "toothpaste", limit=5)
    assert len(results) >= 1
    first = results[0]
    assert "toothpaste" in first["description"].lower()
    assert first["igst_rate_pct"] == 5.0


def test_text_search_medical_oxygen(db_conn):
    """Search for medical oxygen returns Medical grade oxygen."""
    results = text_rate_search(db_conn, "medical oxygen", limit=5)
    assert len(results) >= 1
    first = results[0]
    assert "oxygen" in first["description"].lower()
    assert "medical" in first["description"].lower()
    assert first["code"] == "2804 40 10"


def test_text_search_fly_ash_bricks(db_conn):
    """Search for fly ash bricks returns candidate entries with conditions."""
    results = text_rate_search(db_conn, "fly ash bricks", limit=5)
    assert len(results) >= 2
    has_condition_1 = any(r.get("condition_number") == "1" for r in results)
    assert has_condition_1 is True


def test_retrieve_rates_empty_query():
    assert retrieve_rates("") == []
    assert retrieve_rates("   ") == []
