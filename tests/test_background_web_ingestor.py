"""Comprehensive test suite for the background web ingestion pipeline."""

import time
import pytest
import psycopg

from src.ingestors.background_web_ingestor import (
    classify_and_extract_evidence,
    execute_web_ingestion_job,
    is_trusted_domain,
    schedule_background_web_ingestion,
)
from src.retrievers.rate_retriever import (
    database_url,
    load_rates_csv,
    retrieve_rates,
)
from src.observability.trace_store import ExecutionTrace, trace_store


def test_is_trusted_domain_whitelist():
    """Verify trusted government and reputable tax domains pass, while consumer review sites are rejected."""
    # Trusted domains
    assert is_trusted_domain("https://services.gst.gov.in/services/searchhsnsac") is True
    assert is_trusted_domain("cbic-gst.gov.in") is True
    assert is_trusted_domain("https://gstcouncil.gov.in/rates") is True
    assert is_trusted_domain("taxguru.in") is True
    assert is_trusted_domain("cleartax.in/s/gst-rates") is True
    assert is_trusted_domain("taxmann.com") is True

    # Untrusted domains
    assert is_trusted_domain("https://www.tripadvisor.in/Restaurants-g304554") is False
    assert is_trusted_domain("https://www.zomato.com/mumbai/restaurants") is False
    assert is_trusted_domain("https://swiggy.com/restaurants") is False
    assert is_trusted_domain("https://www.reddit.com/r/india") is False
    assert is_trusted_domain("https://quora.com/What-is-GST") is False
    assert is_trusted_domain("") is False


def test_classify_and_extract_rate_evidence():
    """Verify rate chunk extraction properly identifies HSN, rate %, and conditions."""
    chunk = {
        "url": "https://services.gst.gov.in/services/searchhsnsac",
        "domain": "services.gst.gov.in",
        "title": "GST Rate on Mobile Phones and Smartphones - HSN 8517",
        "snippet": "Telephones for cellular networks or for other wireless networks; smartphones under HSN 8517 attract 18% GST (9% CGST + 9% SGST).",
    }
    cat, record, reason = classify_and_extract_evidence(
        chunk,
        user_query="What is the GST rate on mobile phones?",
        discovered_hsn="8517",
    )
    assert cat == "rate"
    assert record is not None
    assert record["hsn_code"] == "8517"
    assert record["rate"] == "18%"
    assert record["cgst_rate_pct"] == 9.0
    assert record["sgst_utgst_rate_pct"] == 9.0
    assert record["igst_rate_pct"] == 18.0
    assert "8517" in record["normalized_hsn_codes"]


def test_classify_and_extract_restaurant_sac():
    """Verify restaurant service without AC and liquor license extracts SAC 996331, 5% rate, and without ITC condition."""
    chunk = {
        "url": "https://cbic-gst.gov.in/gst-goods-services-rates.html",
        "domain": "cbic-gst.gov.in",
        "title": "Restaurant Services GST Rates - SAC 996331",
        "snippet": "Services provided by restaurants, eating joints including mess, canteen, neither having air-conditioning facility nor alcohol license under SAC 996331 attract 5% GST without ITC.",
    }
    cat, record, reason = classify_and_extract_evidence(
        chunk,
        user_query="GST rate for restaurant services without AC and without alcohol license?",
        discovered_hsn="996331",
    )
    assert cat == "rate"
    assert record is not None
    assert record["hsn_code"] == "996331"
    assert record["rate"] == "5%"
    assert record["condition"] == "Without ITC"
    assert record["category"] == "services"


def test_classify_and_extract_notification():
    """Verify statutory notification chunk classification."""
    chunk = {
        "url": "https://cbic-gst.gov.in/resources//htdocs-cbec/gst/notfctn-11-2017-cgst-rate-english.pdf",
        "domain": "cbic-gst.gov.in",
        "title": "Notification No. 11/2017-Central Tax (Rate)",
        "snippet": "Notification No. 11/2017-Central Tax (Rate), dated 28th June, 2017: Prescribes rates of central tax on supply of services under CGST Act, 2017.",
    }
    cat, record, reason = classify_and_extract_evidence(
        chunk,
        user_query="Notification 11/2017 details",
    )
    assert cat == "notification"
    assert record is not None
    assert "11/2017" in record["notification_number"]
    assert "Notification No. 11/2017" in record["content"]


def test_untrusted_domain_is_skipped():
    """Verify untrusted domains like Zomato/TripAdvisor are rejected with untrusted_domain reason."""
    chunk = {
        "url": "https://www.tripadvisor.in/Restaurant_Review-g304554-d12345.html",
        "domain": "tripadvisor.in",
        "title": "Best non AC restaurants with 5% discount",
        "snippet": "Great food and affordable 5% service at local dining.",
    }
    cat, record, reason = classify_and_extract_evidence(
        chunk,
        user_query="GST rate for restaurant services",
    )
    assert cat is None
    assert record is None
    assert "untrusted_domain" in reason


def test_schedule_background_web_ingestion_is_non_blocking():
    """Verify scheduling returns immediately in less than 50ms without blocking."""
    web_results = [
        {
            "url": "https://services.gst.gov.in/test",
            "domain": "services.gst.gov.in",
            "title": "Test GST Heading",
            "snippet": "Test snippet with 18% GST under HSN 8517.",
        }
    ]
    t0 = time.perf_counter()
    future = schedule_background_web_ingestion(
        execution_id="test_exec_async_123",
        user_query="What is the GST rate on mobile phones?",
        web_results=web_results,
        discovered_hsn="8517",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 50.0  # Must return immediately (<50ms)
    assert future is not None
    # Wait for completion in test
    result = future.result(timeout=10.0)
    assert result["status"] in ("stored", "skipped")


def test_end_to_end_rate_ingestion_and_subsequent_retrieval():
    """Verify newly ingested rate record is persisted to database & cache and is immediately retrieved locally."""
    test_hsn = f"8517{int(time.time() * 1000) % 10000:04d}"
    test_desc = f"Smartphones cellular networks test item {test_hsn}"
    chunk = {
        "url": "https://services.gst.gov.in/services/searchhsnsac",
        "domain": "services.gst.gov.in",
        "title": f"GST Rate on {test_desc} - HSN {test_hsn}",
        "snippet": f"Under HSN {test_hsn}, {test_desc} attract 18% GST.",
    }

    exec_id = f"test_e2e_{test_hsn}"
    trace_store.create_trace(exec_id, "What is the GST rate on smartphones?")

    # Execute ingestion
    result = execute_web_ingestion_job(
        execution_id=exec_id,
        user_query="What is the GST rate on smartphones?",
        web_results=[chunk],
        discovered_hsn=test_hsn,
    )
    assert result["status"] == "stored"
    assert result["records_ingested"] >= 1

    # Check trace_store updated
    trace = trace_store.get_trace(exec_id)
    assert trace is not None
    assert trace["background_ingestion"]["status"] == "stored"
    assert trace["background_ingestion"]["target_table"] == "gst_rates_2025"

    # Now verify subsequent local rate retrieval finds this new record without web search!
    local_hits = retrieve_rates(f"HSN {test_hsn}")
    assert len(local_hits) >= 1
    hit = local_hits[0]
    assert hit["hsn_code"] == test_hsn
    assert hit["rate"] == "18%"


def test_deduplication_skips_existing_record():
    """Verify that ingesting the exact same rate record again skips gracefully with already_exists."""
    test_hsn = f"8517{int(time.time() * 1000 + 77) % 10000:04d}"
    chunk = {
        "url": "https://services.gst.gov.in/services/searchhsnsac",
        "domain": "services.gst.gov.in",
        "title": f"GST Rate on Smartphones - HSN {test_hsn}",
        "snippet": f"Under HSN {test_hsn}, Smartphones attract 18% GST.",
    }
    # Run first time -> stored
    res1 = execute_web_ingestion_job(
        execution_id="test_exec_dup_001",
        user_query="What is the GST rate on smartphones?",
        web_results=[chunk],
        discovered_hsn=test_hsn,
    )
    assert res1["status"] == "stored"

    # Run second time -> skipped
    res2 = execute_web_ingestion_job(
        execution_id="test_exec_dup_002",
        user_query="What is the GST rate on smartphones?",
        web_results=[chunk],
        discovered_hsn=test_hsn,
    )
    assert res2["status"] == "skipped"
    assert res2["records_skipped"] >= 1
    assert res2["details"][0]["reason"] == "already_exists"
