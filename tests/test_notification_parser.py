"""Tests for notification parser (Stage 1 revised)."""

from pathlib import Path
import pytest

from src.parsers.notification_parser import (
    canonicalize_notification_number,
    extract_effective_dates,
    extract_pdf_document,
    extract_statutory_authority,
    parse_indian_gazette_date,
)

NOTIFICATIONS_DIR = Path("data/notifications")


def test_parse_indian_gazette_date():
    """Test date parser handles various Gazette date formats."""
    assert parse_indian_gazette_date("17th September, 2025") == "2025-09-17"
    assert parse_indian_gazette_date("17th September 2025") == "2025-09-17"
    assert parse_indian_gazette_date("16th January, 2025.") == "2025-01-16"
    assert parse_indian_gazette_date("16 January, 2025") == "2025-01-16"
    assert parse_indian_gazette_date("31st December, 2025") == "2025-12-31"
    assert parse_indian_gazette_date("30th April, 2026") == "2026-04-30"
    assert parse_indian_gazette_date("1st day of April, 2025") == "2025-04-01"
    assert parse_indian_gazette_date("22nd day of September, 2025") == "2025-09-22"
    assert parse_indian_gazette_date("22nd September, 2025") == "2025-09-22"
    assert parse_indian_gazette_date("1st May, 2026") == "2026-05-01"
    assert parse_indian_gazette_date("") is None
    assert parse_indian_gazette_date(None) is None


def test_canonicalize_notification_number():
    """Test canonicalization of notification numbers."""
    assert canonicalize_notification_number("Notification No. 9/2025-Central Tax (Rate)") == "09/2025-Central Tax (Rate)"
    assert canonicalize_notification_number("Notification No. 17/2025-Central Tax (Rate)") == "17/2025-Central Tax (Rate)"
    assert canonicalize_notification_number("1/2017- Central Tax (Rate)") == "01/2017-Central Tax (Rate)"
    assert canonicalize_notification_number("15/2025-Central Tax (Rate)") == "15/2025-Central Tax (Rate)"


def test_extract_effective_dates_single():
    """Test extraction of a single general commencement date."""
    text = "2. This notification shall come into force with effect from the 22nd day of September, 2025."
    iso, raw, scopes = extract_effective_dates(text, "2025-09-17")
    assert iso == "2025-09-22"
    assert "22nd day of September, 2025" in raw
    assert len(scopes) == 0


def test_extract_effective_dates_immediate():
    """Test immediate effect commencement."""
    text = "2. This notification shall come into force with immediate effect."
    iso, raw, scopes = extract_effective_dates(text, "2025-01-16")
    assert iso == "2025-01-16"
    assert raw == "immediate effect"
    assert len(scopes) == 0


def test_extract_effective_dates_multi_scope_15_2025():
    """Test multi-scope effective dates as seen in 15/2025."""
    text = (
        "In the said notification,-\n"
        "(1) with effect from the 22nd day of September, 2025,-\n"
        "(a) in the Table, against serial number 3...\n"
        "(2) with effect from the 1st day of April, 2025, in paragraph 4, in clause (xxxvi)..."
    )
    iso, raw, scopes = extract_effective_dates(text, "2025-09-17")
    assert iso is None  # Deferred to scope groups; document-level is null
    assert "SCOPED_DATES" in raw
    assert len(scopes) == 2
    assert scopes[0]["group_number"] == 1
    assert scopes[0]["parsed_date"] == "2025-09-22"
    assert scopes[1]["group_number"] == 2
    assert scopes[1]["parsed_date"] == "2025-04-01"


def test_extract_statutory_authority():
    """Test parsing of multiple statutory sections invoked."""
    text = (
        "In exercise of the powers conferred by sub-sections (1), (3), and (4) of section 9, "
        "sub-sections (1) and (3) of section 11, sub-section (5) of section 15, sub-section (1) "
        "of section 16 and section 148 of the Central Goods and Services Tax Act, 2017 (12 of 2017), "
        "the Central Government hereby makes..."
    )
    raw, sections = extract_statutory_authority(text)
    assert raw is not None
    assert "Section 9(1)" in sections
    assert "Section 9(3)" in sections
    assert "Section 9(4)" in sections
    assert "Section 11(1)" in sections
    assert "Section 11(3)" in sections
    assert "Section 15(5)" in sections
    assert "Section 16(1)" in sections
    assert "Section 148" in sections


@pytest.mark.skipif(not (NOTIFICATIONS_DIR / "09-2025-CTR-eng-2.pdf").exists(), reason="PDF file not found")
def test_extract_09_2025():
    """Test extraction of master rate schedule 09/2025."""
    doc = extract_pdf_document(NOTIFICATIONS_DIR / "09-2025-CTR-eng-2.pdf")
    assert doc.extraction_status == "SUCCESS"
    assert doc.page_count == 56
    assert doc.total_char_count > 150_000
    assert doc.notification_number == "09/2025-Central Tax (Rate)"
    assert doc.notification_date == "2025-09-17"
    assert doc.effective_date == "2025-09-22"
    assert doc.superseded_notification == "01/2017-Central Tax (Rate)"
    assert doc.document_type == "RATE_SCHEDULE"
    assert "Section 9(1)" in doc.sections_invoked
    assert "Section 15(5)" in doc.sections_invoked


@pytest.mark.skipif(not (NOTIFICATIONS_DIR / "15-2025-CTR-eng.pdf").exists(), reason="PDF file not found")
def test_extract_15_2025():
    """Test extraction of multi-date amendment 15/2025."""
    doc = extract_pdf_document(NOTIFICATIONS_DIR / "15-2025-CTR-eng.pdf")
    assert doc.extraction_status == "SUCCESS"
    assert doc.page_count == 9
    assert doc.notification_number == "15/2025-Central Tax (Rate)"
    assert doc.notification_date == "2025-09-17"
    assert doc.effective_date is None  # Must NOT promote scoped date to document-level
    assert len(doc.effective_date_scopes) == 2
    assert doc.effective_date_scopes[0]["parsed_date"] == "2025-09-22"
    assert doc.effective_date_scopes[1]["parsed_date"] == "2025-04-01"
    assert doc.target_notification == "11/2017-Central Tax (Rate)"
    assert doc.document_type == "AMENDMENT"


@pytest.mark.skipif(not (NOTIFICATIONS_DIR / "ctr05-2025.pdf").exists(), reason="PDF file not found")
def test_extract_clause_scoped_date_05_and_06_2025():
    """Test that clause-level effective dates are NOT promoted to document level in 05/2025 and 06/2025."""
    doc05 = extract_pdf_document(NOTIFICATIONS_DIR / "ctr05-2025.pdf")
    assert doc05.extraction_status == "SUCCESS"
    assert doc05.effective_date is None  # NOT promoted to document level
    assert len(doc05.effective_date_scopes) == 1
    assert doc05.effective_date_scopes[0]["parsed_date"] == "2025-04-01"
    assert "paragraph 4" in doc05.effective_date_scopes[0]["context_snippet"]

    doc06 = extract_pdf_document(NOTIFICATIONS_DIR / "ctr06-2025.pdf")
    assert doc06.extraction_status == "SUCCESS"
    assert doc06.effective_date is None  # NOT promoted to document level
    assert len(doc06.effective_date_scopes) == 1
    assert doc06.effective_date_scopes[0]["parsed_date"] == "2025-04-01"
    assert "item (w)" in doc06.effective_date_scopes[0]["context_snippet"]


@pytest.mark.skipif(not (NOTIFICATIONS_DIR / "17-2025-CTR-eng.pdf").exists(), reason="PDF file not found")
def test_extract_17_2025():
    """Test extraction of targeted amendment 17/2025."""
    doc = extract_pdf_document(NOTIFICATIONS_DIR / "17-2025-CTR-eng.pdf")
    assert doc.extraction_status == "SUCCESS"
    assert doc.page_count == 1
    assert doc.notification_number == "17/2025-Central Tax (Rate)"
    assert doc.notification_date == "2025-09-17"
    assert doc.effective_date == "2025-09-22"
    assert doc.target_notification == "17/2017-Central Tax (Rate)"
    assert doc.document_type == "AMENDMENT"
    assert "Section 9(5)" in doc.sections_invoked


@pytest.mark.skipif(not (NOTIFICATIONS_DIR / "18-2025-CTR-Eng.pdf").exists(), reason="PDF file not found")
def test_extract_18_2025_vector_outline():
    """Test graceful handling of vector outline PDF 18/2025 without metadata fabrication."""
    doc = extract_pdf_document(NOTIFICATIONS_DIR / "18-2025-CTR-Eng.pdf")
    assert doc.extraction_status == "FAILED_ZERO_TEXT"
    assert doc.total_char_count == 0
    assert doc.vector_drawings_count > 1000
    assert doc.error_message is not None
    assert "NEEDS_VISUAL_EXTRACTION" in doc.error_message
    assert doc.notification_number is None  # No fabrication from filename
    assert doc.effective_date is None
