"""Tests for Stage 2 Structural Parser."""

from pathlib import Path
import pytest

from src.parsers.notification_parser import extract_pdf_document
from src.parsers.structural_parser import (
    normalize_hsn_codes,
    normalize_notification,
)

NOTIFICATIONS_DIR = Path("data/notifications")


def test_normalize_hsn_codes():
    """Test safe HSN code normalization without fabrication."""
    assert normalize_hsn_codes("0101 21 00, 0101 29") == ["01012100", "010129"]
    assert normalize_hsn_codes("0202, 0203") == ["0202", "0203"]
    assert normalize_hsn_codes("6815") == ["6815"]
    assert normalize_hsn_codes("Any Chapter") == []
    assert normalize_hsn_codes("") == []


@pytest.mark.skipif(not (NOTIFICATIONS_DIR / "17-2025-CTR-eng.pdf").exists(), reason="PDF file not found")
def test_structural_parser_17_2025():
    """Test structural normalization of simple targeted amendment 17/2025."""
    doc = extract_pdf_document(NOTIFICATIONS_DIR / "17-2025-CTR-eng.pdf")
    norm = normalize_notification(doc)

    assert norm.normalization_status == "SUCCESS"
    assert norm.parser_type == "AMENDMENT_PARSER"
    assert len(norm.amendment_operations) == 1

    op = norm.amendment_operations[0]
    assert op["operation_type"] == "INSERT"
    assert op["target_notification"] == "17/2017-Central Tax (Rate)"
    assert op["effective_date"] == "2025-09-22"
    assert "local delivery" in op["explicit_new_text"].lower()


@pytest.mark.skipif(not (NOTIFICATIONS_DIR / "ctr05-2025.pdf").exists(), reason="PDF file not found")
def test_structural_parser_05_2025():
    """Test scoped-date amendment and annexures in 05/2025."""
    doc = extract_pdf_document(NOTIFICATIONS_DIR / "ctr05-2025.pdf")
    norm = normalize_notification(doc)

    assert norm.normalization_status == "SUCCESS"
    assert norm.parser_type == "AMENDMENT_PARSER"
    assert len(norm.annexures) == 3
    assert {a["annexure_title"] for a in norm.annexures} == {"Annexure VII", "Annexure VIII", "Annexure IX"}

    assert len(norm.amendment_operations) == 3
    # Operations 1 and 2 relate to explanation w.e.f. 2025-04-01
    op1 = norm.amendment_operations[0]
    assert op1["operation_type"] == "OMIT"
    assert "clause (xxxv)" in op1["target_clause_item"]
    assert op1["effective_date"] == "2025-04-01"

    op2 = norm.amendment_operations[1]
    assert op2["operation_type"] == "SUBSTITUTE"
    assert "clause (xxxvi)" in op2["target_clause_item"]
    assert op2["effective_date"] == "2025-04-01"

    # Operation 3 inserts Annexures
    op3 = norm.amendment_operations[2]
    assert op3["operation_type"] == "INSERT"
    assert op3["effective_date"] is None


@pytest.mark.skipif(not (NOTIFICATIONS_DIR / "15-2025-CTR-eng.pdf").exists(), reason="PDF file not found")
def test_structural_parser_15_2025():
    """Test multi-date complex amendment 15/2025."""
    doc = extract_pdf_document(NOTIFICATIONS_DIR / "15-2025-CTR-eng.pdf")
    norm = normalize_notification(doc)

    assert norm.normalization_status == "SUCCESS"
    assert norm.parser_type == "AMENDMENT_PARSER"
    assert len(norm.amendment_operations) >= 15

    # Check Group 1 operations inherit 2025-09-22
    g1_ops = [op for op in norm.amendment_operations if op.get("scope_group_number") == 1]
    assert len(g1_ops) > 0
    for op in g1_ops:
        assert op["effective_date"] == "2025-09-22"

    # Check Group 2 operations inherit 2025-04-01
    g2_ops = [op for op in norm.amendment_operations if op.get("scope_group_number") == 2]
    assert len(g2_ops) > 0
    for op in g2_ops:
        assert op["effective_date"] == "2025-04-01"


@pytest.mark.skipif(not (NOTIFICATIONS_DIR / "13-2025-CTR-eng.pdf").exists(), reason="PDF file not found")
def test_structural_parser_13_2025():
    """Test whole-table substitution in 13/2025."""
    doc = extract_pdf_document(NOTIFICATIONS_DIR / "13-2025-CTR-eng.pdf")
    norm = normalize_notification(doc)

    assert norm.normalization_status == "SUCCESS"
    assert norm.parser_type == "AMENDMENT_TABLE_PARSER"
    assert len(norm.amendment_operations) == 1

    op = norm.amendment_operations[0]
    assert op["operation_type"] == "SUBSTITUTE"
    assert op["target_schedule"] == "Table"
    assert op["effective_date"] == "2025-09-22"
    assert op["table_data"] is not None
    assert len(op["table_data"]) == 39
    assert op["table_data"][0]["serial_no"] == "1"
    assert op["table_data"][0]["classification_raw"] == "3406"
    assert op["table_data"][0]["rate_raw"] == "2.5 %"


@pytest.mark.skipif(not (NOTIFICATIONS_DIR / "09-2025-CTR-eng-2.pdf").exists(), reason="PDF file not found")
def test_structural_parser_09_2025():
    """Test comprehensive 7-schedule rate notification 09/2025."""
    doc = extract_pdf_document(NOTIFICATIONS_DIR / "09-2025-CTR-eng-2.pdf")
    norm = normalize_notification(doc)

    assert norm.normalization_status == "SUCCESS"
    assert norm.parser_type == "SCHEDULE_RATE_PARSER"
    assert len(norm.schedules) == 7

    expected_rates = {
        "Schedule I": 2.5,
        "Schedule II": 9.0,
        "Schedule III": 20.0,
        "Schedule IV": 1.5,
        "Schedule V": 0.125,
        "Schedule VI": 0.75,
        "Schedule VII": 14.0,
    }

    for sched_name, rate in expected_rates.items():
        assert sched_name in norm.schedules
        entries = norm.schedules[sched_name]
        assert len(entries) > 0
        assert entries[0]["schedule_rate_pct"] == rate

    # Check multi-page row continuation for entry 9
    sch1 = norm.schedules["Schedule I"]
    entry9 = [e for e in sch1 if e["serial_no"] == "9"][0]
    assert entry9["page_start"] == 1
    assert entry9["page_end"] == 2
    assert "steaming or by boiling" in entry9["description"]

    # Structural sequence validation across all 7 schedules
    expected_counts = {
        "Schedule I": 516,
        "Schedule II": 640,
        "Schedule III": 13,
        "Schedule IV": 15,
        "Schedule V": 3,
        "Schedule VI": 2,
        "Schedule VII": 6,
    }
    total_entries = 0
    for sched_name, expected_count in expected_counts.items():
        entries = norm.schedules[sched_name]
        assert len(entries) == expected_count
        total_entries += len(entries)

        snos = [e["serial_no"] for e in entries]
        # 1. No duplicate serial numbers
        assert len(snos) == len(set(snos)), f"Duplicate serial numbers found in {sched_name}"
        # 2. Continuous sequence from 1 to expected_count without gaps
        numeric_snos = [int(s) for s in snos if s.isdigit()]
        assert len(numeric_snos) == expected_count, f"Non-numeric or malformed serial numbers in {sched_name}"
        assert min(numeric_snos) == 1, f"{sched_name} does not start at 1"
        assert max(numeric_snos) == expected_count, f"{sched_name} does not reach {expected_count}"
        assert set(numeric_snos) == set(range(1, expected_count + 1)), f"Sequence gaps in {sched_name}"

    assert total_entries == 1195


@pytest.mark.skipif(not (NOTIFICATIONS_DIR / "10-2025-CTR-eng.pdf").exists(), reason="PDF file not found")
def test_structural_parser_10_2025():
    """Test exemption schedule and annexures in 10/2025."""
    doc = extract_pdf_document(NOTIFICATIONS_DIR / "10-2025-CTR-eng.pdf")
    norm = normalize_notification(doc)

    assert norm.normalization_status == "SUCCESS"
    assert norm.parser_type == "EXEMPTION_PARSER"
    assert "Exempt Goods Schedule" in norm.schedules
    entries = norm.schedules["Exempt Goods Schedule"]
    assert len(entries) == 172

    # Structural sequence validation for exemption schedule
    snos = [e["serial_no"] for e in entries]
    assert len(snos) == len(set(snos)), "Duplicate serial numbers found in Exemption Schedule"
    numeric_snos = [int(s) for s in snos if s.isdigit()]
    assert len(numeric_snos) == 172
    assert min(numeric_snos) == 1
    assert max(numeric_snos) == 172
    assert set(numeric_snos) == set(range(1, 173)), "Sequence gaps in Exemption Schedule"

    # Exemption semantics verification
    assert entries[0]["tax_treatment"] == "EXEMPT"
    assert entries[0]["schedule_rate_raw"] == "Nil"
    assert entries[0]["schedule_rate_pct"] is None

    # Annexure validation
    assert len(norm.annexures) == 2
    annex1 = norm.annexures[0]
    assert "Annexure-I" in annex1["annexure_title"]
    assert len(annex1["items"]) == 37
    annex1_nos = [int(i["item_no"]) for i in annex1["items"] if i.get("item_no", "").isdigit()]
    assert set(annex1_nos) == set(range(1, 38))

    annex2 = norm.annexures[1]
    assert "Annexure-II" in annex2["annexure_title"]
    assert len(annex2["items"]) == 134
    annex2_nos = [int(i["item_no"]) for i in annex2["items"] if i.get("item_no", "").isdigit()]
    assert set(annex2_nos) == set(range(1, 135))

    # Cross-reference verification
    entry_113 = [e for e in entries if e["serial_no"] == "113"][0]
    assert "Annexure I" in entry_113["description"]
    assert annex1["related_serial_no"] == "113"

    entry_161 = [e for e in entries if e["serial_no"] == "161"][0]
    assert "Annexure II" in entry_161["description"]
    assert annex2["related_serial_no"] == "161"


@pytest.mark.skipif(not (NOTIFICATIONS_DIR / "18-2025-CTR-Eng.pdf").exists(), reason="PDF file not found")
def test_structural_parser_18_2025_excluded():
    """Test that 18/2025 is excluded from normal structural parsing."""
    doc = extract_pdf_document(NOTIFICATIONS_DIR / "18-2025-CTR-Eng.pdf")
    norm = normalize_notification(doc)

    assert norm.parser_type == "EXCLUDED_ZERO_TEXT"
    assert norm.normalization_status == "NEEDS_VISUAL_EXTRACTION"
    assert len(norm.amendment_operations) == 0
    assert len(norm.schedules) == 0


def test_stage1_stage2_identity_consistency():
    """Test that Stage 1 and Stage 2 document identities match 100% across all files."""
    for p in sorted(NOTIFICATIONS_DIR.glob("*.pdf")):
        doc = extract_pdf_document(p)
        norm = normalize_notification(doc)

        assert doc.notification_number == norm.notification_number, f"Mismatch in notif_number for {p.name}"
        assert doc.notification_date == norm.notification_date, f"Mismatch in notif_date for {p.name}"
        assert doc.effective_date == norm.document_effective_date, f"Mismatch in effective_date for {p.name}"
        assert doc.file_name == norm.file_name, f"Mismatch in file_name for {p.name}"

        # Target notification consistency
        if norm.amendment_operations and norm.parser_type not in ("EXCLUDED_ZERO_TEXT", "RATE_SCHEDULE", "EXEMPTION"):
            assert doc.target_notification == norm.amendment_operations[0].get("target_notification")

    # Specific check for CTR-E-updated-1.pdf
    ctr_e_doc = extract_pdf_document(NOTIFICATIONS_DIR / "CTR-E-updated-1.pdf")
    assert ctr_e_doc.notification_number == "01/2026-Central Tax (Rate)"
    assert ctr_e_doc.notification_date == "2026-04-30"
    assert ctr_e_doc.effective_date == "2026-05-01"
    assert ctr_e_doc.target_notification == "09/2025-Central Tax (Rate)"
