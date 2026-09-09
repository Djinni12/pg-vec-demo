"""Tests for Stage 3 Legal-Structure-First Notification Chunker."""

import json
from pathlib import Path
import pytest
from transformers import AutoTokenizer

from src.chunkers.notification_chunker import (
    DEFAULT_MAX_TOKENS,
    NotificationChunker,
    chunk_notifications_batch,
)

NORMALIZED_DIR = Path("data/notifications/normalized")


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture(scope="module")
def batch_results(tokenizer):
    return chunk_notifications_batch(NORMALIZED_DIR, tokenizer=tokenizer)


def test_batch_chunk_metrics(batch_results):
    chunks, report = batch_results
    assert report["total_chunks"] == 1495
    assert report["oversized_chunks_count"] == 0
    assert report["zero_token_chunks_count"] == 0
    assert report["max_tokens"] <= DEFAULT_MAX_TOKENS
    assert report["min_tokens"] > 0
    assert 50 <= report["avg_tokens"] <= 120


def test_chunking_17_2025(tokenizer):
    with open(NORMALIZED_DIR / "17-2025-CTR-eng.json") as f:
        data = json.load(f)
    chunker = NotificationChunker(tokenizer=tokenizer)
    chunks = chunker.chunk_notification(data)

    assert len(chunks) == 1
    ch = chunks[0]
    assert ch.token_count <= DEFAULT_MAX_TOKENS
    assert ch.metadata["notification_number"] == "17/2025-Central Tax (Rate)"
    assert ch.metadata["target_notification"] == "17/2017-Central Tax (Rate)"
    assert ch.metadata["operation_type"] == "INSERT"
    assert ch.metadata["effective_date"] == "2025-09-22"
    assert ch.metadata["chunk_strategy"] == "amendment_operation"
    assert "[Notification No. 17/2025-Central Tax (Rate)" in ch.text
    assert "local delivery" in ch.metadata["raw_legal_text"].lower()


def test_chunking_05_2025(tokenizer):
    with open(NORMALIZED_DIR / "ctr05-2025.json") as f:
        data = json.load(f)
    chunker = NotificationChunker(tokenizer=tokenizer)
    chunks = chunker.chunk_notification(data)

    assert len(chunks) == 6
    # Ops 1 & 2 inherit 2025-04-01
    assert chunks[0].metadata["effective_date"] == "2025-04-01"
    assert chunks[0].metadata["operation_type"] == "OMIT"
    assert chunks[1].metadata["effective_date"] == "2025-04-01"
    assert chunks[1].metadata["operation_type"] == "SUBSTITUTE"

    # Op 3 has effective_date is None
    assert chunks[2].metadata["effective_date"] is None
    assert chunks[2].metadata["operation_type"] == "INSERT"

    # Annexures VII, VIII, IX have effective_date is None (neutral narrative)
    for ch in chunks[3:6]:
        assert ch.metadata["chunk_type"] == "ANNEXURE_FORM"
        assert ch.metadata["chunk_strategy"] == "annexure_form"
        assert ch.metadata["effective_date"] is None
        assert "Annexure" in ch.text


def test_chunking_15_2025(tokenizer):
    with open(NORMALIZED_DIR / "15-2025-CTR-eng.json") as f:
        data = json.load(f)
    chunker = NotificationChunker(tokenizer=tokenizer)
    chunks = chunker.chunk_notification(data)

    assert len(chunks) == 41  # 39 operations, with 2 large operations split into 2 chunks each

    # Group 1 operations inherit 2025-09-22
    g1_chunks = [ch for ch in chunks if ch.metadata.get("scope_group_number") == 1]
    assert len(g1_chunks) > 0
    for ch in g1_chunks:
        assert ch.metadata["effective_date"] == "2025-09-22"
        assert ch.metadata["target_notification"] == "11/2017-Central Tax (Rate)"

    # Group 2 operation inherits 2025-04-01
    g2_chunks = [ch for ch in chunks if ch.metadata.get("scope_group_number") == 2]
    assert len(g2_chunks) == 1
    assert g2_chunks[0].metadata["effective_date"] == "2025-04-01"
    assert "premises" in g2_chunks[0].text.lower()

    # Split operations verified
    splits = [ch for ch in chunks if ch.metadata.get("chunk_strategy") == "amendment_token_split"]
    assert len(splits) == 4
    for sp in splits:
        assert sp.token_count <= DEFAULT_MAX_TOKENS


def test_chunking_13_2025(tokenizer):
    with open(NORMALIZED_DIR / "13-2025-CTR-eng.json") as f:
        data = json.load(f)
    chunker = NotificationChunker(tokenizer=tokenizer)
    chunks = chunker.chunk_notification(data)

    assert len(chunks) == 39  # 39 substituted table rows, each an atomic chunk
    for ch in chunks:
        assert ch.token_count <= DEFAULT_MAX_TOKENS
        assert ch.metadata["target_notification"] == "21/2018-Central Tax (Rate)"
        assert ch.metadata["effective_date"] == "2025-09-22"
        assert ch.metadata["operation_type"] == "SUBSTITUTE"
        assert ch.metadata["chunk_strategy"] == "table_row"
        assert ch.metadata["chunk_type"] == "AMENDMENT_TABLE_ROW"
        # Parent context must be preserved in text
        assert "In notification No. 21/2018-Central Tax (Rate)" in ch.text
        assert "for the Table and the entries relating thereto, the following shall be substituted" in ch.text

    # Row-level page provenance check: S. No. 1 on page 1, S. No. 5 on page 2, S. No. 26 on page 3
    assert chunks[0].metadata["source_page_start"] == 1
    assert chunks[0].metadata["source_page_end"] == 1
    assert chunks[4].metadata["source_page_start"] == 2
    assert chunks[4].metadata["source_page_end"] == 2
    assert chunks[25].metadata["source_page_start"] == 3
    assert chunks[25].metadata["source_page_end"] == 3
    distinct_pages = {(ch.metadata["source_page_start"], ch.metadata["source_page_end"]) for ch in chunks}
    assert distinct_pages == {(1, 1), (2, 2), (3, 3)}


def test_chunking_09_2025(tokenizer):
    with open(NORMALIZED_DIR / "09-2025-CTR-eng-2.json") as f:
        data = json.load(f)
    chunker = NotificationChunker(tokenizer=tokenizer)
    chunks = chunker.chunk_notification(data)

    assert len(chunks) == 1197  # 1,195 schedule entries + 2 explanation split chunks
    for ch in chunks:
        assert ch.token_count <= DEFAULT_MAX_TOKENS
        assert ch.token_count > 0

    # Verify all 1,195 schedule entries are present as individual atomic chunks
    sched_chunks = [ch for ch in chunks if ch.metadata.get("chunk_strategy") == "schedule_entry"]
    assert len(sched_chunks) == 1195
    schedules_represented = {ch.metadata["schedule"] for ch in sched_chunks}
    assert schedules_represented == {
        "Schedule I", "Schedule II", "Schedule III", "Schedule IV", "Schedule V", "Schedule VI", "Schedule VII"
    }


def test_chunking_10_2025(tokenizer):
    with open(NORMALIZED_DIR / "10-2025-CTR-eng.json") as f:
        data = json.load(f)
    chunker = NotificationChunker(tokenizer=tokenizer)
    chunks = chunker.chunk_notification(data)

    assert len(chunks) == 178  # 172 exemption entries + 4 annexures + 2 explanation splits

    # Check exemption schedule chunks
    sched_chunks = [ch for ch in chunks if ch.metadata.get("schedule") == "Exempt Goods Schedule"]
    assert len(sched_chunks) == 172
    for ch in sched_chunks:
        assert ch.metadata["tax_treatment"] == "EXEMPT"
        assert ch.metadata["schedule_rate_raw"] == "Nil"
        assert ch.metadata["chunk_strategy"] == "schedule_entry"
        # Never generate "0%" wording in context header or rate metadata
        assert "0%" not in ch.metadata["inherited_context_header"]
        assert "0 %" not in ch.metadata["inherited_context_header"]
        assert "Rate: 0%" not in ch.text
        assert ch.token_count <= DEFAULT_MAX_TOKENS

    # Check Annexures
    annex_chunks = [ch for ch in chunks if "ANNEXURE" in ch.metadata.get("chunk_type", "")]
    assert len(annex_chunks) == 4
    annex1_ch = [ch for ch in annex_chunks if "Annexure-I:" in ch.text]
    assert len(annex1_ch) == 1
    assert "113" in annex1_ch[0].metadata["serial_numbers"]

    annex2_ch = [ch for ch in annex_chunks if "Annexure-II:" in ch.text]
    assert len(annex2_ch) == 3
    assert "161" in annex2_ch[0].metadata["serial_numbers"]


def test_chunking_18_2025_excluded(tokenizer):
    with open(NORMALIZED_DIR / "18-2025-CTR-Eng.json") as f:
        data = json.load(f)
    chunker = NotificationChunker(tokenizer=tokenizer)
    chunks = chunker.chunk_notification(data)
    assert len(chunks) == 0
