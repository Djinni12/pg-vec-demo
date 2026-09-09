"""Generate dense embeddings for GST Central Tax (Rate) notification chunks.

Reuses the BAAI/bge-m3 model (1024 dimensions) and normalization approach
from src/embedders/act_embedder.py.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.embedders.act_embedder import (
    MODEL_NAME,
    encode_chunk_texts,
    find_empty_chunks,
    load_embedding_model,
)

load_dotenv()

DEFAULT_CHUNKS_JSON = Path("data/notifications/notification_chunks.json")
DEFAULT_EMBEDDINGS_JSONL = Path("data/notifications/notification_chunks_bge_m3_embeddings.jsonl")
EXPECTED_EMBEDDING_DIMENSION = 1024


def load_notification_chunks(path: Path = DEFAULT_CHUNKS_JSON) -> List[Dict[str, Any]]:
    """Load notification chunk dictionaries from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if not isinstance(chunks, list):
        raise ValueError(f"{path}: Expected chunk JSON root to be a list")
    return chunks


def load_existing_embeddings(path: Path = DEFAULT_EMBEDDINGS_JSONL) -> Dict[str, Dict[str, Any]]:
    """Load any already-generated embeddings to allow resumption."""
    if not path.exists():
        return {}
    existing: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cid = record.get("chunk_id")
            if cid:
                existing[cid] = record
    return existing


def generate_notification_embeddings(
    chunks_path: Path = DEFAULT_CHUNKS_JSON,
    output_path: Path = DEFAULT_EMBEDDINGS_JSONL,
    model_name: str = MODEL_NAME,
    batch_size: int = 8,
    normalize_embeddings: bool = True,
) -> Dict[str, Any]:
    """Generate dense embeddings for all notification chunks and persist to JSONL."""
    chunks = load_notification_chunks(chunks_path)

    # 1. Validation before embedding
    for c in chunks:
        cid = c.get("chunk_id", "")
        if "18-2025" in cid:
            raise ValueError(f"Excluded notification 18/2025 found in chunk: {cid}")

    empty = find_empty_chunks(chunks)
    if empty:
        raise ValueError(f"Found empty chunks without text: {empty[:5]}")

    existing_records = load_existing_embeddings(output_path)
    total_chunks = len(chunks)

    # Filter chunks that need embedding
    needed_chunks = [c for c in chunks if c["chunk_id"] not in existing_records]
    print(f"Total chunks: {total_chunks}")
    print(f"Already embedded: {len(existing_records)}")
    print(f"To embed: {len(needed_chunks)}")

    start_time = time.time()

    if needed_chunks:
        print(f"Loading embedding model: {model_name}...")
        model = load_embedding_model(model_name)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Open in append mode so existing records are kept
        with open(output_path, "a", encoding="utf-8") as out_file:
            for i in range(0, len(needed_chunks), batch_size):
                batch = needed_chunks[i : i + batch_size]
                texts = [b["text"] for b in batch]

                embeddings = encode_chunk_texts(
                    model,
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=normalize_embeddings,
                )

                for chunk, emb in zip(batch, embeddings):
                    dense_emb = emb.tolist() if hasattr(emb, "tolist") else [float(x) for x in emb]
                    if len(dense_emb) != EXPECTED_EMBEDDING_DIMENSION:
                        raise ValueError(
                            f"Wrong embedding dimension for {chunk['chunk_id']}: "
                            f"expected {EXPECTED_EMBEDDING_DIMENSION}, got {len(dense_emb)}"
                        )

                    rec = {
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "token_count": chunk["token_count"],
                        "metadata": chunk.get("metadata", {}),
                        "embedding": dense_emb,
                    }
                    existing_records[chunk["chunk_id"]] = rec
                    out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_file.flush()

                processed = min(i + batch_size, len(needed_chunks))
                pct = (processed / len(needed_chunks)) * 100
                print(f"Embedded {processed}/{len(needed_chunks)} ({pct:.1f}%)")

    elapsed_time = time.time() - start_time

    # Final assertion on full dataset
    final_records = load_existing_embeddings(output_path)
    if len(final_records) != total_chunks:
        raise ValueError(
            f"Expected {total_chunks} embeddings in {output_path}, but found {len(final_records)}"
        )

    for cid, rec in final_records.items():
        emb = rec.get("embedding", [])
        if len(emb) != EXPECTED_EMBEDDING_DIMENSION:
            raise ValueError(f"Chunk {cid} has wrong dimension: {len(emb)}")

    return {
        "model_name": model_name,
        "total_chunks": total_chunks,
        "total_embeddings": len(final_records),
        "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
        "time_seconds": round(elapsed_time, 2),
        "output_path": str(output_path),
    }
