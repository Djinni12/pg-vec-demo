"""Generate dense embeddings for structure-aware CGST Act chunks."""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

MODEL_NAME = "BAAI/bge-m3"
DEFAULT_CHUNKS_JSON = Path("data/acts/central_gst_act_2017_chunks.json")
DEFAULT_EMBEDDINGS_JSONL = Path("data/acts/central_gst_act_2017_bge_m3_embeddings.jsonl")


def load_chunks(path=DEFAULT_CHUNKS_JSON):
    """Load chunk dictionaries from JSON."""
    with Path(path).open(encoding="utf-8") as handle:
        chunks = json.load(handle)
    if not isinstance(chunks, list):
        raise ValueError("Expected chunk JSON root to be a list")
    return chunks


def chunk_text(chunk):
    """Return the text field used for embedding."""
    return str(chunk.get("text") or "")


def find_empty_chunks(chunks):
    """Return chunk ids/positions whose embeddable text is empty."""
    empty = []
    for index, chunk in enumerate(chunks, 1):
        if not chunk_text(chunk).strip():
            empty.append(chunk.get("chunk_id") or f"#{index}")
    return empty


def load_embedding_model(model_name=MODEL_NAME):
    """Load the SentenceTransformer embedding model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def encode_chunk_texts(model, texts, batch_size=8, normalize_embeddings=True):
    """Generate dense embeddings for chunk texts."""
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize_embeddings,
    )


def embedding_to_list(embedding):
    """Convert numpy/list embeddings to plain JSON-serializable floats."""
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    return [float(value) for value in embedding]


def build_embedding_records(
    chunks,
    model=None,
    model_name=MODEL_NAME,
    batch_size=8,
    normalize_embeddings=True,
):
    """Return embedding records and a summary report for parsed chunks."""
    empty_chunks = find_empty_chunks(chunks)
    if empty_chunks:
        raise ValueError(f"Cannot embed empty chunks: {', '.join(empty_chunks)}")

    model = model or load_embedding_model(model_name)
    texts = [chunk_text(chunk) for chunk in chunks]
    embeddings = encode_chunk_texts(model, texts, batch_size=batch_size, normalize_embeddings=normalize_embeddings)

    if len(chunks) != len(embeddings):
        raise ValueError(f"chunk count {len(chunks)} != embedding count {len(embeddings)}")

    records = []
    embedding_dimension = 0
    for chunk, embedding in zip(chunks, embeddings):
        dense_embedding = embedding_to_list(embedding)
        if not dense_embedding:
            raise ValueError(f"Empty embedding for chunk {chunk.get('chunk_id')}")
        embedding_dimension = embedding_dimension or len(dense_embedding)
        if len(dense_embedding) != embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch for chunk {chunk.get('chunk_id')}: "
                f"expected {embedding_dimension}, got {len(dense_embedding)}"
            )
        records.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "text": chunk.get("text"),
                "token_count": chunk.get("token_count"),
                "metadata": chunk.get("metadata", {}),
                "embedding_model": model_name,
                "embedding_normalized": normalize_embeddings,
                "embedding_dimension": len(dense_embedding),
                "embedding": dense_embedding,
            }
        )

    report = {
        "chunk_count": len(chunks),
        "embedding_count": len(records),
        "embedding_dimension": embedding_dimension,
        "model_name": model_name,
        "embedding_normalized": normalize_embeddings,
        "failed_or_empty_chunks": empty_chunks,
    }
    if report["chunk_count"] != report["embedding_count"]:
        raise ValueError("chunk count != embedding count")
    return records, report


def write_jsonl(records, path):
    """Write one embedding record per line for inspection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(records, path):
    """Write embedding records as an indented JSON array."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
