"""Embed a query with BGE-M3 and LaBSE, then check compatibility with the stored corpus vectors.

This is query-only: it does not generate corpus embeddings, does not create tables,
and does not ingest anything into PostgreSQL.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv(dotenv_path=ROOT_DIR / ".env")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import SentenceTransformer

from act_embedder import MODEL_NAME as BGE_M3_MODEL

LABSE_MODEL = "sentence-transformers/LaBSE"
EXISTING_CORPUS_DIMENSION = 1024
DEFAULT_QUERIES = [
    "How do I register for GST?",
    "Which form is used for GST registration?",
    "What are the conditions for input tax credit?",
    "How can registration be cancelled?",
    "Which form is used to claim a refund?",
]


def embed_query(model: SentenceTransformer, query: str):
    start = time.perf_counter()
    vector = model.encode(query, normalize_embeddings=True, show_progress_bar=False)
    elapsed = time.perf_counter() - start
    return len(vector), elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Run one query instead of the built-in sample queries")
    args = parser.parse_args()

    queries = [args.query] if args.query else DEFAULT_QUERIES

    print("Loading query embedding models...")
    models = {
        "BGE-M3": SentenceTransformer(BGE_M3_MODEL),
        "LaBSE": SentenceTransformer(LABSE_MODEL),
    }
    print(f"Existing stored corpus embedding dimension: {EXISTING_CORPUS_DIMENSION}")
    print("Existing corpus tables: act_chunks, rule_chunks, form_chunks")
    print()

    for query in queries:
        print(f"Query: {query}")
        print("-" * (len(query) + 7))
        for label, model in models.items():
            dimension, elapsed = embed_query(model, query)
            compatible = dimension == EXISTING_CORPUS_DIMENSION
            status = "compatible" if compatible else "NOT compatible"
            print(
                f"{label}: query_dim={dimension} embed_time={elapsed:.3f}s "
                f"vs corpus_dim={EXISTING_CORPUS_DIMENSION} => {status}"
            )
        print()

    print("Note: LaBSE query vectors are 768-dimensional, so they cannot be searched against the existing BGE-M3 VECTOR(1024) corpus tables.")


if __name__ == "__main__":
    main()
