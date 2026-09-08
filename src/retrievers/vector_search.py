"""pgvector semantic retrieval for GST descriptions."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.ingestors.ingest_gst import MODEL_NAME


def vector_search(conn, query, limit=10, model=None):
    """Return GST rows by vector distance, nearest matches first."""
    if limit < 1:
        raise ValueError("limit must be positive")
    if not query.strip():
        return []

    from pgvector.psycopg import register_vector
    from sentence_transformers import SentenceTransformer

    register_vector(conn)
    if model is None:
        model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode(query)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT code, description, embedding <=> %s AS distance,
                   cgst_rate_pct, sgst_utgst_rate_pct, igst_rate_pct, metadata
            FROM gst_documents
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query_embedding, query_embedding, limit),
        )
        return cur.fetchall()
