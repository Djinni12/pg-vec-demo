"""Timescale pg_textsearch BM25 keyword retrieval with a trigram fallback."""


def bm25_search(conn, query, limit=5):
    """Return positive BM25 relevance scores, best matches first.

    pg_textsearch's <@> returns negative BM25 scores. Keep the ascending index
    order, exclude nonmatches, and negate scores for higher-is-better output.
    Explicit index context also supports sequential plans on small datasets.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT code, description,
                   -(search_text <@> to_bm25query(%s, 'gst_documents_search_bm25_idx')) AS score,
                   cgst_rate_pct, sgst_utgst_rate_pct, igst_rate_pct, metadata
            FROM gst_documents
            WHERE search_text <@> to_bm25query(%s, 'gst_documents_search_bm25_idx') < 0
            ORDER BY search_text <@> to_bm25query(%s, 'gst_documents_search_bm25_idx')
            LIMIT %s
            """,
            (query, query, query, limit),
        )
        return cur.fetchall()


def fuzzy_search(conn, query, limit=5):
    """Return descriptions passing the session's pg_trgm similarity threshold."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT code, description, similarity(description, %s) AS score,
                   cgst_rate_pct, sgst_utgst_rate_pct, igst_rate_pct, metadata
            FROM gst_documents
            WHERE description %% %s
            ORDER BY score DESC, code, description
            LIMIT %s
            """,
            (query, query, limit),
        )
        return cur.fetchall()


def keyword_search(conn, query, limit=5):
    """Return (rows, method); try trigrams only when BM25 returns no rows.

    Scores are specific to the returned method and are never mixed. Database
    errors propagate rather than being treated as a reason to fall back.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    if not query.strip():
        return [], "bm25"

    results = bm25_search(conn, query, limit)
    if results:
        return results, "bm25"
    return fuzzy_search(conn, query, limit), "trigram"


def exact_lookup(conn, code, limit=20):
    """Return all matching explicit HSN/SAC entries up to limit; no prefix inference."""
    import re

    normalized = re.sub(r"\s+", "", code)
    if not re.fullmatch(r"(?:\d{2}|\d{4}|\d{6}|\d{8})", normalized):
        raise ValueError("Use a 2, 4, 6, or 8 digit classification code")
    if limit < 1:
        raise ValueError("limit must be positive")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT code, description, NULL::double precision AS score,
                   cgst_rate_pct, sgst_utgst_rate_pct, igst_rate_pct, metadata
            FROM gst_documents
            WHERE exact_codes @> ARRAY[%s]::text[]
            ORDER BY source_file, source_row
            LIMIT %s
            """,
            (normalized, limit),
        )
        return cur.fetchall()
