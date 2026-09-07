"""Reciprocal Rank Fusion for independent GST retrieval results."""

from keyword_search import bm25_search
from vector_search import vector_search


def document_key(row):
    """Return the stable GST document key carried in row metadata."""
    metadata = row[6] or {}
    try:
        source_file = metadata["source_file"]
        source_row = metadata["source_row"]
    except KeyError as exc:
        raise ValueError("Result row metadata must include source_file and source_row") from exc
    return source_file, source_row


def reciprocal_rank_fusion(result_lists, k=60, top_k=5):
    """Fuse ranked result lists using only rank positions."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    fused = {}
    for rows in result_lists:
        seen = set()
        for rank, row in enumerate(rows, 1):
            key = document_key(row)
            if key in seen:
                continue
            seen.add(key)
            if key not in fused:
                fused[key] = {"row": row, "score": 0.0}
            fused[key]["score"] += 1 / (k + rank)

    ordered = sorted(
        fused.values(),
        key=lambda item: (-item["score"], document_key(item["row"])),
    )
    return [
        {
            "row": item["row"],
            "rrf_score": item["score"],
        }
        for item in ordered[:top_k]
    ]


def rerank_rows(query, rows, reranker):
    """Return rows ordered by cross-encoder relevance score descending."""
    if not rows:
        return []
    scores = reranker.predict([(query, row[1]) for row in rows])
    return [
        row
        for row, _ in sorted(
            zip(rows, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
    ]


def hybrid_rrf_search(
    conn,
    query,
    retrieve_limit=20,
    top_k=5,
    k=60,
    vector_model=None,
    vector_reranker=None,
):
    """Retrieve BM25 and vector candidates, optionally rerank vector, then fuse."""
    if retrieve_limit < 1:
        raise ValueError("retrieve_limit must be positive")
    bm25_rows = bm25_search(conn, query, retrieve_limit)
    vector_rows = vector_search(conn, query, retrieve_limit, model=vector_model)
    if vector_reranker is not None:
        vector_rows = rerank_rows(query, vector_rows, vector_reranker)
    return reciprocal_rank_fusion([bm25_rows, vector_rows], k=k, top_k=top_k)
