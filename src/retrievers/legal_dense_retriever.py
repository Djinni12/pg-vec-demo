"""Unified dense retrieval over GST legal Act, Rule, and Form chunk tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.embedders.act_embedder import MODEL_NAME

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class DenseSearchResult:
    document_type: str
    chunk_id: str
    reference: str
    title: str
    content: str
    source_metadata: dict
    score: float
    distance: float


def load_query_model(model_name: str = MODEL_NAME):
    """Load the BGE-M3 SentenceTransformer used by the legal chunk embeddings."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_query(query: str, model=None, model_name: str = MODEL_NAME):
    """Embed a query with normalized BGE-M3 dense vectors for cosine search."""
    if model is None:
        model = load_query_model(model_name)
    return model.encode(query, normalize_embeddings=True)


def _fetch_rows(conn, sql: str, query_embedding, limit: int):
    with conn.cursor() as cur:
        cur.execute(sql, (query_embedding, query_embedding, limit))
        return cur.fetchall()


def search_act_chunks(conn, query_embedding, limit: int):
    rows = _fetch_rows(
        conn,
        """
        SELECT
            chunk_id,
            COALESCE(section_number, '') AS reference,
            COALESCE(section_title, '') AS title,
            content,
            embedding <=> %s AS distance,
            jsonb_build_object(
                'act_name', act_name,
                'chapter', chapter,
                'section_number', section_number,
                'section_title', section_title,
                'subsection_numbers', subsection_numbers,
                'status', status,
                'token_count', token_count
            ) AS source_metadata
        FROM act_chunks
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        query_embedding,
        limit,
    )
    return [
        DenseSearchResult(
            document_type="act",
            chunk_id=row[0],
            reference=f"Section {row[1]}".strip(),
            title=row[2],
            content=row[3],
            distance=float(row[4]),
            score=1.0 - float(row[4]),
            source_metadata=row[5],
        )
        for row in rows
    ]


def search_rule_chunks(conn, query_embedding, limit: int):
    rows = _fetch_rows(
        conn,
        """
        SELECT
            chunk_id,
            COALESCE(rule_number, '') AS reference,
            COALESCE(rule_title, '') AS title,
            content,
            embedding <=> %s AS distance,
            jsonb_build_object(
                'chapter', chapter,
                'chapter_title', chapter_title,
                'rule_number', rule_number,
                'rule_title', rule_title,
                'subrule_numbers', subrule_numbers,
                'status', status,
                'token_count', token_count,
                'chunk_strategy', chunk_strategy
            ) AS source_metadata
        FROM rule_chunks
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        query_embedding,
        limit,
    )
    return [
        DenseSearchResult(
            document_type="rule",
            chunk_id=row[0],
            reference=f"Rule {row[1]}".strip(),
            title=row[2],
            content=row[3],
            distance=float(row[4]),
            score=1.0 - float(row[4]),
            source_metadata=row[5],
        )
        for row in rows
    ]


def search_form_chunks(conn, query_embedding, limit: int):
    rows = _fetch_rows(
        conn,
        """
        SELECT
            chunk_id,
            COALESCE(form_number, '') AS reference,
            COALESCE(title, form_title, '') AS title,
            content,
            embedding <=> %s AS distance,
            jsonb_build_object(
                'form_uid', form_uid,
                'form_number', form_number,
                'form_family', form_family,
                'form_code', form_code,
                'title', COALESCE(title, form_title, ''),
                'language', language,
                'rule_references', rule_references,
                'part_number', part_number,
                'section_label', section_label,
                'source_start_page', source_start_page,
                'source_end_page', source_end_page,
                'token_count', token_count,
                'chunk_strategy', chunk_strategy
            ) AS source_metadata
        FROM form_chunks
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        query_embedding,
        limit,
    )
    return [
        DenseSearchResult(
            document_type="form",
            chunk_id=row[0],
            reference=row[1],
            title=row[2],
            content=row[3],
            distance=float(row[4]),
            score=1.0 - float(row[4]),
            source_metadata=row[5],
        )
        for row in rows
    ]


def result_to_dict(result: DenseSearchResult):
    return {
        "document_type": result.document_type,
        "chunk_id": result.chunk_id,
        "reference": result.reference,
        "title": result.title,
        "content": result.content,
        "source_metadata": result.source_metadata,
        "score": result.score,
        "distance": result.distance,
    }


def load_reranker_model(model_name: str = DEFAULT_RERANKER_MODEL):
    """Load the CrossEncoder reranker used after dense candidate retrieval."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rerank_legal_results(query: str, results: list[dict], reranker=None, model_name: str = DEFAULT_RERANKER_MODEL):
    """Return legal dense results reordered by CrossEncoder pair scores.

    The reranker scores only the already-retrieved candidate pool. It does not query
    PostgreSQL and cannot recover chunks that dense retrieval did not return.
    """
    if not results:
        return []
    if reranker is None:
        reranker = load_reranker_model(model_name)

    pairs = [(query, result.get("content") or "") for result in results]
    scores = reranker.predict(pairs)
    reranked = []
    for result, score in zip(results, scores):
        updated = dict(result)
        if "dense_score" not in updated:
            updated["dense_score"] = result.get("score")
        updated["rerank_score"] = float(score)
        updated["score"] = float(score)
        reranked.append(updated)
    return sorted(reranked, key=lambda result: result["rerank_score"], reverse=True)


def candidate_counts(results: Iterable[DenseSearchResult]):
    counts = {"act": 0, "rule": 0, "form": 0}
    for result in results:
        counts[result.document_type] = counts.get(result.document_type, 0) + 1
    return counts


def dense_search_legal_corpus(
    conn,
    query: str,
    top_k: int = 10,
    per_table_limit: int | None = None,
    model=None,
    reranker=None,
    rerank: bool = False,
):
    """Search Act, Rule, and Form chunks together and optionally rerank candidates."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if per_table_limit is None:
        per_table_limit = top_k
    if per_table_limit < 1:
        raise ValueError("per_table_limit must be positive")
    if not query.strip():
        return [], {"act": 0, "rule": 0, "form": 0}

    from pgvector.psycopg import register_vector

    register_vector(conn)
    query_embedding = embed_query(query, model=model)
    candidates = [
        *search_act_chunks(conn, query_embedding, per_table_limit),
        *search_rule_chunks(conn, query_embedding, per_table_limit),
        *search_form_chunks(conn, query_embedding, per_table_limit),
    ]
    counts = candidate_counts(candidates)
    ranked = sorted(candidates, key=lambda result: result.score, reverse=True)
    result_dicts = [result_to_dict(result) for result in ranked]
    if rerank or reranker is not None:
        result_dicts = rerank_legal_results(query, result_dicts, reranker=reranker)
    return result_dicts[:top_k], counts
