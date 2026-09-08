"""Retrievers for GST legal document search."""

from .keyword_search import bm25_search, fuzzy_search, keyword_search, exact_lookup
from .vector_search import vector_search
from .rrf import reciprocal_rank_fusion, reciprocal_rank_fusion_by_key, rerank_rows, hybrid_rrf_search, document_key
from .legal_dense_retriever import (
    DenseSearchResult,
    dense_search_legal_corpus,
    embed_query,
    load_query_model,
    load_reranker_model,
    rerank_legal_results,
    search_act_chunks,
    search_rule_chunks,
    search_form_chunks,
    result_to_dict,
    candidate_counts,
)
from .legal_hybrid_retriever import (
    DEFAULT_RETRIEVE_LIMIT,
    DEFAULT_RRF_K,
    DEFAULT_TOP_K,
    chunk_key,
    fuse_legal_results,
    hybrid_search_legal_corpus,
)

__all__ = [
    "bm25_search",
    "fuzzy_search", 
    "keyword_search",
    "exact_lookup",
    "vector_search",
    "reciprocal_rank_fusion",
    "reciprocal_rank_fusion_by_key",
    "rerank_rows",
    "hybrid_rrf_search",
    "document_key",
    "DenseSearchResult",
    "dense_search_legal_corpus",
    "embed_query",
    "load_query_model",
    "load_reranker_model",
    "rerank_legal_results",
    "search_act_chunks",
    "search_rule_chunks",
    "search_form_chunks",
    "result_to_dict",
    "candidate_counts",
    "DEFAULT_RETRIEVE_LIMIT",
    "DEFAULT_RRF_K",
    "DEFAULT_TOP_K",
    "chunk_key",
    "fuse_legal_results",
    "hybrid_search_legal_corpus",
]
