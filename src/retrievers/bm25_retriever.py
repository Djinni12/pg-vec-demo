"""
PostgreSQL-native BM25 retrieval for GST legal knowledge base using pg_search (ParadeDB).

This module provides true BM25 lexical search across:
- act_chunks (GST Act sections)
- rule_chunks (CGST Rules)
- form_chunks (GST Forms)

Requires pg_search extension to be installed and enabled in PostgreSQL.
"""

import time
from typing import List, Dict, Any, Tuple, Optional
import psycopg


class BM25Retriever:
    """
    PostgreSQL-native BM25 retriever using pg_search (ParadeDB).
    
    Performs lexical search across legal documents using the BM25 algorithm,
    which is superior to TF-IDF for information retrieval tasks.
    """
    
    def __init__(self, conn_params: Dict[str, Any]):
        """
        Initialize the BM25 retriever.
        
        Args:
            conn_params: Database connection parameters including:
                - host: PostgreSQL host
                - port: PostgreSQL port
                - dbname: Database name
                - user: Username
                - password: Password
        """
        self.conn_params = conn_params
    
    def _get_connection(self):
        """Create a new database connection."""
        return psycopg.connect(**self.conn_params)
    
    def _build_query(self, query: str) -> str:
        """
        Build a BM25 search query string.
        
        Args:
            query: User's natural language query
            
        Returns:
            Formatted query string for pg_search
        """
        # Simple tokenization - pg_search handles stemming and normalization
        return query.strip()
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 10
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Retrieve top-k results using BM25 across all chunk types.
        
        Query flow:
        1. Search act_chunks → top 10
        2. Search rule_chunks → top 10  
        3. Search form_chunks → top 10
        4. Normalize results into common structure
        5. Globally rank by BM25 score
        6. Return final top 10
        
        Args:
            query: Natural language query string
            top_k: Number of results to return (default: 10)
            
        Returns:
            Tuple of (results, stats) where:
                - results: List of result dictionaries with keys:
                    - rank: Final rank after global sorting
                    - document_type: Type of document (act/rule/form)
                    - reference: Section/rule/form reference number
                    - title: Title of the section/rule/form
                    - bm25_score: BM25 relevance score (higher is better)
                    - chunk_id: Unique chunk identifier
                    - snippet: Relevant text excerpt
                - stats: Dictionary with retrieval statistics:
                    - retrieval_time: Total query time in seconds
                    - acts_fetched: Number of act candidates retrieved
                    - rules_fetched: Number of rule candidates retrieved
                    - forms_fetched: Number of form candidates retrieved
                    
        Raises:
            ValueError: If query is empty or top_k < 1
            psycopg.Error: If database error occurs
        """
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        start_time = time.time()
        search_query = self._build_query(query)
        
        all_results = []
        stats = {
            'retrieval_time': 0.0,
            'acts_fetched': 0,
            'rules_fetched': 0,
            'forms_fetched': 0
        }
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Search act_chunks
                    act_results = self._search_acts(cur, search_query, top_k)
                    all_results.extend(act_results)
                    stats['acts_fetched'] = len(act_results)
                    
                    # Search rule_chunks
                    rule_results = self._search_rules(cur, search_query, top_k)
                    all_results.extend(rule_results)
                    stats['rules_fetched'] = len(rule_results)
                    
                    # Search form_chunks
                    form_results = self._search_forms(cur, search_query, top_k)
                    all_results.extend(form_results)
                    stats['forms_fetched'] = len(form_results)
                    
                    # Globally rank by BM25 score and take top_k
                    all_results.sort(key=lambda x: x['bm25_score'], reverse=True)
                    final_results = all_results[:top_k]
                    
                    # Update ranks after global sorting
                    for i, result in enumerate(final_results):
                        result['rank'] = i + 1
                        
        except psycopg.Error as e:
            raise psycopg.Error(f"Database error during BM25 retrieval: {e}")
        
        stats['retrieval_time'] = time.time() - start_time
        return final_results, stats
    
    def _search_acts(
        self, 
        cur: psycopg.Cursor, 
        query: str, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Search act_chunks using BM25.
        
        Args:
            cur: Database cursor
            query: Formatted search query
            limit: Maximum number of results
            
        Returns:
            List of result dictionaries
        """
        sql = """
            SELECT 
                chunk_id,
                act_name,
                chapter,
                section_number,
                section_title,
                content,
                paradedb.score(chunk_id) AS bm25_score
            FROM act_chunks
            WHERE content @@@ %s
               OR section_title @@@ %s
               OR section_number @@@ %s
               OR chapter @@@ %s
               OR act_name @@@ %s
            ORDER BY bm25_score DESC
            LIMIT %s
        """
        
        cur.execute(sql, (query, query, query, query, query, limit))
        rows = cur.fetchall()
        
        results = []
        for row in rows:
            (chunk_id, act_name, chapter, section_number, 
             section_title, content, score) = row
            
            # Build reference from section info
            reference = section_number or ""
            if chapter:
                reference = f"{chapter}, {reference}" if reference else chapter
            
            results.append({
                'rank': 0,  # Will be updated after global ranking
                'document_type': 'act',
                'reference': reference,
                'title': section_title or act_name,
                'bm25_score': float(score) if score is not None else 0.0,
                'chunk_id': chunk_id,
                'snippet': self._create_snippet(content, query)
            })
        
        return results
    
    def _search_rules(
        self, 
        cur: psycopg.Cursor, 
        query: str, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Search rule_chunks using BM25.
        
        Args:
            cur: Database cursor
            query: Formatted search query
            limit: Maximum number of results
            
        Returns:
            List of result dictionaries
        """
        sql = """
            SELECT 
                chunk_id,
                rule_number,
                rule_title,
                chapter,
                chapter_title,
                content,
                paradedb.score(chunk_id) AS bm25_score
            FROM rule_chunks
            WHERE content @@@ %s
               OR rule_title @@@ %s
               OR rule_number @@@ %s
               OR chapter @@@ %s
               OR chapter_title @@@ %s
            ORDER BY bm25_score DESC
            LIMIT %s
        """
        
        cur.execute(sql, (query, query, query, query, query, limit))
        rows = cur.fetchall()
        
        results = []
        for row in rows:
            (chunk_id, rule_number, rule_title, chapter, 
             chapter_title, content, score) = row
            
            # Build reference from rule info
            reference = rule_number or ""
            if chapter:
                reference = f"{chapter}, {reference}" if reference else chapter
            
            results.append({
                'rank': 0,
                'document_type': 'rule',
                'reference': reference,
                'title': rule_title or chapter_title or "Rule",
                'bm25_score': float(score) if score is not None else 0.0,
                'chunk_id': chunk_id,
                'snippet': self._create_snippet(content, query)
            })
        
        return results
    
    def _search_forms(
        self, 
        cur: psycopg.Cursor, 
        query: str, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Search form_chunks using BM25.
        
        Args:
            cur: Database cursor
            query: Formatted search query
            limit: Maximum number of results
            
        Returns:
            List of result dictionaries
        """
        sql = """
            SELECT 
                chunk_id,
                form_number,
                form_title,
                form_uid,
                section_label,
                content,
                paradedb.score(chunk_id) AS bm25_score
            FROM form_chunks
            WHERE content @@@ %s
               OR title @@@ %s
               OR form_title @@@ %s
               OR form_number @@@ %s
               OR section_label @@@ %s
            ORDER BY bm25_score DESC
            LIMIT %s
        """
        
        cur.execute(sql, (query, query, query, query, query, limit))
        rows = cur.fetchall()
        
        results = []
        for row in rows:
            (chunk_id, form_number, form_title, form_uid, 
             section_label, content, score) = row
            
            # Build reference from form info
            reference = form_number or form_uid or ""
            
            results.append({
                'rank': 0,
                'document_type': 'form',
                'reference': reference,
                'title': form_title or section_label or "Form",
                'bm25_score': float(score) if score is not None else 0.0,
                'chunk_id': chunk_id,
                'snippet': self._create_snippet(content, query)
            })
        
        return results
    
    def _create_snippet(
        self, 
        content: str, 
        query: str, 
        max_length: int = 200
    ) -> str:
        """
        Create a relevant snippet from content.
        
        Extracts a portion of the content that contains query terms,
        providing context for the search result.
        
        Args:
            content: Full text content
            query: Original search query
            max_length: Maximum snippet length in characters
            
        Returns:
            Snippet string with ellipsis if truncated
        """
        if not content:
            return ""
        
        # Simple approach: return beginning of content
        # Could be enhanced to find query term proximity
        if len(content) <= max_length:
            return content
        
        # Truncate and add ellipsis
        # Find last complete word before max_length
        truncate_pos = content.rfind(' ', 0, max_length)
        if truncate_pos == -1:
            truncate_pos = max_length
        
        return content[:truncate_pos] + "..."


def create_bm25_retriever(conn_params: Dict[str, Any]) -> BM25Retriever:
    """
    Factory function to create a BM25Retriever instance.
    
    Args:
        conn_params: Database connection parameters
        
    Returns:
        Configured BM25Retriever instance
    """
    return BM25Retriever(conn_params)
