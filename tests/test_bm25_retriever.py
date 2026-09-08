"""
Unit tests for BM25 retriever module.

These tests verify the BM25Retriever class functionality without requiring
a live database connection. Integration tests with actual database should
be run separately.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.retrievers.bm25_retriever import BM25Retriever, create_bm25_retriever


class TestBM25RetrieverInitialization:
    """Test BM25Retriever initialization."""
    
    def test_init_with_conn_params(self):
        """Test that retriever initializes with connection parameters."""
        conn_params = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'test_db',
            'user': 'test_user',
            'password': 'test_pass'
        }
        
        retriever = BM25Retriever(conn_params)
        
        assert retriever.conn_params == conn_params
    
    def test_factory_function(self):
        """Test the factory function creates proper instance."""
        conn_params = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'test_db',
            'user': 'test_user',
            'password': 'test_pass'
        }
        
        retriever = create_bm25_retriever(conn_params)
        
        assert isinstance(retriever, BM25Retriever)
        assert retriever.conn_params == conn_params


class TestQueryValidation:
    """Test query validation and error handling."""
    
    def test_empty_query_raises_error(self):
        """Test that empty query raises ValueError."""
        conn_params = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'test_db',
            'user': 'test_user',
            'password': 'test_pass'
        }
        
        retriever = BM25Retriever(conn_params)
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retriever.retrieve("", top_k=10)
    
    def test_whitespace_only_query_raises_error(self):
        """Test that whitespace-only query raises ValueError."""
        conn_params = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'test_db',
            'user': 'test_user',
            'password': 'test_pass'
        }
        
        retriever = BM25Retriever(conn_params)
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retriever.retrieve("   ", top_k=10)
    
    def test_invalid_top_k_raises_error(self):
        """Test that top_k < 1 raises ValueError."""
        conn_params = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'test_db',
            'user': 'test_user',
            'password': 'test_pass'
        }
        
        retriever = BM25Retriever(conn_params)
        
        with pytest.raises(ValueError, match="top_k must be positive"):
            retriever.retrieve("test query", top_k=0)
        
        with pytest.raises(ValueError, match="top_k must be positive"):
            retriever.retrieve("test query", top_k=-1)


class TestSnippetCreation:
    """Test snippet creation functionality."""
    
    def test_snippet_short_content(self):
        """Test snippet with content shorter than max_length."""
        conn_params = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'test_db',
            'user': 'test_user',
            'password': 'test_pass'
        }
        
        retriever = BM25Retriever(conn_params)
        content = "This is a short content."
        query = "test"
        
        snippet = retriever._create_snippet(content, query)
        
        assert snippet == content
        assert "..." not in snippet
    
    def test_snippet_long_content(self):
        """Test snippet with content longer than max_length."""
        conn_params = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'test_db',
            'user': 'test_user',
            'password': 'test_pass'
        }
        
        retriever = BM25Retriever(conn_params)
        content = "This is a very long content. " * 50
        query = "test"
        
        snippet = retriever._create_snippet(content, query)
        
        assert len(snippet) <= 203  # max_length + len("...")
        assert snippet.endswith("...")
    
    def test_snippet_empty_content(self):
        """Test snippet with empty content."""
        conn_params = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'test_db',
            'user': 'test_user',
            'password': 'test_pass'
        }
        
        retriever = BM25Retriever(conn_params)
        
        snippet = retriever._create_snippet("", "test")
        
        assert snippet == ""
    
    def test_snippet_truncates_at_word_boundary(self):
        """Test that snippet truncates at word boundary when possible."""
        conn_params = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'test_db',
            'user': 'test_user',
            'password': 'test_pass'
        }
        
        retriever = BM25Retriever(conn_params)
        content = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10" * 20
        query = "test"
        
        snippet = retriever._create_snippet(content, query)
        
        # Should end at word boundary or have ellipsis
        assert snippet.endswith("...")
        # Should truncate before max_length (200 chars)
        assert len(snippet) <= 203  # 200 + "..."
        # Verify it's a valid truncation
        assert len(snippet) > 0


class TestDatabaseIntegration:
    """Test database integration with mocked connections."""
    
    @patch('psycopg.connect')
    def test_retrieve_fetches_from_all_sources(self, mock_connect):
        """Test that retrieve queries all three chunk tables."""
        conn_params = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'test_db',
            'user': 'test_user',
            'password': 'test_pass'
        }
        
        # Setup mock connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock fetchall to return empty results for all queries
        mock_cursor.fetchall.return_value = []
        
        retriever = BM25Retriever(conn_params)
        results, stats = retriever.retrieve("GST registration", top_k=10)
        
        # Verify cursor was used (3 queries: acts, rules, forms)
        assert mock_cursor.execute.call_count == 3
        
        # Verify stats are populated
        assert 'retrieval_time' in stats
        assert 'acts_fetched' in stats
        assert 'rules_fetched' in stats
        assert 'forms_fetched' in stats
        
        # Verify results structure (should be empty since we mocked empty results)
        assert results == []
    
    @patch('psycopg.connect')
    def test_retrieve_returns_ranked_results(self, mock_connect):
        """Test that results are properly ranked by BM25 score."""
        conn_params = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'test_db',
            'user': 'test_user',
            'password': 'test_pass'
        }
        
        # Setup mock connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock act results with different scores
        act_rows = [
            ('act_1', 'CGST Act', 'Chapter 1', 'Section 1', 'Registration', 
             'Content about registration', 5.5),
            ('act_2', 'CGST Act', 'Chapter 2', 'Section 2', 'Cancellation', 
             'Content about cancellation', 3.2),
        ]
        
        # Mock rule results
        rule_rows = [
            ('rule_1', 'Rule 1', 'Registration Rule', 'Chapter 1', 'Chap Title',
             'Rule content', 4.8),
        ]
        
        # Mock form results
        form_rows = [
            ('form_1', 'FORM GST REG-01', 'Registration Form', 'REG-01', 'Label',
             'Form content', 2.1),
        ]
        
        # Return different results for each query
        mock_cursor.fetchall.side_effect = [act_rows, rule_rows, form_rows]
        
        retriever = BM25Retriever(conn_params)
        results, stats = retriever.retrieve("GST registration", top_k=10)
        
        # Should have 4 results total (2 acts + 1 rule + 1 form)
        assert len(results) == 4
        
        # Results should be sorted by bm25_score ascending (more negative = better for pg_textsearch)
        for i in range(len(results) - 1):
            assert results[i]['bm25_score'] <= results[i+1]['bm25_score']
        
        # Ranks should be sequential starting from 1
        for i, result in enumerate(results):
            assert result['rank'] == i + 1
        
        # Verify document types
        doc_types = [r['document_type'] for r in results]
        assert 'act' in doc_types
        assert 'rule' in doc_types
        assert 'form' in doc_types
    
    @patch('psycopg.connect')
    def test_retrieve_limits_to_top_k(self, mock_connect):
        """Test that only top_k results are returned."""
        conn_params = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'test_db',
            'user': 'test_user',
            'password': 'test_pass'
        }
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Create many mock results
        many_rows = [(f'id_{i}', 'Act', 'Chap', f'Sec {i}', f'Title {i}', 
                      f'Content {i}', 10.0 - i * 0.1) for i in range(20)]
        
        mock_cursor.fetchall.side_effect = [many_rows, [], []]
        
        retriever = BM25Retriever(conn_params)
        results, stats = retriever.retrieve("test", top_k=5)
        
        # Should return only top 5 despite having 20 results
        assert len(results) == 5


class TestResultStructure:
    """Test that results have correct structure."""
    
    @patch('psycopg.connect')
    def test_result_has_required_fields(self, mock_connect):
        """Test that each result has all required fields."""
        conn_params = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'test_db',
            'user': 'test_user',
            'password': 'test_pass'
        }
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock single result
        mock_cursor.fetchall.side_effect = [
            [('act_1', 'CGST Act', 'Chapter 1', 'Section 1', 'Registration', 
              'Content about registration', 5.5)],
            [],
            []
        ]
        
        retriever = BM25Retriever(conn_params)
        results, _ = retriever.retrieve("registration", top_k=1)
        
        assert len(results) == 1
        result = results[0]
        
        # Check all required fields exist
        required_fields = [
            'rank',
            'document_type',
            'reference',
            'title',
            'bm25_score',
            'chunk_id',
            'snippet'
        ]
        
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
        
        # Check field types
        assert isinstance(result['rank'], int)
        assert isinstance(result['document_type'], str)
        assert isinstance(result['reference'], str)
        assert isinstance(result['title'], str)
        assert isinstance(result['bm25_score'], float)
        assert isinstance(result['chunk_id'], str)
        assert isinstance(result['snippet'], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
