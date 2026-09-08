#!/usr/bin/env python3
"""
Diagnostic test script for BM25 retrieval.

This script tests the BM25 retriever with a sample query against a live database.
Requires:
- PostgreSQL with pg_search extension installed
- Database populated with act_chunks, rule_chunks, and form_chunks tables
- BM25 indexes created via migration/add_bm25_indexes.sql

Usage:
    python scripts/test_bm25_diagnostic.py
    
Environment variables (optional):
    POSTGRES_HOST: Database host (default: localhost)
    POSTGRES_PORT: Database port (default: 5432)
    POSTGRES_DB: Database name (default: hybrid_rag)
    POSTGRES_USER: Database user (default: postgres)
    POSTGRES_PASSWORD: Database password (default: postgres)
"""

import os
import sys
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.retrievers.bm25_retriever import BM25Retriever


def print_separator(char: str = "=", length: int = 100):
    """Print a separator line."""
    print(char * length)


def print_result(result: dict, index: int):
    """Print a single search result in formatted way."""
    print(f"\n{index}. [{result['document_type'].upper()}] {result['title']}")
    print(f"   Reference: {result['reference']}")
    print(f"   Chunk ID: {result['chunk_id']}")
    print(f"   BM25 Score: {result['bm25_score']:.4f}")
    print(f"   Snippet: {result['snippet']}")


def main():
    """Run diagnostic BM25 retrieval test."""
    
    # Get connection parameters from environment or use defaults
    conn_params = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': int(os.getenv('POSTGRES_PORT', 5432)),
        'dbname': os.getenv('POSTGRES_DB', 'hybrid_rag'),
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
    }
    
    # Diagnostic query
    query = "How can GST registration be cancelled?"
    top_k = 10
    
    print_separator()
    print("BM25 RETRIEVAL DIAGNOSTIC TEST")
    print_separator()
    print(f"\nQuery: \"{query}\"")
    print(f"Top-K: {top_k}")
    print(f"\nDatabase: {conn_params['dbname']}@{conn_params['host']}:{conn_params['port']}")
    print_separator()
    
    try:
        # Initialize retriever
        print("\n[1/3] Initializing BM25 retriever...")
        retriever = BM25Retriever(conn_params)
        print("      ✓ Retriever initialized")
        
        # Perform retrieval
        print(f"\n[2/3] Executing BM25 search...")
        start_time = datetime.now()
        results, stats = retriever.retrieve(query, top_k=top_k)
        end_time = datetime.now()
        
        print(f"      ✓ Search completed in {stats['retrieval_time']:.4f} seconds")
        
        # Display results
        print(f"\n[3/3] Results:")
        print_separator("-", 80)
        
        if not results:
            print("\n⚠ No results found.")
            print("\nPossible reasons:")
            print("  - BM25 indexes not created (run migration/add_bm25_indexes.sql)")
            print("  - No matching content in database")
            print("  - pg_search extension not installed/enabled")
        else:
            print(f"\nFound {len(results)} results:")
            print(f"  - From Acts: {stats['acts_fetched']} candidates")
            print(f"  - From Rules: {stats['rules_fetched']} candidates")
            print(f"  - From Forms: {stats['forms_fetched']} candidates")
            print(f"\nRetrieval Statistics:")
            print(f"  - Total time: {stats['retrieval_time']:.4f}s")
            print(f"  - Results returned: {len(results)}")
            
            print_separator("-", 80)
            for i, result in enumerate(results, 1):
                print_result(result, i)
            
            print_separator("-", 80)
        
        print("\n✓ Diagnostic test completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error occurred: {type(e).__name__}: {e}")
        print("\nTroubleshooting steps:")
        print("  1. Ensure PostgreSQL is running")
        print("  2. Verify pg_search extension is installed:")
        print("     SELECT * FROM pg_extension WHERE extname = 'pg_search';")
        print("  3. Check that BM25 indexes exist:")
        print("     SELECT * FROM pg_indexes WHERE indexname LIKE '%bm25%';")
        print("  4. Run migration script if needed:")
        print("     psql -h localhost -U postgres -d hybrid_rag -f migration/add_bm25_indexes.sql")
        sys.exit(1)
    
    print_separator()


if __name__ == "__main__":
    main()
