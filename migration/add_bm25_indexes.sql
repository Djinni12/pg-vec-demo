-- Migration script to add BM25 indexes for legal knowledge base tables
-- Requires pg_search (ParadeDB) extension for true BM25 scoring

-- Enable pg_search extension
CREATE EXTENSION IF NOT EXISTS pg_search;

-- BM25 index for act_chunks table
-- Indexes content, section_title, section_number, and chapter for lexical search
CALL paradedb.create_bm25(
    table_name => 'act_chunks',
    index_name => 'act_chunks_bm25_idx',
    key_field => 'chunk_id',
    text_fields => '{
        "content": {},
        "section_title": {},
        "section_number": {},
        "chapter": {},
        "act_name": {}
    }'
);

-- BM25 index for rule_chunks table
-- Indexes content, rule_title, rule_number, and chapter fields
CALL paradedb.create_bm25(
    table_name => 'rule_chunks',
    index_name => 'rule_chunks_bm25_idx',
    key_field => 'chunk_id',
    text_fields => '{
        "content": {},
        "rule_title": {},
        "rule_number": {},
        "chapter": {},
        "chapter_title": {}
    }'
);

-- BM25 index for form_chunks table
-- Indexes content, form_title, form_number, and related metadata fields
CALL paradedb.create_bm25(
    table_name => 'form_chunks',
    index_name => 'form_chunks_bm25_idx',
    key_field => 'chunk_id',
    text_fields => '{
        "content": {},
        "title": {},
        "form_title": {},
        "form_number": {},
        "form_uid": {},
        "section_label": {}
    }'
);

-- Verify indexes were created
SELECT indexname, tablename 
FROM pg_indexes 
WHERE indexname LIKE '%bm25%' 
ORDER BY tablename, indexname;
