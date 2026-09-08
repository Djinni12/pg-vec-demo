CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pg_textsearch;

-- Separate GST corpus: existing ICD-style documents are retained but not queried.
CREATE TABLE IF NOT EXISTS gst_documents (
    source_file TEXT NOT NULL,
    source_row INTEGER NOT NULL CHECK (source_row > 0),
    record_type TEXT NOT NULL CHECK (record_type IN ('goods', 'service')),
    code TEXT NOT NULL,
    exact_codes TEXT[] NOT NULL DEFAULT '{}',
    description TEXT NOT NULL,
    search_text TEXT NOT NULL,
    cgst_rate_pct NUMERIC,
    sgst_utgst_rate_pct NUMERIC,
    igst_rate_pct NUMERIC,
    metadata JSONB NOT NULL,
    embedding VECTOR(384) NOT NULL,
    PRIMARY KEY (source_file, source_row)
);

CREATE INDEX IF NOT EXISTS gst_documents_search_bm25_idx
    ON gst_documents USING bm25(search_text)
    WITH (text_config='english', k1=1.2, b=0.75);

CREATE INDEX IF NOT EXISTS gst_documents_exact_codes_idx
    ON gst_documents USING GIN(exact_codes);

-- Legal Act chunks with BGE-M3 embeddings for semantic retrieval experiments.
CREATE TABLE IF NOT EXISTS act_chunks (
    chunk_id TEXT PRIMARY KEY,
    act_name TEXT NOT NULL,
    chapter TEXT,
    section_number TEXT NOT NULL,
    section_title TEXT NOT NULL,
    subsection_numbers TEXT[] NOT NULL DEFAULT '{}',
    status TEXT NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL CHECK (token_count > 0),
    embedding VECTOR(1024) NOT NULL
);

CREATE INDEX IF NOT EXISTS act_chunks_embedding_hnsw_idx
    ON act_chunks USING hnsw (embedding vector_cosine_ops);

-- CGST Rules chunks with BGE-M3 embeddings for semantic retrieval experiments.
CREATE TABLE IF NOT EXISTS rule_chunks (
    chunk_id TEXT PRIMARY KEY,
    rule_number TEXT NOT NULL,
    rule_title TEXT NOT NULL,
    chapter TEXT,
    chapter_title TEXT,
    subrule_numbers TEXT[] NOT NULL DEFAULT '{}',
    status TEXT NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL CHECK (token_count > 0),
    chunk_strategy TEXT NOT NULL,
    embedding VECTOR(1024) NOT NULL
);

CREATE INDEX IF NOT EXISTS rule_chunks_embedding_hnsw_idx
    ON rule_chunks USING hnsw (embedding vector_cosine_ops);


-- GST Forms chunks with BGE-M3 embeddings for semantic retrieval experiments.
CREATE TABLE IF NOT EXISTS form_chunks (
    chunk_id TEXT PRIMARY KEY,
    form_uid TEXT NOT NULL,
    form_number TEXT NOT NULL,
    form_family TEXT NOT NULL,
    form_code TEXT NOT NULL,
    form_title TEXT NOT NULL,
    title TEXT NOT NULL,
    language TEXT NOT NULL,
    rule_references TEXT[] NOT NULL DEFAULT '{}',
    part_number INTEGER,
    section_label TEXT,
    source_start_page INTEGER,
    source_end_page INTEGER,
    page_start INTEGER,
    page_end INTEGER,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL CHECK (token_count > 0),
    chunk_strategy TEXT NOT NULL,
    metadata JSONB NOT NULL,
    embedding VECTOR(1024) NOT NULL
);


ALTER TABLE form_chunks ADD COLUMN IF NOT EXISTS form_uid TEXT;
ALTER TABLE form_chunks ADD COLUMN IF NOT EXISTS title TEXT;
ALTER TABLE form_chunks ADD COLUMN IF NOT EXISTS language TEXT;
ALTER TABLE form_chunks ADD COLUMN IF NOT EXISTS rule_references TEXT[] NOT NULL DEFAULT '{}';
ALTER TABLE form_chunks ADD COLUMN IF NOT EXISTS part_number INTEGER;
ALTER TABLE form_chunks ADD COLUMN IF NOT EXISTS section_label TEXT;
ALTER TABLE form_chunks ADD COLUMN IF NOT EXISTS source_start_page INTEGER;
ALTER TABLE form_chunks ADD COLUMN IF NOT EXISTS source_end_page INTEGER;

CREATE INDEX IF NOT EXISTS form_chunks_embedding_hnsw_idx
    ON form_chunks USING hnsw (embedding vector_cosine_ops);
