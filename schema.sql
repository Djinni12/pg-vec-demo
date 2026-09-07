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
