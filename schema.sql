CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pg_textsearch;

CREATE TABLE IF NOT EXISTS documents (
    code TEXT,
    description TEXT,
    embedding VECTOR(384)
);

CREATE INDEX IF NOT EXISTS documents_description_bm25_idx
    ON documents USING bm25(description)
    WITH (text_config='english', k1=1.2, b=0.75);
