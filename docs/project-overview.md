# Project architecture, tools, and techniques

## Purpose and implemented scope

The project demonstrates retrieval over a small set of medical-code descriptions. It combines a semantic retrieval demonstration with a separate keyword search. Keyword search uses Timescale `pg_textsearch` BM25 matching first and character-based fuzzy similarity only when no BM25 rows match. A second model reranks the semantic candidates to improve their ordering for the query.

The source contains 22 manually embedded code/description pairs covering hypertension, hypotension, diabetes, cardiovascular and respiratory conditions, kidney conditions, and symptoms. There is no external terminology import, document ingestion pipeline, or validation of the descriptions against a medical coding catalogue. The code is a search demonstration; it does not implement clinical decision-making or validate coding correctness.

The repository name describes a hybrid RAG direction. Currently, semantic retrieval and keyword retrieval run independently with different demonstration queries. There is no result fusion and no answer-generation stage.

## Data flow

```mermaid
flowchart TD
    A[22 code and description pairs in test_pgvector.py] --> B[SentenceTransformer: all-MiniLM-L6-v2]
    B --> C[PostgreSQL documents table with pgvector embeddings]
    D[Semantic query: poorly controlled high blood pressure] --> E[Embed query with the same model]
    E --> F[Cosine-distance SQL search: up to 10 candidates]
    C --> F
    F --> G[CrossEncoder scores query-description pairs]
    G --> H[Sort scores descending and print up to 5 results]
    I[Keyword query] --> J[pg_textsearch BM25 ranking]
    C --> J
    J --> L{Any matches?}
    L -->|Yes| K[Print BM25 results]
    L -->|No| M[pg_trgm fallback for the same query]
    C --> M
    M --> N[Print fuzzy results or no matches]
```

## Tools and technologies

This inventory covers direct dependencies and the infrastructure and model choices explicitly used or declared in the repository. Transitive library versions are not recorded by a lockfile.

| Tool or component | How this project uses it |
| --- | --- |
| Python | Implements sample loading, embedding inference, SQL execution, reranking, and console output in two standalone scripts and a keyword-search module. |
| `psycopg[binary]` | PostgreSQL driver; opens connections, executes parameterized SQL, fetches rows, and commits inserted data. The requirement requests its binary distribution option. |
| `pgvector` Python package | Supplies `pgvector.psycopg.register_vector(conn)` so vector values can be adapted through Psycopg. This package is distinct from the database extension. |
| PostgreSQL 17 | Stores codes, descriptions, and embeddings and executes semantic, BM25, and fallback queries. |
| pgvector PostgreSQL extension | Provides the vector column type and the `<=>` cosine-distance operator used for semantic retrieval. |
| Timescale `pg_textsearch` 1.4.0 | Supplies the primary BM25 keyword index and `<@>` scoring operator; requires preloading at server startup. |
| `pg_trgm` PostgreSQL extension | Provides `similarity()` and the `%` threshold operator used only for fallback fuzzy matching. It is required by the search SQL but is not enabled by either script. |
| `sentence-transformers` | Provides the `SentenceTransformer` and `CrossEncoder` model interfaces. |
| `all-MiniLM-L6-v2` | Encodes each description and the semantic query into 384-dimensional dense vectors. |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Scores each retrieved query/description pair before the semantic results are reordered. |
| Docker and Docker Compose | Run the database as the `hybrid-rag-db` container using a custom image based on `pgvector/pgvector:pg17`, with host port `5432` and a persistent `pgdata17` volume. `Dockerfile.db` builds the pinned `pg_textsearch` release using Make, a C compiler, and PostgreSQL development headers; curl downloads the release archive. |
| `python-dotenv` | Declared in `requirements.txt`, but not imported or used. `.env.example` is empty. |
| `pip` and Python `venv` | Used in the documented local installation workflow to install dependencies into an isolated environment. |

The code uses local model inference through Sentence Transformers; it does not call an embedding or generation API. Model files may need downloading when first loaded. No explicit device selection, model revision, dependency version pin, or inference tuning is configured.

## Techniques in detail

### 1. Dense embeddings for semantic retrieval

`test_pgvector.py` calls `model.encode(description)` once per sample and stores the resulting vector with its code and description. `search_pgvec.py` encodes the query using the same model, placing queries and descriptions into a comparable vector space.

This permits matching by learned semantic similarity rather than requiring an exact word overlap. The phrase `high blood pressure`, for example, can be compared to descriptions containing `hypertension`. This is the intent of the example, not an asserted ranking guarantee.

Each short description is embedded as a whole. There is no chunking, text-cleaning pipeline, batch embedding call, metadata enrichment, or explicit vector normalization in the application code.

### 2. Cosine-distance nearest-neighbor retrieval

The semantic SQL query computes `embedding <=> query_embedding`, orders by this distance in ascending order, and applies `LIMIT 10`. Lower distance means a closer match under the cosine-distance metric.

The repository defines no vector index. With the minimal schema in the README, this is an exact search over stored vectors, rather than an approximate nearest-neighbor search. HNSW and IVFFlat are not configured. Any indexes created independently in an existing database are outside what the repository records.

### 3. Retrieve, then rerank

After fetching candidates, the script builds `(query, description)` pairs and passes them to `reranker.predict(pairs)`. The cross-encoder evaluates each pair jointly and produces a relevance score. The script converts each score to a Python float, sorts descending, and prints the top five candidates.

This is a two-stage retrieval architecture: embeddings select a small candidate set, then a more detailed pairwise model scores that set. The reranker can change the order of those candidates but cannot recover a document excluded from the initial top 10.

Each printed semantic result includes its code, description, original vector distance, and reranking score. The final order uses only the reranking score. These model scores are not calibrated probabilities or combined with the vector distance.

### 4. BM25 keyword search with trigram fallback

`keyword_search.py` provides `keyword_search(conn, query, limit=5)`, returning `(rows, method)`. Each row contains a code, description, and method-specific score.

`schema.sql` creates `documents_description_bm25_idx` using Timescale `pg_textsearch`, with English text processing and BM25 parameters `k1=1.2` and `b=0.75`. BM25 ranks keyword relevance using term frequency, corpus-wide term rarity, and document length. English processing handles stemming and stopwords. Documents can match any query term; there is no phrase or Boolean query interface in this application.

`bm25_search` uses `description <@> to_bm25query(query, 'documents_description_bm25_idx')`. The explicit index supplies scoring context even for small-table sequential plans. The operator returns negative BM25 scores, so SQL orders ascending and filters to scores below zero. This prevents zero-relevance records from suppressing fallback. Returned scores are negated into positive `bm25_score` values, with higher values representing better matches. Equal-score ordering is unspecified. See the [pinned pg_textsearch documentation](https://github.com/timescale/pg_textsearch/blob/v1.4.0/README.md).

Only if BM25 returns zero relevant rows does the wrapper call `fuzzy_search` with the same query and limit. That function uses `similarity(description, query)` and the `%` threshold operator, ranking descending. The SQL escapes the literal operator as `%%` for Psycopg. Matching uses the entire description and the session's trigram threshold; no trigram index or threshold change is configured by the application.

Partial BM25 results are returned without fuzzy top-up. Blank input returns no rows without accessing the database; nonpositive limits raise `ValueError`. Stopword-only input attempts fallback after finding no BM25 matches. Database errors propagate rather than triggering fallback. BM25 scores and trigram similarities are not combined.

The script demonstrates `hypertension` and the misspelling `hypertensoin`, printing the method that supplied the results. Keyword results remain separate from vector retrieval and cross-encoder reranking.

### 5. Parameterized SQL and transactions

Both scripts pass values separately from SQL using Psycopg placeholders. This covers inserted data, vector queries, keyword query text, and keyword result limits; values are not interpolated into SQL strings.

The loader inserts all samples on one connection and commits after the loop. It does not use an upsert or check for existing codes. Both scripts use cursor context managers and close their connections on the successful execution path. There is no explicit error recovery or connection context manager covering failures.

### 6. Containerized persistence

Docker Compose supplies a reproducible database service configuration and maps `pgdata17` to `/var/lib/postgresql/data`. The new PostgreSQL 17 volume preserves the initialized database and sample rows across ordinary container restarts and `docker compose down`.

Compose declares database credentials and the database name, but has no schema initialization mount or health check. The server command preloads `pg_textsearch`; database readiness and running `schema.sql` are manual steps in the README. The PostgreSQL 16 to 17 transition uses a new volume and a documented logical backup/restore procedure. Python runs on the host and connects through the mapped port; there is no application container.

## Data model

Both scripts assume a table named `documents` with these fields:

| Column | Compatible type | Meaning |
| --- | --- | --- |
| `code` | `TEXT` | Medical-code identifier from the sample list. |
| `description` | `TEXT` | Human-readable text to embed, display, and fuzzy-match. |
| `embedding` | `VECTOR(384)` | Dense representation produced by the embedding model. |

The Python scripts contain no DDL; `schema.sql` creates a compatible table and the required BM25 index. The README explains initialization and links to the database upgrade procedure. Changing the embedding model requires checking the vector dimension and regenerating stored embeddings in a consistent vector space.

## Current limitations and unfinished components

- **Hybrid result fusion:** no weighted score combination, reciprocal rank fusion, shared candidate pool, or deduplication across retrieval methods.
- **RAG answer generation:** no generative model, prompt construction, retrieved context assembly, citations, or conversation handling.
- **Data ingestion:** sample records are hardcoded; `data/` and `src/` are empty. No file parser, terminology sync, chunking, or incremental update path exists.
- **Repeatable loading:** repeated loader runs append duplicates unless the database has independently added constraints; the database upgrade is documented, but no upsert is provided.
- **Evaluation:** `test_pgvector.py` seeds data and has no assertions. Keyword routing and optional database integration tests live in `test_keyword_search.py`. There are no relevance labels, recall/precision measurements, reranker comparisons, or latency benchmarks.
- **Search configuration:** query strings, candidate count, output count, model names, and credentials are embedded in source. The keyword, BM25, and fuzzy functions expose a `limit` argument.
- **Performance:** `schema.sql` defines a BM25 index, but no vector or trigram indexes; embeddings are inserted one record at a time, and `search_pgvec.py` loads the embedding model twice. The second instance, `embedding_model`, is unused.
- **Robustness:** no explicit handling for an empty semantic candidate set before reranking, failed model loads, failed database operations, or cleanup after exceptions.
- **Packaging and operations:** no API, UI, CLI parser, application container, CI configuration, lockfile, structured logging, or production deployment configuration is present. `CrossEncoder` is imported but unused in the loader.

Possible extensions are to combine both retrieval methods for the same query, deduplicate and rerank their candidate pool, add measured retrieval evaluation, and then introduce an answer-generation stage if needed. These are future directions, not implemented behavior.

## Source map

- [Sample loader](../test_pgvector.py): sample dataset, description embeddings, inserts, and commit.
- [Search demonstration](../search_pgvec.py): query embedding, cosine-distance retrieval, reranking, and keyword demonstrations.
- [Keyword search](../keyword_search.py): primary BM25 SQL and trigram fallback routing.
- [Keyword tests](../test_keyword_search.py): routing and PostgreSQL integration checks.
- [Database schema](../schema.sql): extension initialization and the BM25 index.
- [Database image](../Dockerfile.db): PostgreSQL 17, pgvector, and Timescale `pg_textsearch`.
- [Database upgrade](postgresql-upgrade.md): preserve PostgreSQL 16 data when moving to PostgreSQL 17.
- [Database service](../docker-compose.yml): container image, port, credentials, and persistent storage.
- [Python dependencies](../requirements.txt): direct package requirements.
- [Setup and execution](../README.md): local commands, compatible schema initialization, and troubleshooting.
