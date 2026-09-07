# Project architecture, tools, and techniques

## Purpose and implemented scope

The project demonstrates retrieval over a small set of medical-code descriptions. It explores two ways to find relevant text: semantic similarity, which can connect differently worded phrases, and character-based fuzzy similarity, which can tolerate some spelling differences. A second model reranks the semantic candidates to improve their ordering for the query.

The source contains 22 manually embedded code/description pairs covering hypertension, hypotension, diabetes, cardiovascular and respiratory conditions, kidney conditions, and symptoms. There is no external terminology import, document ingestion pipeline, or validation of the descriptions against a medical coding catalogue. The code is a search demonstration; it does not implement clinical decision-making or validate coding correctness.

The repository name describes a hybrid RAG direction. Currently, semantic retrieval and fuzzy retrieval run independently with different demonstration queries. There is no result fusion and no answer-generation stage.

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
    I[Fuzzy query: hypertensoin] --> J[pg_trgm threshold filter and similarity ordering]
    C --> J
    J --> K[Print up to 5 fuzzy results separately]
```

## Tools and technologies

This inventory covers direct dependencies and the infrastructure and model choices explicitly used or declared in the repository. Transitive library versions are not recorded by a lockfile.

| Tool or component | How this project uses it |
| --- | --- |
| Python | Implements sample loading, embedding inference, SQL execution, reranking, and console output in two standalone scripts. |
| `psycopg[binary]` | PostgreSQL driver; opens connections, executes parameterized SQL, fetches rows, and commits inserted data. The requirement requests its binary distribution option. |
| `pgvector` Python package | Supplies `pgvector.psycopg.register_vector(conn)` so vector values can be adapted through Psycopg. This package is distinct from the database extension. |
| PostgreSQL 16 | Stores codes, descriptions, and embeddings and executes both retrieval queries. |
| pgvector PostgreSQL extension | Provides the vector column type and the `<=>` cosine-distance operator used for semantic retrieval. |
| `pg_trgm` PostgreSQL extension | Provides `similarity()` and the `%` threshold operator used for fuzzy matching. It is required by the search SQL but is not enabled by either script. |
| `sentence-transformers` | Provides the `SentenceTransformer` and `CrossEncoder` model interfaces. |
| `all-MiniLM-L6-v2` | Encodes each description and the semantic query into 384-dimensional dense vectors. |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Scores each retrieved query/description pair before the semantic results are reordered. |
| Docker and Docker Compose | Run the database as the `hybrid-rag-db` container using `pgvector/pgvector:pg16`, with host port `5432` and a persistent `pgdata` volume. |
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

### 4. Trigram fuzzy matching

`fuzzy_search(conn, query, limit=5)` scores each description with `similarity(description, query)`, filters it with PostgreSQL's `%` operator, sorts by score descending, and limits the results. Trigrams compare character groups, making this a lexical matching technique that can tolerate some spelling variation.

The SQL string contains `description %% %s`: Psycopg uses `%s` for parameter binding, so the literal PostgreSQL `%` operator is escaped as `%%` in that string.

The query is the deliberately misspelled `hypertensoin`. Matching applies to the whole description and uses the database session's configured trigram similarity threshold. The script does not set that threshold or create a trigram index. The limit is a maximum, not a promise that five rows match.

The fuzzy function is called after the semantic demonstration with a different query. Its results do not participate in semantic candidate selection or reranking.

### 5. Parameterized SQL and transactions

Both scripts pass values separately from SQL using Psycopg placeholders. This covers inserted data, vector queries, fuzzy query text, and the fuzzy result limit; values are not interpolated into SQL strings.

The loader inserts all samples on one connection and commits after the loop. It does not use an upsert or check for existing codes. Both scripts use cursor context managers and close their connections on the successful execution path. There is no explicit error recovery or connection context manager covering failures.

### 6. Containerized persistence

Docker Compose supplies a reproducible database service configuration and maps `pgdata` to `/var/lib/postgresql/data`. The named volume preserves the initialized database and sample rows across ordinary container restarts and `docker compose down`.

Compose declares database credentials and the database name, but has no schema initialization mount or health check. Database readiness and enabling extensions are manual steps in the README. Python runs on the host and connects through the mapped port; there is no application container.

## Data model

Both scripts assume a table named `documents` with these fields:

| Column | Compatible type | Meaning |
| --- | --- | --- |
| `code` | `TEXT` | Medical-code identifier from the sample list. |
| `description` | `TEXT` | Human-readable text to embed, display, and fuzzy-match. |
| `embedding` | `VECTOR(384)` | Dense representation produced by the embedding model. |

The original scripts contain no DDL or migrations. The README supplies a minimal compatible initialization example, rather than claiming a schema or constraints already exist in a running database. Changing the embedding model requires checking the vector dimension and regenerating stored embeddings in a consistent vector space.

## Current limitations and unfinished components

- **Hybrid result fusion:** no weighted score combination, reciprocal rank fusion, shared candidate pool, or deduplication across retrieval methods.
- **RAG answer generation:** no generative model, prompt construction, retrieved context assembly, citations, or conversation handling.
- **Data ingestion:** sample records are hardcoded; `data/` and `src/` are empty. No file parser, terminology sync, chunking, or incremental update path exists.
- **Repeatable loading:** repeated loader runs append duplicates unless the database has independently added constraints; no schema migration or upsert is provided.
- **Evaluation:** `test_pgvector.py` seeds data and has no assertions. There are no relevance labels, recall/precision measurements, reranker comparisons, latency benchmarks, or automated tests.
- **Search configuration:** query strings, candidate count, output count, model names, and credentials are embedded in source. The fuzzy function alone exposes a `limit` argument.
- **Performance:** no vector or trigram indexes are defined, embeddings are inserted one record at a time, and `search_pgvec.py` loads the embedding model twice. The second instance, `embedding_model`, is unused.
- **Robustness:** no explicit handling for an empty semantic candidate set before reranking, failed model loads, failed database operations, or cleanup after exceptions.
- **Packaging and operations:** no API, UI, CLI parser, application container, CI configuration, lockfile, structured logging, or production deployment configuration is present. `CrossEncoder` is imported but unused in the loader.

Possible extensions are to combine both retrieval methods for the same query, deduplicate and rerank their candidate pool, add measured retrieval evaluation, and then introduce an answer-generation stage if needed. These are future directions, not implemented behavior.

## Source map

- [Sample loader](../test_pgvector.py): sample dataset, description embeddings, inserts, and commit.
- [Search demonstration](../search_pgvec.py): query embedding, cosine-distance retrieval, reranking, and fuzzy SQL.
- [Database service](../docker-compose.yml): container image, port, credentials, and persistent storage.
- [Python dependencies](../requirements.txt): direct package requirements.
- [Setup and execution](../README.md): local commands, compatible schema initialization, and troubleshooting.
