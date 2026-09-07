# Hybrid RAG retrieval proof of concept

This project explores searching medical-code descriptions using PostgreSQL and local machine-learning models. It embeds 22 sample descriptions, finds descriptions related to a natural-language query, reranks the candidates, and demonstrates a separate BM25 keyword search with a typo-tolerant trigram fallback.

The current implementation is the retrieval portion of a possible hybrid retrieval-augmented generation (RAG) system. It does **not** yet merge semantic and keyword results, construct a prompt, or use a language model to generate an answer. Results are printed in the terminal.

For the complete explanation of the architecture, tools, techniques, and limitations, see [Project documentation](docs/project-overview.md).

## Quick start

Prerequisites: Python with `venv` and `pip`, Docker with Docker Compose and permission to access its daemon, and an available local port `5432`. Model loading requires network access on the first run unless the models are already cached. Dependency versions and a supported Python version are not pinned in this repository.

### 1. Install Python dependencies

Run these commands from the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Start PostgreSQL

```bash
docker compose up -d --build db
docker compose exec db pg_isready -U postgres -d hybrid_rag
```

Wait until the readiness check reports that the database is accepting connections. Compose builds PostgreSQL 17 with pgvector and Timescale `pg_textsearch` 1.4.0, preloads `pg_textsearch`, exposes port `5432`, and stores database files in the new `pgdata17` volume. The first build downloads and compiles the extension.

**Existing PostgreSQL 16 users:** follow [the migration instructions](docs/postgresql-upgrade.md) before starting the new container. The previous `pgdata` volume is retained, but PostgreSQL 17 cannot directly open it.

### 3. Initialize the database

Neither Python script creates extensions or tables. For a fresh database, run the following manual setup before running either script:

```bash
docker compose exec -T db psql -v ON_ERROR_STOP=1 -U postgres -d hybrid_rag < schema.sql
```

`schema.sql` enables all three extensions, creates the compatible 384-dimensional embedding table if absent, and creates the required BM25 index over `description`. It can also initialize an existing compatible table after migration; no re-embedding is needed. It does not validate or migrate an incompatible table schema, deduplicate rows, or add uniqueness constraints.

### 4. Insert sample descriptions

```bash
python test_pgvector.py
```

Despite its name, this is a data-loading script, not an automated test. It inserts 22 code/description pairs and their embeddings, commits them, and prints `Inserted test documents.` Running it again inserts the same data again with the schema above.

### 5. Run the search demonstration

```bash
python search_pgvec.py
```

The script demonstrates semantic retrieval and two keyword queries:

- Semantic search for `poorly controlled high blood pressure`: retrieve up to 10 candidates by cosine distance, rerank them with a cross-encoder, and print up to five results with both scores.
- Keyword search for `hypertension` and `hypertensoin`: try Timescale `pg_textsearch` BM25 first. Only when it returns zero rows, run `pg_trgm` fuzzy search for the same query. Print up to five results and label the method (`bm25` or `trigram`) and score (`bm25_score` or `similarity`). Fallback results may also be empty; partial BM25 results are not topped up.

Queries and database connection settings are hardcoded in the scripts. There are no command-line arguments or interactive query prompts.

### 6. Stop the database

```bash
docker compose down
```

The named database volume is retained for future runs.

## Repository contents

| Path | Purpose |
| --- | --- |
| `test_pgvector.py` | Defines sample medical-code descriptions, embeds them, and inserts them into PostgreSQL. |
| `search_pgvec.py` | Runs vector retrieval, cross-encoder reranking, and BM25 keyword search with trigram fallback. |
| `keyword_search.py` | BM25 SQL, trigram SQL, and fallback routing without model-loading side effects. |
| `test_keyword_search.py` | Routing tests and optional PostgreSQL integration tests. |
| `Dockerfile.db` | Adds Timescale `pg_textsearch` 1.4.0 to the PostgreSQL 17/pgvector image. |
| `schema.sql` | Enables extensions and creates the documents table and BM25 index. |
| `docker-compose.yml` | Configures the PostgreSQL/pgvector container and persistent volume. |
| `requirements.txt` | Lists direct Python dependencies without version pins. |
| `.env.example` | Empty placeholder; the scripts do not currently load environment variables. |
| `docs/project-overview.md` | Detailed architecture and inventory of tools and techniques. |
| `data/`, `src/` | Currently empty placeholders; executable code and sample data are in the root scripts. |
| `hello.txt` | Incidental text file, unused by the retrieval scripts. |

## Configuration and troubleshooting

Both scripts connect to `dbname=hybrid_rag user=postgres password=postgres host=localhost port=5432`. These match the local Compose configuration and are development credentials. Changing Compose settings alone does not update the scripts. Although `python-dotenv` is listed as a dependency, it is not used.

| Symptom | What to check |
| --- | --- |
| Connection refused | Start the database, confirm readiness, and check whether another service occupies port `5432`. |
| Vector registration fails or `documents` does not exist | Run the extension and table initialization above in the `hybrid_rag` database. |
| `similarity` or the trigram operator is unavailable | Enable `pg_trgm` in the database used by the scripts. |
| BM25 operator/index is unavailable | Build the new database image, confirm `pg_textsearch` is preloaded, and run `schema.sql`. |
| Model download fails | Check network access or availability of the model files in the local cache. |
| Repeated results | The loader appends rows each time; it has no deduplication or upsert behavior. |
| Fuzzy output is empty | Matching uses a threshold against the entire description; a typo query is not guaranteed to pass it. |

Run the keyword routing tests with:

```bash
venv/bin/python -m unittest test_keyword_search -v
```

To also run PostgreSQL integration tests against the initialized database:

```bash
TEST_DATABASE_URL='dbname=hybrid_rag user=postgres password=postgres host=localhost port=5432' venv/bin/python -m unittest test_keyword_search -v
```

Integration tests use a temporary table and roll back without changing stored documents. They cover BM25 matching and score ordering, stemming, stopwords, limits, typo fallback, and empty results. There is no retrieval-quality evaluation dataset or assertion of model rankings.
# pg-vec-demo
