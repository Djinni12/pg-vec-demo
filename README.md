# Hybrid RAG retrieval proof of concept

This project explores searching medical-code descriptions using PostgreSQL and local machine-learning models. It embeds 22 sample descriptions, finds descriptions related to a natural-language query, reranks the candidates, and demonstrates a separate typo-tolerant text search.

The current implementation is the retrieval portion of a possible hybrid retrieval-augmented generation (RAG) system. It does **not** yet merge semantic and fuzzy results, construct a prompt, or use a language model to generate an answer. Results are printed in the terminal.

For the complete explanation of the architecture, tools, techniques, and limitations, see [Project documentation](docs/project-overview.md).

## Quick start

Prerequisites: Python with `venv` and `pip`, Docker with Docker Compose, and an available local port `5432`. Model loading requires network access on the first run unless the models are already cached. Dependency versions and a supported Python version are not pinned in this repository.

### 1. Install Python dependencies

Run these commands from the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Start PostgreSQL

```bash
docker compose up -d db
docker compose exec db pg_isready -U postgres -d hybrid_rag
```

Wait until the readiness check reports that the database is accepting connections. Compose runs PostgreSQL 16 with pgvector, exposes port `5432`, and stores database files in the `pgdata` named volume.

### 3. Initialize the database

Neither Python script creates extensions or tables. For a fresh database, run the following manual setup before running either script:

```bash
docker compose exec -T db psql -v ON_ERROR_STOP=1 -U postgres -d hybrid_rag <<'SQL'
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS documents (
    code TEXT,
    description TEXT,
    embedding VECTOR(384)
);
SQL
```

This is a minimal schema compatible with the current scripts and the 384-dimensional `all-MiniLM-L6-v2` embeddings. It does not add uniqueness constraints or indexes. `IF NOT EXISTS` does not validate or migrate an existing table with a different schema.

### 4. Insert sample descriptions

```bash
python test_pgvector.py
```

Despite its name, this is a data-loading script, not an automated test. It inserts 22 code/description pairs and their embeddings, commits them, and prints `Inserted test documents.` Running it again inserts the same data again with the schema above.

### 5. Run the search demonstration

```bash
python search_pgvec.py
```

The script performs two demonstrations:

- Semantic search for `poorly controlled high blood pressure`: retrieve up to 10 candidates by cosine distance, rerank them with a cross-encoder, and print up to five results with both scores.
- Fuzzy search for `hypertensoin`: print up to five descriptions that pass PostgreSQL's trigram similarity threshold, ordered by similarity. Depending on the threshold and stored descriptions, this section may have fewer results or none.

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
| `search_pgvec.py` | Runs vector retrieval, cross-encoder reranking, and an independent trigram fuzzy search. |
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
| Model download fails | Check network access or availability of the model files in the local cache. |
| Repeated results | The loader appends rows each time; it has no deduplication or upsert behavior. |
| Fuzzy output is empty | Matching uses a threshold against the entire description; a typo query is not guaranteed to pass it. |

There is no automated assertion-based test suite or evaluation dataset. A manual smoke check is to load the samples and run the search script successfully; exact rankings are not asserted.
# pg-vec-demo
