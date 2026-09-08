# Hybrid RAG retrieval proof of concept

This project searches goods and services from the [Kaggle GST rates dataset](https://www.kaggle.com/datasets/prasad22/goods-and-service-tax-rates-dataset). It embeds descriptions, retrieves semantic matches with pgvector, runs BM25 keyword search with trigram fallback, and can combine the BM25 and vector lists with Reciprocal Rank Fusion (RRF). An explicit HSN/SAC code lookup is also available.

The inspected dataset contains 1,850 goods rows and 232 service rows; ingestion retains 1,729 searchable records. See [dataset inspection and field mapping](docs/gst-dataset.md) for all columns, rate handling, and exact-code limitations.

The current implementation is the retrieval portion of a possible hybrid retrieval-augmented generation (RAG) system. It does not construct a prompt or use a language model to generate an answer. Results are printed in the terminal.

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

Python does not create extensions or tables. Run this setup before ingestion or search, including on an existing project database:

```bash
docker compose exec -T db psql -v ON_ERROR_STOP=1 -U postgres -d hybrid_rag < schema.sql
```

`schema.sql` enables the extensions and creates `gst_documents`, a BM25 index over `search_text`, and an array index for exact codes. Existing medical demo data in `documents` is retained but no longer queried. No table drop or legacy embedding migration is needed.

### 4. Download, inspect, and ingest GST data

Download and extract `Goods.csv` and `Services.csv` from the Kaggle dataset into `data/gst/csvs/`. The files downloaded during this implementation are already there locally and ignored by Git. On a new checkout, a public download can be attempted with:

```bash
curl -fL 'https://www.kaggle.com/api/v1/datasets/download/prasad22/goods-and-service-tax-rates-dataset' -o /tmp/gst-kaggle.zip
mkdir -p data/gst/csvs
python -m zipfile -e /tmp/gst-kaggle.zip data/gst/csvs
```

If Kaggle requires authentication, download through the dataset page instead.

```bash
python ingest_gst.py --dry-run
python ingest_gst.py
```

The dry run validates both files and prints row counts, skipped rows, encoding, and hashes without downloading models or writing to PostgreSQL. The import embeds descriptions and upserts GST records in one transaction. Repeat imports update existing records and remove stale rows from the same source files. `python tests/test_pgvector.py` is a compatibility alias for the loader.

Goods fractional rates are converted to percentages; service percentage values remain unchanged. Conditional or missing rates retain their source text and have no numeric value. The [dataset guide](docs/gst-dataset.md) explains these decisions.

### 5. Run the search demonstration

```bash
python search_pgvec.py "roasted coffee beans"
python search_pgvec.py "roasted coffee beans" --rrf
python search_pgvec.py "roasted coffee beans" --rrf --rerank-vector-before-rrf
python search_pgvec.py "cofee"
python search_pgvec.py --exact-code 0901
```

Text queries run two independent paths:

- Semantic search: retrieve up to 10 description embeddings by cosine distance, rerank them with the existing cross-encoder, and print up to five results.
- Keyword search: retrieve up to five positive BM25 matches over description, classification, and condition/cess. Only if there are no matches, use trigram fuzzy matching over descriptions. Partial BM25 results are not topped up.
- RRF hybrid search: with `--rrf`, retrieve Top-N BM25 and Top-N vector candidates independently, merge by `(source_file, source_row)`, sum `1 / (60 + rank)` contributions, and print the fused Top-K. RRF uses only rank positions, not BM25 scores or vector distances. Add `--rerank-vector-before-rrf` to reorder only the vector candidate list with the existing CrossEncoder before rank fusion.

Exact lookup skips the models and returns up to 20 rows containing the explicit code. It does not infer ranges or code hierarchies. Results display classification, description, source filename/row, and GST rates.

Both ingestion and search accept `--database-url` to override the local connection string.

### 5a. Run the retrieval inspector web app

The inspector is a local FastAPI + React UI for viewing retrieval results only. It does not call an LLM, create prompts, stream tokens, or store search history.

```bash
source venv/bin/activate
python -m pip install -r requirements.txt
docker compose up -d db
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/` and search with a request body equivalent to:

```json
{ "query": "How can GST registration be cancelled?", "top_k": 10 }
```

The `POST /search` endpoint returns final reranked results plus Dense, BM25, and RRF intermediate lists. Timings are measured per request with `time.perf_counter()`. BGE-M3 and the existing cross-encoder reranker are loaded once during FastAPI startup; that model initialization time is reported separately as metadata.

### 6. Parse CGST Rules

To inspect the Central GST Rules PDF and write rule/sub-rule JSON:

```bash
python scripts/inspect_gst_rules.py --json-output data/rules/gst_rules_rules.json
python scripts/validate_gst_rules.py data/rules/gst_rules_rules.json
```

This parser records chapter metadata on each rule and keeps clauses, provisos, explanations, and tables inside rule/sub-rule text.

### 7. Generate CGST Act chunks

After parsing and validating sections, generate structure-aware chunks in JSON without embeddings or database writes:

```bash
python scripts/chunk_gst_act_sections.py
```

The script keeps small sections whole, groups complete subsections for larger sections, and only uses token-overlap when a single subsection must be split.

### 8. Generate BGE-M3 embeddings for CGST chunks

After chunk inspection, generate one dense BGE-M3 embedding per chunk into a separate JSONL file:

```bash
python scripts/embed_gst_act_chunks.py
```

This writes `data/acts/central_gst_act_2017_bge_m3_embeddings.jsonl` for inspection only. It does not connect to PostgreSQL or create retrieval indexes.

### 9. Ingest Act chunks into Docker PostgreSQL/pgvector

After BGE-M3 embedding generation, initialize the Docker database schema and upsert all Act chunk embeddings:

```bash
docker compose up -d db
docker compose exec -T db psql -v ON_ERROR_STOP=1 -U postgres -d hybrid_rag < schema.sql
python ingest_act_chunks.py
```

`ingest_act_chunks.py` reads the existing `*_bge_m3_embeddings.jsonl` files, validates every vector is dimension `1024`, namespaces each JSON chunk id with the Act slug, and upserts by that stable `chunk_id` into `act_chunks`. The table has a cosine HNSW index on `embedding`.

### 10. Generate and ingest CGST Rules chunks

After validating `data/rules/gst_rules_rules.json`, generate structure-aware Rule chunks, embed them with BGE-M3, initialize the database schema, and upsert them into `rule_chunks`:

```bash
python scripts/chunk_gst_rules.py
python scripts/embed_gst_rule_chunks.py
docker compose up -d db
docker compose exec -T db psql -v ON_ERROR_STOP=1 -U postgres -d hybrid_rag < schema.sql
python ingest_rule_chunks.py
```

`rules_chunker.py` keeps small rules whole, groups complete sub-rules for larger rules, and only token-splits inside an oversized sub-rule. `ingest_rule_chunks.py` reads `data/rules/gst_rules_bge_m3_embeddings.jsonl`, validates every vector is dimension `1024`, and upserts by `chunk_id`. The separate `rule_chunks` table has a cosine HNSW index on `embedding`.

### 11. Stop the database

```bash
docker compose down
```

The named database volume is retained for future runs.

## Repository contents

| Path | Purpose |
| --- | --- |
| `ingest_gst.py` | Validates GST CSVs, normalizes fields, embeds descriptions, and upserts records. |
| `tests/test_pgvector.py` | Compatibility entry point for the GST loader. |
| `tests/test_ingest_gst.py` | CSV/rate/code parsing tests and a transactional PostgreSQL ingestion test. |
| `search_pgvec.py` | Runs vector retrieval, cross-encoder reranking, and BM25 keyword search with trigram fallback. |
| `keyword_search.py` | BM25 SQL, trigram fallback routing, and exact code lookup without model-loading side effects. |
| `vector_search.py` | pgvector semantic retrieval helper reused by the CLI and RRF. |
| `rrf.py` | Reciprocal Rank Fusion over BM25 and vector result ranks. |
| `gst_act_parser.py` | Parses GST Act PDFs into sections/subsections without database or embedding work. |
| `gst_rules_parser.py` | Parses CGST Rules PDF into chapters, rules, and sub-rules. |
| `scripts/inspect_gst_act_sections.py` | Prints parsed GST Act section previews and writes inspected JSON. |
| `scripts/inspect_gst_rules.py` | Prints parsed CGST Rules previews and writes inspected JSON. |
| `scripts/validate_gst_rules.py` | Validates parsed CGST Rules structure. |
| `scripts/validate_sections.py` | Validates parsed section JSON before chunking. |
| `act_chunker.py` | Builds structure-aware CGST Act chunks from parsed sections/subsections. |
| `rules_chunker.py` | Builds structure-aware CGST Rules chunks from parsed rules/sub-rules. |
| `scripts/chunk_gst_act_sections.py` | Generates chunk JSON and prints chunk size/sample reports. |
| `scripts/chunk_gst_rules.py` | Generates CGST Rules chunk JSON and prints chunk size/sample reports. |
| `act_embedder.py` | Generates dense BGE-M3 embeddings for chunk JSON records. |
| `ingest_act_chunks.py` | Upserts Act chunk embeddings into Docker PostgreSQL/pgvector. |
| `ingest_rule_chunks.py` | Upserts CGST Rules chunk embeddings into Docker PostgreSQL/pgvector. |
| `scripts/embed_gst_act_chunks.py` | Writes chunk embeddings to JSONL or JSON for inspection. |
| `scripts/embed_gst_rule_chunks.py` | Writes CGST Rules chunk embeddings to JSONL or JSON for inspection. |
| `tests/test_gst_act_parser.py` | Unit tests for section heading, omission, duplicate, and chapter parsing. |
| `tests/test_rules_chunker.py` | Unit tests for CGST Rules chunking behavior. |
| `tests/test_ingest_rule_chunks.py` | Unit tests for CGST Rules embedding ingestion validation/upsert behavior. |
| `tests/dummy_search_test.py` | Prints BM25, vector, and RRF sections for manual command-line smoke tests. |
| `tests/test_keyword_search.py` | Routing tests and optional PostgreSQL integration tests. |
| `tests/test_rrf.py` | Unit tests for rank fusion, single-list results, duplicate merging, and ordering. |
| `Dockerfile.db` | Adds Timescale `pg_textsearch` 1.4.0 to the PostgreSQL 17/pgvector image. |
| `schema.sql` | Enables extensions and creates GST, Act chunk, and Rule chunk tables/indexes. |
| `docker-compose.yml` | Configures the PostgreSQL/pgvector container and persistent volume. |
| `requirements.txt` | Lists direct Python dependencies without version pins. |
| `.env.example` | Template for local environment values such as `HF_TOKEN`. |
| `docs/project-overview.md` | Detailed architecture and inventory of tools and techniques. |
| `data/gst/csvs/` | Local Kaggle CSVs, ignored by Git. |
| `data/acts/central_gst_act_2017_chunks.json` | Generated CGST Act chunks for inspection before retrieval ingestion. |
| `data/acts/central_gst_act_2017_bge_m3_embeddings.jsonl` | Generated BGE-M3 chunk embeddings, ignored if produced locally. |
| `data/rules/gst_rules_chunks.json` | Generated CGST Rules chunks for inspection before retrieval ingestion. |
| `data/rules/gst_rules_bge_m3_embeddings.jsonl` | Generated CGST Rules BGE-M3 chunk embeddings. |
| `docs/gst-dataset.md` | Inspected files, source columns, schema, cleaning, and field mapping. |

## Configuration and troubleshooting

By default, ingestion and search connect to `dbname=hybrid_rag user=postgres password=postgres host=localhost port=5432`. These match the local Compose configuration and are development credentials. Changing Compose settings alone does not update the default Python connection string; pass `--database-url`.

The scripts load `.env` through `python-dotenv`. To authenticate Hugging Face model downloads, set a real read token in `.env` or in your shell:

```bash
HF_TOKEN="hf_your_read_token_here"
```

| Symptom | What to check |
| --- | --- |
| Connection refused | Start the database, confirm readiness, and check whether another service occupies port `5432`. |
| Vector registration fails or `gst_documents` does not exist | Run the extension and table initialization above in the `hybrid_rag` database. |
| `similarity` or the trigram operator is unavailable | Enable `pg_trgm` in the database used by the scripts. |
| BM25 operator/index is unavailable | Build the new database image, confirm `pg_textsearch` is preloaded, and run `schema.sql`. |
| Model download fails | Check network access or availability of the model files in the local cache. |
| Hugging Face still says unauthenticated | Confirm `HF_TOKEN` is visible to Python and is not the placeholder from `.env.example`. |
| Same code appears more than once | Several GST source entries can share a classification; inspect description, rates, and source metadata. Reimports do not duplicate source rows. |
| CSV decoding or columns fail | Use the original Goods.csv and Services.csv; the loader accepts UTF-8 and Windows-1252 and validates headers. |
| Fuzzy output is empty | Matching uses a threshold against the entire description; a typo query is not guaranteed to pass it. |

Run the parsing and keyword routing tests with:

```bash
venv/bin/python -m unittest discover -s tests -v
```

To also run PostgreSQL integration tests against the initialized PostgreSQL 17 database:

```bash
TEST_DATABASE_URL='dbname=hybrid_rag user=postgres password=postgres host=localhost port=5432' venv/bin/python -m unittest discover -s tests -v
```

Integration tests use a temporary table and roll back without changing stored documents. They cover ingestion upserts and stale-row removal, exact codes, BM25 matching and score ordering, stemming, stopwords, limits, typo fallback, and empty results. There is no retrieval-quality evaluation dataset or assertion of model rankings.
# pg-vec-demo
