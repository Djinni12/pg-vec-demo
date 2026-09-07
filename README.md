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

Download and extract `Goods.csv` and `Services.csv` from the Kaggle dataset into `data/gst/`. The files downloaded during this implementation are already there locally and ignored by Git. On a new checkout, a public download can be attempted with:

```bash
curl -fL 'https://www.kaggle.com/api/v1/datasets/download/prasad22/goods-and-service-tax-rates-dataset' -o /tmp/gst-kaggle.zip
python -m zipfile -e /tmp/gst-kaggle.zip data/gst
```

If Kaggle requires authentication, download through the dataset page instead.

```bash
python ingest_gst.py data/gst --dry-run
python ingest_gst.py data/gst
```

The dry run validates both files and prints row counts, skipped rows, encoding, and hashes without downloading models or writing to PostgreSQL. The import embeds descriptions and upserts GST records in one transaction. Repeat imports update existing records and remove stale rows from the same source files. `python test_pgvector.py data/gst` is a compatibility alias for the loader.

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

### 6. Stop the database

```bash
docker compose down
```

The named database volume is retained for future runs.

## Repository contents

| Path | Purpose |
| --- | --- |
| `ingest_gst.py` | Validates GST CSVs, normalizes fields, embeds descriptions, and upserts records. |
| `test_pgvector.py` | Compatibility entry point for the GST loader. |
| `test_ingest_gst.py` | CSV/rate/code parsing tests and a transactional PostgreSQL ingestion test. |
| `search_pgvec.py` | Runs vector retrieval, cross-encoder reranking, and BM25 keyword search with trigram fallback. |
| `keyword_search.py` | BM25 SQL, trigram fallback routing, and exact code lookup without model-loading side effects. |
| `vector_search.py` | pgvector semantic retrieval helper reused by the CLI and RRF. |
| `rrf.py` | Reciprocal Rank Fusion over BM25 and vector result ranks. |
| `dummy_search_test.py` | Prints BM25, vector, and RRF sections for manual command-line smoke tests. |
| `test_keyword_search.py` | Routing tests and optional PostgreSQL integration tests. |
| `test_rrf.py` | Unit tests for rank fusion, single-list results, duplicate merging, and ordering. |
| `Dockerfile.db` | Adds Timescale `pg_textsearch` 1.4.0 to the PostgreSQL 17/pgvector image. |
| `schema.sql` | Enables extensions and creates the GST table, BM25 index, and exact-code array index. |
| `docker-compose.yml` | Configures the PostgreSQL/pgvector container and persistent volume. |
| `requirements.txt` | Lists direct Python dependencies without version pins. |
| `.env.example` | Empty placeholder; the scripts do not currently load environment variables. |
| `docs/project-overview.md` | Detailed architecture and inventory of tools and techniques. |
| `data/gst/` | Local Kaggle CSVs, ignored by Git. |
| `src/` | Unused placeholder; code remains in the root scripts. |
| `docs/gst-dataset.md` | Inspected files, source columns, schema, cleaning, and field mapping. |
| `hello.txt` | Incidental text file, unused by the retrieval scripts. |

## Configuration and troubleshooting

By default, ingestion and search connect to `dbname=hybrid_rag user=postgres password=postgres host=localhost port=5432`. These match the local Compose configuration and are development credentials. Changing Compose settings alone does not update the default Python connection string; pass `--database-url`. Although `python-dotenv` is listed as a dependency, it is not used.

| Symptom | What to check |
| --- | --- |
| Connection refused | Start the database, confirm readiness, and check whether another service occupies port `5432`. |
| Vector registration fails or `gst_documents` does not exist | Run the extension and table initialization above in the `hybrid_rag` database. |
| `similarity` or the trigram operator is unavailable | Enable `pg_trgm` in the database used by the scripts. |
| BM25 operator/index is unavailable | Build the new database image, confirm `pg_textsearch` is preloaded, and run `schema.sql`. |
| Model download fails | Check network access or availability of the model files in the local cache. |
| Same code appears more than once | Several GST source entries can share a classification; inspect description, rates, and source metadata. Reimports do not duplicate source rows. |
| CSV decoding or columns fail | Use the original Goods.csv and Services.csv; the loader accepts UTF-8 and Windows-1252 and validates headers. |
| Fuzzy output is empty | Matching uses a threshold against the entire description; a typo query is not guaranteed to pass it. |

Run the parsing and keyword routing tests with:

```bash
venv/bin/python -m unittest test_ingest_gst test_keyword_search test_rrf -v
```

To also run PostgreSQL integration tests against the initialized PostgreSQL 17 database:

```bash
TEST_DATABASE_URL='dbname=hybrid_rag user=postgres password=postgres host=localhost port=5432' venv/bin/python -m unittest test_ingest_gst test_keyword_search test_rrf -v
```

Integration tests use a temporary table and roll back without changing stored documents. They cover ingestion upserts and stale-row removal, exact codes, BM25 matching and score ordering, stemming, stopwords, limits, typo fallback, and empty results. There is no retrieval-quality evaluation dataset or assertion of model rankings.
# pg-vec-demo
