# Upgrade the local database for BM25

Timescale `pg_textsearch` 1.4.0 supports PostgreSQL 17 and 18; the previous project image used PostgreSQL 16. The new Compose configuration builds a PostgreSQL 17 image with both pgvector and `pg_textsearch`, preloads the latter, and uses a separate `pgdata17` volume. See the [extension's installation requirements](https://github.com/timescale/pg_textsearch/blob/v1.4.0/README.md).

Do not mount the old PostgreSQL 16 data directory into PostgreSQL 17. The commands below preserve the old volume and transfer the database with a logical backup. Run them from the repository root with Docker access. No upgrade has been applied automatically.

## Existing running PostgreSQL 16 database

1. Back up the existing container before replacing it. Choose an unused backup filename if this one already exists:

   ```bash
   docker exec hybrid-rag-db pg_dump -U postgres -d hybrid_rag --no-owner --no-acl > hybrid_rag_pg16.sql
   ```

   Confirm the command succeeds and the file contains the database dump. Keep the old volume and backup until the new database is verified. If the old container is stopped, start that existing container first with `docker start hybrid-rag-db`.

2. Build and start the new database:

   ```bash
   docker compose up -d --build db
   docker compose exec db pg_isready -U postgres -d hybrid_rag
   ```

   Wait for readiness. Compose replaces the old service container but does not delete its old named volume. The new `pgdata17` volume starts empty; if you have already used it, restore into a separate empty database instead of overwriting existing tables.

3. Restore the backup into the fresh `hybrid_rag` database **before** running `schema.sql`:

   ```bash
   docker compose exec -T db psql -v ON_ERROR_STOP=1 -U postgres -d hybrid_rag < hybrid_rag_pg16.sql
   docker compose exec -T db psql -v ON_ERROR_STOP=1 -U postgres -d hybrid_rag < schema.sql
   ```

   Existing code, description, and embedding rows are preserved. The schema script adds the BM25 extension and index. Do not rerun the sample loader after restoring unless you intend to append duplicate samples.

4. Verify extensions, data, and keyword retrieval:

   ```bash
   docker compose exec db psql -U postgres -d hybrid_rag -c 'SELECT extname, extversion FROM pg_extension;'
   docker compose exec db psql -U postgres -d hybrid_rag -c 'SELECT count(*) FROM documents;'
   TEST_DATABASE_URL='dbname=hybrid_rag user=postgres password=postgres host=localhost port=5432' venv/bin/python -m unittest test_keyword_search -v
   python search_pgvec.py
   ```

The integration tests use a temporary table and index and roll back, preserving the restored data. Fresh installations can follow the README without this backup/restore procedure.
