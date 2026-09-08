"""Compatibility wrapper for the query-only BGE-M3 vs LaBSE embedding check.

LaBSE corpus search was intentionally removed because the user only wants to embed
the query with a different model, not the whole dataset.
"""

from __future__ import annotations

from compare_query_embeddings import main


if __name__ == "__main__":
    main()
