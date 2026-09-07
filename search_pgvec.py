"""Search GST descriptions through separate semantic and keyword paths."""

import argparse

import psycopg
from pgvector.psycopg import register_vector

from ingest_gst import DATABASE_URL, MODEL_NAME
from keyword_search import exact_lookup, keyword_search


def print_result(row, score_label=None):
    code, description, score, cgst, sgst, igst, metadata = row
    raw_rates = metadata.get("rates_raw", {})
    rates = []
    for label, key, value in (("CGST", "cgst", cgst), ("SGST/UTGST", "sgst_utgst", sgst),
                              ("IGST", "igst", igst)):
        displayed = f"{value}%" if value is not None else raw_rates.get(key) or "unspecified"
        rates.append(f"{label}={displayed}")
    source = f"{metadata.get('source_file', '')}:{metadata.get('source_row', '')}"
    scoring = f" | {score_label}={score:.4f}" if score_label else ""
    print(f"{source} | {code} | {description} | {' | '.join(rates)}{scoring}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="roasted coffee beans")
    parser.add_argument("--exact-code", help="Look up an explicit HSN/SAC code without loading models")
    parser.add_argument("--database-url", default=DATABASE_URL)
    args = parser.parse_args()

    with psycopg.connect(args.database_url) as conn:
        if args.exact_code:
            results = exact_lookup(conn, args.exact_code)
            for row in results:
                print_result(row)
            if not results:
                print("No explicit code matches found.")
            return

        from sentence_transformers import SentenceTransformer, CrossEncoder

        register_vector(conn)
        model = SentenceTransformer(MODEL_NAME)
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        query_embedding = model.encode(args.query)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT code, description, embedding <=> %s AS distance,
                       cgst_rate_pct, sgst_utgst_rate_pct, igst_rate_pct, metadata
                FROM gst_documents
                ORDER BY embedding <=> %s
                LIMIT 10
                """,
                (query_embedding, query_embedding),
            )
            results = cur.fetchall()

        print("\nSEMANTIC RESULTS (RERANKED)\n")
        if results:
            scores = reranker.predict([(args.query, row[1]) for row in results])
            reranked = sorted(zip(results, scores), key=lambda item: float(item[1]), reverse=True)
            for row, score in reranked[:5]:
                print_result(row, "vector_distance")
                print(f"  rerank_score={float(score):.4f}")
        else:
            print("No semantic matches found. Run ingest_gst.py first.")

        results, method = keyword_search(conn, args.query)
        print(f"\nKEYWORD RESULTS ({method})\n")
        for row in results:
            print_result(row, "bm25_score" if method == "bm25" else "similarity")
        if not results:
            print("No keyword matches found.")


if __name__ == "__main__":
    main()
