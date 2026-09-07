import psycopg
from sentence_transformers import SentenceTransformer,CrossEncoder
from pgvector.psycopg import register_vector

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

conn = psycopg.connect(
    "dbname=hybrid_rag user=postgres password=postgres host=localhost port=5432"
)

register_vector(conn)

query = "poorly controlled high blood pressure"
query_embedding = model.encode(query)

with conn.cursor() as cur:
    cur.execute(
        """
        SELECT
            code,
            description,
            embedding <=> %s AS distance
        FROM documents
        ORDER BY embedding <=> %s
        LIMIT 10;
        """,
        (query_embedding, query_embedding),
    )

    results = cur.fetchall()
pairs = [
    (query, description)
    for code, description, distance in results
]
rerank_scores = reranker.predict(pairs)

reranked = []

for result, score in zip(results, rerank_scores):
    code, description, distance = result

    reranked.append(
        (code, description, distance, float(score))
    )
reranked.sort(key=lambda x: x[3], reverse=True)

print("\nRERANKED RESULTS\n")

for code, description, distance, score in reranked[:5]:
    print(
        f"{code} | {description} | "
        f"vector_distance={distance:.4f} | "
        f"rerank_score={score:.4f}"
    )

# for code, description, distance in results:
#     print(f"{code} | {description} | distance={distance:.4f}")



def fuzzy_search(conn, query, limit=5):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                code,
                description,
                similarity(description, %s) AS score
            FROM documents
            WHERE description %% %s
            ORDER BY score DESC
            LIMIT %s
            """,
            (query, query, limit)
        )

        return cur.fetchall()

query = "hypertensoin"

results = fuzzy_search(conn, query)

print("\nFUZZY RESULTS\n")

for code, description, score in results:
    print(
        f"{code} | {description} | "
        f"similarity={score:.4f}"
    )

conn.close()