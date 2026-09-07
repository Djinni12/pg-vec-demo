import psycopg
from sentence_transformers import SentenceTransformer, CrossEncoder
from pgvector.psycopg import register_vector

model = SentenceTransformer("all-MiniLM-L6-v2")




conn = psycopg.connect(
    "dbname=hybrid_rag user=postgres password=postgres host=localhost port=5432"
)



register_vector(conn)

data = [
    ("I10", "Essential hypertension"),
    ("I11.9", "Hypertensive heart disease without heart failure"),
    ("I12.9", "Hypertensive chronic kidney disease"),
    ("I15.0", "Renovascular hypertension"),
    ("I16.9", "Hypertensive crisis, unspecified"),

    ("R03.0", "Elevated blood-pressure reading, without diagnosis of hypertension"),
    ("I95.0", "Idiopathic hypotension"),
    ("I95.9", "Hypotension, unspecified"),

    ("E11.9", "Type 2 diabetes mellitus without complications"),
    ("E11.65", "Type 2 diabetes mellitus with hyperglycemia"),
    ("E78.5", "Hyperlipidemia, unspecified"),

    ("I25.10", "Atherosclerotic heart disease of native coronary artery"),
    ("I50.9", "Heart failure, unspecified"),
    ("I48.91", "Unspecified atrial fibrillation"),

    ("J45.909", "Unspecified asthma, uncomplicated"),
    ("J44.9", "Chronic obstructive pulmonary disease, unspecified"),

    ("N18.9", "Chronic kidney disease, unspecified"),
    ("N17.9", "Acute kidney failure, unspecified"),

    ("R07.9", "Chest pain, unspecified"),
    ("R06.02", "Shortness of breath"),

    ("G43.909", "Migraine, unspecified, not intractable"),
    ("M54.50", "Low back pain, unspecified"),
]

with conn.cursor() as cur:
    for code, description in data:
        embedding = model.encode(description)

        cur.execute(
            """
            INSERT INTO documents (code, description, embedding)
            VALUES (%s, %s, %s)
            """,
            (code, description, embedding),
        )

conn.commit()
conn.close()

print("Inserted test documents.")