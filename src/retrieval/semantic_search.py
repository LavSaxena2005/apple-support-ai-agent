import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path


# ============================================================
# FILE PATH
# ============================================================

CASES_FILE = Path("data/processed/apple_support_cases.csv")


# ============================================================
# LOAD DATA
# ============================================================

print("Loading historical AppleSupport cases...")

df = pd.read_csv(CASES_FILE)

print("Total historical cases:", len(df))


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print("Creating embeddings...")

texts = df["customer_text"].fillna("").astype(str).tolist()

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
)


# ============================================================
# NORMALIZE EMBEDDINGS
# ============================================================

faiss.normalize_L2(embeddings)


# ============================================================
# CREATE FAISS INDEX
# ============================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("FAISS index created.")

print("Number of indexed cases:", index.ntotal)


# ============================================================
# SEARCH FUNCTION
# ============================================================

def search_similar_cases(query, top_k=5):

    # Convert query to embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    # Normalize
    faiss.normalize_L2(query_embedding)

    # Search
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        row = df.iloc[idx]

        results.append(
            {
                "score": float(score),
                "customer_text": row["customer_text"],
                "brand_reply": row["brand_reply"],
            }
        )

    return results


# ============================================================
# TEST SEARCH
# ============================================================

if __name__ == "__main__":

    print()
    print("==============================================")
    print("Testing Semantic Search")
    print("==============================================")

    query = input(
        "\nEnter a customer problem: "
    )

    results = search_similar_cases(
        query,
        top_k=5
    )

    print()
    print("Similar historical cases:")
    print("----------------------------------------------")

    for i, result in enumerate(results, start=1):

        print()
        print(f"RESULT {i}")
        print(f"Similarity Score: {result['score']:.4f}")

        print()
        print("Customer:")
        print(result["customer_text"])

        print()
        print("AppleSupport Reply:")
        print(result["brand_reply"])

        print("----------------------------------------------")