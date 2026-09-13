import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path


CASES_FILE = "data/processed/apple_support_cases.csv"

INDEX_FILE = "data/processed/apple_support.faiss"

MODEL_NAME = "all-MiniLM-L6-v2"


print("Loading historical cases...")

cases = pd.read_csv(CASES_FILE)

print("Cases:", len(cases))


print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)


print("\nCreating embeddings...")

texts = (
    cases["customer_text"]
    .fillna("")
    .astype(str)
    .tolist()
)

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True
)


print("\nNormalizing embeddings...")

faiss.normalize_L2(embeddings)


print("\nCreating FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)


Path(
    "data/processed"
).mkdir(
    parents=True,
    exist_ok=True
)


faiss.write_index(
    index,
    INDEX_FILE
)


print("\n================================")
print("FAISS INDEX CREATED")
print("================================")

print("Cases:", index.ntotal)

print("Saved:", INDEX_FILE)