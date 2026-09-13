import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CASES_FILE = "data/processed/apple_support_cases.csv"

GOLDEN_FILE = "evaluation/golden_set.csv"

INDEX_FILE = "data/processed/apple_support_eval_safe.faiss"

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# LOAD HISTORICAL CASES
# ============================================================

print("=" * 60)
print("Building leakage-safe FAISS index")
print("=" * 60)

print()

print("Loading historical cases...")

cases = pd.read_csv(
    CASES_FILE
)

print(
    f"Total historical cases: {len(cases)}"
)


# ============================================================
# LOAD GOLDEN SET
# ============================================================

print()

print("Loading golden evaluation set...")

golden = pd.read_csv(
    GOLDEN_FILE
)

print(
    f"Golden examples: {len(golden)}"
)


# ============================================================
# PREPARE IDS
# ============================================================

cases["customer_tweet_id"] = (
    cases["customer_tweet_id"]
    .astype(str)
)

golden["tweet_id"] = (
    golden["tweet_id"]
    .astype(str)
)


# ============================================================
# REMOVE GOLDEN EXAMPLES
# ============================================================

golden_ids = set(
    golden["tweet_id"]
)


before_count = len(cases)


cases_safe = cases[
    ~cases["customer_tweet_id"].isin(
        golden_ids
    )
].copy()


after_count = len(cases_safe)


removed_count = (
    before_count - after_count
)


print()

print(
    f"Cases before filtering: {before_count}"
)

print(
    f"Golden cases removed: {removed_count}"
)

print(
    f"Cases used for retrieval: {after_count}"
)


# ============================================================
# SAVE SAFE CASE DATA
# ============================================================

SAFE_CASES_FILE = (
    "data/processed/"
    "apple_support_cases_eval_safe.csv"
)


cases_safe.to_csv(
    SAFE_CASES_FILE,
    index=False
)


print()

print(
    f"Saved safe cases: {SAFE_CASES_FILE}"
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print()

print("Loading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)

print(
    "Embedding model ready."
)


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print()

print(
    "Creating embeddings..."
)

texts = (
    cases_safe["customer_text"]
    .fillna("")
    .astype(str)
    .tolist()
)


embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True,
    batch_size=64
)


# ============================================================
# NORMALIZE
# ============================================================

print()

print(
    "Normalizing embeddings..."
)

faiss.normalize_L2(
    embeddings
)


# ============================================================
# CREATE FAISS INDEX
# ============================================================

dimension = (
    embeddings.shape[1]
)


index = faiss.IndexFlatIP(
    dimension
)


index.add(
    embeddings
)


print()

print(
    f"FAISS index contains "
    f"{index.ntotal} cases."
)


# ============================================================
# SAVE INDEX
# ============================================================

faiss.write_index(
    index,
    INDEX_FILE
)


print()

print(
    f"Saved leakage-safe index:"
)

print(
    INDEX_FILE
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()

print("=" * 60)

print(
    "Leakage-safe index completed"
)

print("=" * 60)

print()

print(
    f"Original cases : {before_count}"
)

print(
    f"Golden cases   : {len(golden)}"
)

print(
    f"Removed        : {removed_count}"
)

print(
    f"Retrieval cases: {after_count}"
)

print()

print(
    "Evaluation retrieval is now safer."
)