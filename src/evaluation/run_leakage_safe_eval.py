
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Import the existing agent module.
from src.agent import pipeline


PROJECT_ROOT = Path(".")
TWCS_FILE = PROJECT_ROOT / "data" / "raw" / "twcs.csv"
CASES_FILE = PROJECT_ROOT / "data" / "processed" / "apple_support_cases.csv"
GOLDEN_FILE = PROJECT_ROOT / "evaluation" / "golden_set.csv"
RESULT_FILE = PROJECT_ROOT / "evaluation" / "results" / "leakage_safe_eval_50.csv"

N_EVAL = 50
RANDOM_STATE = 42


def build_thread_map(twcs: pd.DataFrame):
    """
    Build tweet_id -> conversation/root_id.

    A tweet belongs to the same conversation as the tweet it replies to.
    We follow in_response_to_tweet_id until reaching the root.
    """
    parent = {}

    for _, row in twcs.iterrows():
        tweet_id = str(row["tweet_id"])
        parent_value = row["in_response_to_tweet_id"]

        if pd.isna(parent_value) or str(parent_value).strip() in ("", "nan", "None"):
            parent[tweet_id] = None
        else:
            parent[tweet_id] = str(parent_value)

    cache = {}

    def root_of(tweet_id):
        tweet_id = str(tweet_id)

        if tweet_id in cache:
            return cache[tweet_id]

        seen = set()
        current = tweet_id

        while current in parent and parent[current] is not None:
            if current in seen:
                # Defensive handling for malformed cycles.
                break

            seen.add(current)
            current = parent[current]

            if current in cache:
                current = cache[current]
                break

        root = current

        for item in seen:
            cache[item] = root

        cache[tweet_id] = root
        return root

    return root_of


def make_leakage_safe_index():
    """
    Remove every conversation containing a golden-set example
    from the retrieval corpus.

    This is stronger than merely removing the exact evaluation tweet.
    """
    print("Loading original TWCS data for conversation-level split...")
    twcs = pd.read_csv(TWCS_FILE, low_memory=False)

    print("Loading golden set...")
    golden = pd.read_csv(GOLDEN_FILE)

    if "tweet_id" not in golden.columns:
        raise ValueError("golden_set.csv must contain tweet_id")

    root_of = build_thread_map(twcs)

    golden_ids = {
        str(x)
        for x in golden["tweet_id"].dropna().tolist()
    }

    print(f"Golden examples: {len(golden_ids)}")

    golden_roots = set()

    for tweet_id in golden_ids:
        golden_roots.add(root_of(tweet_id))

    print(f"Golden conversations to hold out: {len(golden_roots)}")

    # Map every original tweet to its conversation root.
    # This is done only once for the source dataset.
    print("Building conversation IDs...")
    twcs["conversation_root"] = twcs["tweet_id"].astype(str).map(root_of)

    golden_thread_tweets = set(
        twcs.loc[
            twcs["conversation_root"].isin(golden_roots),
            "tweet_id"
        ].astype(str)
    )

    print(
        f"Tweets excluded from retrieval because they belong "
        f"to golden conversations: {len(golden_thread_tweets)}"
    )

    cases = pipeline.cases_df.copy()

    # A case is excluded if either its customer tweet or brand reply
    # belongs to a held-out golden conversation.
    customer_ids = cases["customer_tweet_id"].astype(str)
    reply_ids = cases["brand_reply_id"].astype(str)

    keep_mask = (
        ~customer_ids.isin(golden_thread_tweets)
        & ~reply_ids.isin(golden_thread_tweets)
    )

    filtered_cases = cases.loc[keep_mask].reset_index(drop=True)

    print(f"Original retrieval cases: {len(cases)}")
    print(f"Leakage-safe retrieval cases: {len(filtered_cases)}")

    if len(filtered_cases) == 0:
        raise RuntimeError("All retrieval cases were removed.")

    # Reuse the existing FAISS vectors. The saved index was built in
    # exactly the same row order as cases_df.
    print("Building temporary leakage-safe FAISS index...")
    vectors = pipeline.faiss_index.reconstruct_n(
        0,
        pipeline.faiss_index.ntotal
    )

    vectors = np.asarray(vectors, dtype="float32")

    # Keep the same rows as filtered_cases.
    original_positions = np.flatnonzero(keep_mask.to_numpy())

    filtered_vectors = vectors[original_positions].copy()

    # Normalize again so cosine similarity via inner product remains valid.
    faiss.normalize_L2(filtered_vectors)

    safe_index = faiss.IndexFlatIP(filtered_vectors.shape[1])
    safe_index.add(filtered_vectors)

    # Replace only the in-memory retrieval objects.
    pipeline.cases_df = filtered_cases
    pipeline.faiss_index = safe_index

    return golden


def run_evaluation(golden):
    print()
    print("=" * 70)
    print("LEAKAGE-SAFE 50-EXAMPLE EVALUATION")
    print("=" * 70)

    # Select 50 from the 200 human-labelled golden examples.
    eval_df = golden.sample(
        n=min(N_EVAL, len(golden)),
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    rows = []

    for i, row in eval_df.iterrows():
        message = str(row["text"]).strip()

        print()
        print(f"[{i + 1}/{len(eval_df)}] {message}")

        try:
            result = pipeline.run_agent(message)

            rows.append({
                "tweet_id": str(row["tweet_id"]),
                "text": message,
                "true_intent": str(row["intent"]),
                "true_escalate": str(row["escalate"]).strip().lower(),
                "predicted_intent": result.intent,
                "confidence": result.confidence,
                "decision": result.decision,
                "reason": result.reason,
                "reply": result.reply,
                "top_similarity": (
                    float(
                        result.evidence[0]
                        .split("|")[0]
                        .replace("Similarity", "")
                        .strip()
                    )
                    if result.evidence
                    else 0.0
                ),
                "evidence": " || ".join(result.evidence),
            })

            print(
                f"  True intent : {row['intent']}\n"
                f"  Pred intent : {result.intent}\n"
                f"  Decision    : {result.decision}"
            )

        except Exception as error:
            print(f"  ERROR: {error}")

            rows.append({
                "tweet_id": str(row["tweet_id"]),
                "text": message,
                "true_intent": str(row["intent"]),
                "true_escalate": str(row["escalate"]).strip().lower(),
                "predicted_intent": "ERROR",
                "confidence": 0.0,
                "decision": "ERROR",
                "reason": str(error),
                "reply": "",
                "top_similarity": 0.0,
                "evidence": "",
            })

    results = pd.DataFrame(rows)

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULT_FILE, index=False)

    valid = results[results["predicted_intent"] != "ERROR"].copy()

    if len(valid) == 0:
        raise RuntimeError("No successful evaluation examples.")

    intent_accuracy = accuracy_score(
        valid["true_intent"],
        valid["predicted_intent"]
    )

    intent_macro_f1 = f1_score(
        valid["true_intent"],
        valid["predicted_intent"],
        average="macro",
        zero_division=0
    )

    true_escalate = (
        valid["true_escalate"]
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False
        })
        .fillna(False)
    )

    predicted_escalate = valid["decision"].eq("ESCALATE")

    escalation_accuracy = accuracy_score(
        true_escalate,
        predicted_escalate
    )

    escalation_macro_f1 = f1_score(
        true_escalate,
        predicted_escalate,
        average="macro",
        zero_division=0
    )

    print()
    print("=" * 70)
    print("FINAL LEAKAGE-SAFE RESULTS")
    print("=" * 70)

    print(f"Examples evaluated       : {len(valid)}")
    print(f"Intent Accuracy          : {intent_accuracy:.4f}")
    print(f"Intent Macro-F1          : {intent_macro_f1:.4f}")
    print(f"Escalation Accuracy     : {escalation_accuracy:.4f}")
    print(f"Escalation Macro-F1     : {escalation_macro_f1:.4f}")

    print()
    print("Intent classification report:")
    print(
        classification_report(
            valid["true_intent"],
            valid["predicted_intent"],
            zero_division=0
        )
    )

    print(f"Saved results: {RESULT_FILE}")


def main():
    golden = make_leakage_safe_index()
    run_evaluation(golden)


if __name__ == "__main__":
    main()
