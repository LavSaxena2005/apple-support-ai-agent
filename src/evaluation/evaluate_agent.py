import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import faiss
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.agent.pipeline import (
    classify_intent,
    decide_escalation,
    embedding_model
)


# ============================================================
# FILES
# ============================================================

GOLDEN_FILE = "evaluation/golden_set.csv"

CASES_FILE = (
    "data/processed/"
    "apple_support_cases_eval_safe.csv"
)

INDEX_FILE = (
    "data/processed/"
    "apple_support_eval_safe.faiss"
)

OUTPUT_FILE = (
    "evaluation/results/"
    "agent_predictions.csv"
)


# ============================================================
# LOAD SAFE HISTORICAL CASES
# ============================================================

print("Loading leakage-safe historical cases...")

cases_df = pd.read_csv(
    CASES_FILE
)

print(
    "Safe historical cases:",
    len(cases_df)
)


# ============================================================
# LOAD SAFE FAISS INDEX
# ============================================================

print("Loading leakage-safe FAISS index...")

safe_index = faiss.read_index(
    INDEX_FILE
)

print(
    "Safe FAISS index:",
    safe_index.ntotal,
    "cases"
)


# ============================================================
# SAFE RETRIEVAL FUNCTION
# ============================================================

def retrieve_safe_cases(
    message: str,
    top_k: int = 5
):

    # Create embedding
    query_embedding = embedding_model.encode(
        [message],
        convert_to_numpy=True
    )

    # Make sure float32
    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # Normalize
    faiss.normalize_L2(
        query_embedding
    )

    # Search SAFE index
    scores, indices = safe_index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        row = cases_df.iloc[int(idx)]

        results.append({
            "score": float(score),
            "customer_text": str(
                row["customer_text"]
            ),
            "brand_reply": str(
                row["brand_reply"]
            ),
            "customer_tweet_id": str(
                row["customer_tweet_id"]
            ),
            "brand_reply_id": str(
                row["brand_reply_id"]
            )
        })

    return results


# ============================================================
# LOAD GOLDEN SET
# ============================================================

print("\nLoading golden set...")

golden = pd.read_csv(
    GOLDEN_FILE
)

print(
    "Golden examples:",
    len(golden)
)


# ============================================================
# RUN EVALUATION
# ============================================================

print("\nRunning leakage-safe evaluation...")

print(
    "Qwen will NOT be called."
)

print(
    "Intent + retrieval + escalation "
    "will be evaluated."
)

print()


results = []


for i, row in golden.iterrows():

    text = str(row["text"])

    try:

        # ----------------------------------------------------
        # Intent
        # ----------------------------------------------------

        predicted_intent, confidence = (
            classify_intent(text)
        )


        # ----------------------------------------------------
        # Safe historical retrieval
        # ----------------------------------------------------

        retrieved_cases = retrieve_safe_cases(
            text,
            top_k=5
        )


        # ----------------------------------------------------
        # Escalation
        # ----------------------------------------------------

        decision, reason = decide_escalation(
            message=text,
            intent=predicted_intent,
            confidence=confidence,
            retrieved_cases=retrieved_cases
        )


        predicted_escalate = (
            decision == "ESCALATE"
        )


        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append({

            "tweet_id":
                row["tweet_id"],

            "text":
                text,

            "true_intent":
                row["intent"],

            "predicted_intent":
                predicted_intent,

            "intent_confidence":
                confidence,

            "true_escalate":
                bool(row["escalate"]),

            "predicted_escalate":
                predicted_escalate,

            "agent_decision":
                decision,

            "agent_reason":
                reason,

            "top_retrieval_score":
                (
                    retrieved_cases[0]["score"]
                    if retrieved_cases
                    else 0
                ),

            "retrieved_evidence":
                " || ".join(
                    [
                        case["brand_reply"]
                        for case in retrieved_cases
                    ]
                )
        })


    except Exception as error:

        print(
            f"ERROR on example "
            f"{i + 1}: {error}"
        )

        results.append({

            "tweet_id":
                row["tweet_id"],

            "text":
                text,

            "true_intent":
                row["intent"],

            "predicted_intent":
                "ERROR",

            "intent_confidence":
                0,

            "true_escalate":
                bool(row["escalate"]),

            "predicted_escalate":
                True,

            "agent_decision":
                "ESCALATE",

            "agent_reason":
                str(error),

            "top_retrieval_score":
                0,

            "retrieved_evidence":
                ""
        })


    # Progress
    if (i + 1) % 25 == 0:

        print(
            f"Processed "
            f"{i + 1}/{len(golden)}"
        )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions = pd.DataFrame(
    results
)


Path(
    "evaluation/results"
).mkdir(
    parents=True,
    exist_ok=True
)


predictions.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# REMOVE ERRORS
# ============================================================

valid = predictions[
    predictions["predicted_intent"] != "ERROR"
].copy()


print("\n")
print(
    "Valid examples:",
    len(valid)
)

print(
    "Failed examples:",
    len(predictions) - len(valid)
)


# ============================================================
# INTENT METRICS
# ============================================================

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


print("\n")
print("=" * 60)
print(
    "APPLE SUPPORT AI AGENT — INTENT RESULTS"
)
print("=" * 60)


print(
    "Intent Accuracy:",
    round(intent_accuracy, 4)
)


print(
    "Intent Macro-F1:",
    round(intent_macro_f1, 4)
)


print("\nClassification Report:")


print(
    classification_report(
        valid["true_intent"],
        valid["predicted_intent"],
        zero_division=0
    )
)


# ============================================================
# ESCALATION METRICS
# ============================================================

escalation_accuracy = accuracy_score(
    valid["true_escalate"],
    valid["predicted_escalate"]
)


escalation_f1 = f1_score(
    valid["true_escalate"],
    valid["predicted_escalate"],
    average="binary",
    zero_division=0
)


print("\n")
print("=" * 60)
print(
    "APPLE SUPPORT AI AGENT — "
    "ESCALATION RESULTS"
)
print("=" * 60)


print(
    "Escalation Accuracy:",
    round(escalation_accuracy, 4)
)


print(
    "Escalation F1:",
    round(escalation_f1, 4)
)


# ============================================================
# ESCALATION CONFUSION MATRIX
# ============================================================

print("\n")
print("=" * 60)
print(
    "ESCALATION CONFUSION MATRIX"
)
print("=" * 60)


cm = confusion_matrix(
    valid["true_escalate"],
    valid["predicted_escalate"],
    labels=[False, True]
)


print(
    "Rows    = True  [False, True]"
)

print(
    "Columns = Pred  [False, True]"
)

print()

print(cm)


# ============================================================
# RETRIEVAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print(
    "RETRIEVAL SUMMARY"
)
print("=" * 60)


print(
    "Average top retrieval score:",
    round(
        valid[
            "top_retrieval_score"
        ].mean(),
        4
    )
)


print(
    "Minimum top retrieval score:",
    round(
        valid[
            "top_retrieval_score"
        ].min(),
        4
    )
)


print(
    "Maximum top retrieval score:",
    round(
        valid[
            "top_retrieval_score"
        ].max(),
        4
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)


print(
    f"Intent Accuracy      : "
    f"{intent_accuracy:.4f}"
)


print(
    f"Intent Macro-F1      : "
    f"{intent_macro_f1:.4f}"
)


print(
    f"Escalation Accuracy  : "
    f"{escalation_accuracy:.4f}"
)


print(
    f"Escalation F1        : "
    f"{escalation_f1:.4f}"
)


print("\nPredictions saved to:")

print(
    OUTPUT_FILE
)


print("\nEvaluation complete.")