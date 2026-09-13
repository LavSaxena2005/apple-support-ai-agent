import pandas as pd
from pathlib import Path

from src.agent.pipeline import run_agent


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = "evaluation/reply_eval_50.csv"

OUTPUT_FILE = (
    "evaluation/results/"
    "reply_eval_50_results.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("APPLE SUPPORT — REPLY GENERATION EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load evaluation set
    # --------------------------------------------------------

    print("\nLoading evaluation set...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Examples: {len(df)}")

    results = []

    # --------------------------------------------------------
    # Process every example
    # --------------------------------------------------------

    for i, row in df.iterrows():

        message = str(row["text"])

        print("\n" + "-" * 60)
        print(
            f"Example {i + 1}/{len(df)}"
        )

        print(
            "Customer:",
            message
        )

        try:

            # ------------------------------------------------
            # Run complete AI agent
            # ------------------------------------------------

            result = run_agent(message)

            # ------------------------------------------------
            # Convert evidence safely to text
            # ------------------------------------------------

            evidence_items = []

            if result.evidence:

                for item in result.evidence:

                    # Evidence can already be a string
                    if isinstance(item, str):

                        evidence_items.append(item)

                    # Evidence can also be a dictionary
                    elif isinstance(item, dict):

                        evidence_items.append(
                            str(
                                item.get(
                                    "brand_reply",
                                    ""
                                )
                            )
                        )

                    # Fallback for any other object
                    else:

                        evidence_items.append(
                            str(item)
                        )

            evidence_text = " || ".join(
                evidence_items
            )

            # ------------------------------------------------
            # Save result
            # ------------------------------------------------

            results.append({

                "tweet_id":
                    row["tweet_id"],

                "text":
                    message,

                "true_intent":
                    row["intent"],

                "true_escalate":
                    row["escalate"],

                "predicted_intent":
                    result.intent,

                "confidence":
                    result.confidence,

                "decision":
                    result.decision,

                "reason":
                    result.reason,

                "reply":
                    result.reply,

                "evidence":
                    evidence_text
            })

            # ------------------------------------------------
            # Display result
            # ------------------------------------------------

            print(
                "Intent:",
                result.intent
            )

            print(
                "Confidence:",
                result.confidence
            )

            print(
                "Decision:",
                result.decision
            )

            print(
                "Reply:",
                result.reply
            )

        except Exception as error:

            # ------------------------------------------------
            # Handle failed example
            # ------------------------------------------------

            print(
                "ERROR:",
                error
            )

            results.append({

                "tweet_id":
                    row["tweet_id"],

                "text":
                    message,

                "true_intent":
                    row["intent"],

                "true_escalate":
                    row["escalate"],

                "predicted_intent":
                    "ERROR",

                "confidence":
                    0,

                "decision":
                    "ERROR",

                "reason":
                    str(error),

                "reply":
                    "",

                "evidence":
                    ""
            })

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (i + 1) % 10 == 0:

            print(
                "\nProgress: "
                f"{i + 1}/{len(df)}"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output = pd.DataFrame(results)

    Path(
        "evaluation/results"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    successful = (
        output["predicted_intent"] != "ERROR"
    ).sum()

    failed = (
        output["predicted_intent"] == "ERROR"
    ).sum()

    print("\n" + "=" * 60)
    print("REPLY EVALUATION COMPLETE")
    print("=" * 60)

    print(
        "Total examples:",
        len(output)
    )

    print(
        "Successful:",
        successful
    )

    print(
        "Failed:",
        failed
    )

    print("\nSaved to:")

    print(
        OUTPUT_FILE
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()