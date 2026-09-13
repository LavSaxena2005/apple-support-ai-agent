import re
import time
from pathlib import Path

import pandas as pd
import requests


INPUT_FILE = Path(
    "evaluation/results/leakage_safe_eval_50.csv"
)

OUTPUT_FILE = Path(
    "evaluation/results/llm_judge_50.csv"
)

OLLAMA_URL = (
    "http://localhost:11434/api/generate"
)

MODEL = "qwen2.5:3b"


def judge_reply(row):

    customer = str(row["text"])
    evidence = str(row["evidence"])
    reply = str(row["reply"])

    prompt = f"""
Task: rate an Apple Support AI reply.

Customer:
{customer}

Historical support evidence:
{evidence}

AI reply:
{reply}

Rate the AI reply:

5 = excellent
4 = good
3 = acceptable
2 = poor
1 = very poor

Return ONLY one digit: 1, 2, 3, 4, or 5.
"""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "temperature": 0,
        "options": {
            "num_predict": 100
        }
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get(
            "response",
            ""
        ).strip()

        print(
            f"Model response: {answer[:100]}"
        )

        # Remove thinking block
        if "<think>" in answer:

            if "</think>" in answer:

                answer = answer.split(
                    "</think>",
                    1
                )[1].strip()

        # Look for a standalone 1-5
        matches = re.findall(
            r"(?<!\d)([1-5])(?!\d)",
            answer
        )

        if not matches:

            raise ValueError(
                "No valid score found"
            )

        # Take the final score mentioned
        score = int(
            matches[-1]
        )

        if score not in [1, 2, 3, 4, 5]:

            raise ValueError(
                f"Invalid score: {score}"
            )

        return score

    except Exception as e:

        print(
            f"Judge failed for "
            f"{row['tweet_id']}: {e}"
        )

        return None


def main():

    if not INPUT_FILE.exists():

        print(
            f"ERROR: Input file not found: "
            f"{INPUT_FILE}"
        )

        return

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Loaded {len(df)} evaluation examples."
    )

    # Always start from zero.
    if OUTPUT_FILE.exists():

        OUTPUT_FILE.unlink()

        print(
            "Removed previous judge results."
        )

    results = []

    for i, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        tweet_id = str(
            row["tweet_id"]
        )

        print(
            f"\nJudging {i}/{len(df)}: "
            f"{tweet_id}"
        )

        score = judge_reply(
            row
        )

        # Retry once
        if score is None:

            print(
                "Retrying..."
            )

            time.sleep(1)

            score = judge_reply(
                row
            )

        results.append(
            {
                "tweet_id": tweet_id,
                "llm_overall": score
            }
        )

        # Save immediately
        pd.DataFrame(
            results
        ).to_csv(
            OUTPUT_FILE,
            index=False
        )

        if score is not None:

            print(
                f"LLM score: {score}/5"
            )

        else:

            print(
                "LLM score: FAILED"
            )

        time.sleep(0.2)

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    final_df = pd.DataFrame(
        results
    )

    successful = (
        final_df["llm_overall"]
        .notna()
        .sum()
    )

    failed = (
        final_df["llm_overall"]
        .isna()
        .sum()
    )

    print(
        "\n======================================"
    )

    print(
        "LLM JUDGE COMPLETE"
    )

    print(
        "======================================"
    )

    print(
        f"Total examples: {len(final_df)}"
    )

    print(
        f"Successful judgments: {successful}"
    )

    print(
        f"Failed judgments: {failed}"
    )

    if successful > 0:

        average = (
            final_df["llm_overall"]
            .dropna()
            .mean()
        )

        print(
            f"Average LLM score: "
            f"{average:.2f}/5"
        )

        print(
            "\nScore distribution:"
        )

        print(
            final_df[
                "llm_overall"
            ]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":

    main()