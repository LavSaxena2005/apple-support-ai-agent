import json
import re
import time
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "evaluation/results/reply_eval_50_results.csv"
)

OUTPUT_FILE = Path(
    "evaluation/results/reply_quality_20_results.csv"
)

OLLAMA_URL = (
    "http://localhost:11434/api/generate"
)

OLLAMA_MODEL = "qwen3:4b"

# First test 20 examples.
MAX_EXAMPLES = 20

# Qwen3 can be slow on local CPU.
TIMEOUT_SECONDS = 90


# ============================================================
# OLLAMA
# ============================================================

def call_ollama(prompt: str):

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,

        # Ask Ollama for JSON when possible.
        "format": "json",

        "options": {
            "temperature": 0,
            "num_predict": 250
        }
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=TIMEOUT_SECONDS
        )

        response.raise_for_status()

        data = response.json()

        return data.get(
            "response",
            ""
        ).strip()

    except requests.exceptions.Timeout:

        print(
            f"    Ollama timeout after "
            f"{TIMEOUT_SECONDS} seconds."
        )

        return None

    except Exception as error:

        print(
            f"    Ollama error: {error}"
        )

        return None


# ============================================================
# SCORE CLEANING
# ============================================================

def clean_score(value):

    if value is None:
        return 0

    # Handle strings such as "4/5"
    text = str(value)

    match = re.search(
        r"\b([1-5])\b",
        text
    )

    if not match:
        return 0

    score = int(
        match.group(1)
    )

    if score < 1 or score > 5:
        return 0

    return score


# ============================================================
# JSON EXTRACTION
# ============================================================

def try_json(text):

    if not text:
        return None

    text = str(
        text
    ).strip()

    # --------------------------------------------------------
    # First try entire response
    # --------------------------------------------------------

    try:

        result = json.loads(
            text
        )

        if isinstance(result, dict):
            return result

    except Exception:
        pass

    # --------------------------------------------------------
    # Remove thinking section
    # --------------------------------------------------------

    if "</think>" in text:

        after_think = text.split(
            "</think>",
            1
        )[1].strip()

        try:

            result = json.loads(
                after_think
            )

            if isinstance(result, dict):
                return result

        except Exception:
            pass

        text = after_think

    # --------------------------------------------------------
    # Find JSON object inside response
    # --------------------------------------------------------

    matches = re.findall(
        r"\{.*?\}",
        text,
        flags=re.DOTALL
    )

    # Try from the end because final answer
    # is usually near the end.
    for candidate in reversed(matches):

        try:

            result = json.loads(
                candidate
            )

            if isinstance(result, dict):

                return result

        except Exception:
            continue

    return None


# ============================================================
# TEXT SCORE EXTRACTION
# ============================================================

def extract_labeled_scores(text):

    if not text:
        return None

    text = str(
        text
    )

    # --------------------------------------------------------
    # Prefer final answer after </think>
    # --------------------------------------------------------

    if "</think>" in text:

        after_think = text.split(
            "</think>",
            1
        )[1]

        if after_think.strip():

            text = after_think

    # --------------------------------------------------------
    # Different possible spellings
    # --------------------------------------------------------

    patterns = {

        "correctness":
            r"correctness\s*[:=\-]\s*([1-5])",

        "groundedness":
            r"groundedness\s*[:=\-]\s*([1-5])",

        "helpfulness":
            r"helpfulness\s*[:=\-]\s*([1-5])",

        "brand_consistency":
            r"(?:brand[_ ]?consistency|brand)\s*[:=\-]\s*([1-5])",

        "safety":
            r"safety\s*[:=\-]\s*([1-5])",

        "overall":
            r"overall\s*[:=\-]\s*([1-5])"
    }

    scores = {}

    for key, pattern in patterns.items():

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            scores[key] = int(
                match.group(1)
            )

    required = [
        "correctness",
        "groundedness",
        "helpfulness",
        "brand_consistency",
        "safety",
        "overall"
    ]

    if all(
        key in scores
        for key in required
    ):

        return scores

    return None


# ============================================================
# SIMPLE SIX-NUMBER EXTRACTION
# ============================================================

def extract_six_numbers(text):

    if not text:
        return None

    text = str(
        text
    ).strip()

    # Prefer content after </think>.
    if "</think>" in text:

        after_think = text.split(
            "</think>",
            1
        )[1].strip()

        if after_think:

            text = after_think

    # --------------------------------------------------------
    # Look for explicit six-number sequence
    # --------------------------------------------------------

    patterns = [

        r"\b([1-5])\s+([1-5])\s+([1-5])\s+([1-5])\s+([1-5])\s+([1-5])\b",

        r"\b([1-5])\s*,\s*([1-5])\s*,\s*([1-5])\s*,\s*([1-5])\s*,\s*([1-5])\s*,\s*([1-5])\b",

        r"\b([1-5])\|([1-5])\|([1-5])\|([1-5])\|([1-5])\|([1-5])\b"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text
        )

        if match:

            values = [
                int(x)
                for x in match.groups()
            ]

            if len(values) == 6:

                return {
                    "correctness":
                        values[0],

                    "groundedness":
                        values[1],

                    "helpfulness":
                        values[2],

                    "brand_consistency":
                        values[3],

                    "safety":
                        values[4],

                    "overall":
                        values[5]
                }

    return None


# ============================================================
# COMPLETE SCORE PARSER
# ============================================================

def parse_judge_response(text):

    if not text:
        return None

    # --------------------------------------------------------
    # Method 1: JSON
    # --------------------------------------------------------

    result = try_json(
        text
    )

    if result:

        required = [
            "correctness",
            "groundedness",
            "helpfulness",
            "brand_consistency",
            "safety",
            "overall"
        ]

        if all(
            key in result
            for key in required
        ):

            cleaned = {}

            for key in required:

                cleaned[key] = clean_score(
                    result[key]
                )

            if all(
                cleaned[key] > 0
                for key in required
            ):

                return cleaned

    # --------------------------------------------------------
    # Method 2: labelled scores
    # --------------------------------------------------------

    result = extract_labeled_scores(
        text
    )

    if result:

        return result

    # --------------------------------------------------------
    # Method 3: six-number sequence
    # --------------------------------------------------------

    result = extract_six_numbers(
        text
    )

    if result:

        return result

    return None


# ============================================================
# LLM JUDGE
# ============================================================

def judge_reply(
    customer_message,
    intent,
    reply,
    evidence
):

    # --------------------------------------------------------
    # Keep prompt small.
    # --------------------------------------------------------

    customer_message = str(
        customer_message
    )[:500]

    intent = str(
        intent
    )[:100]

    reply = str(
        reply
    )[:600]

    evidence = str(
        evidence
    )[:700]

    # --------------------------------------------------------
    # Compact prompt
    # --------------------------------------------------------

    prompt = f"""
Evaluate this AppleSupport AI reply.

Customer:
{customer_message}

Intent:
{intent}

Reply:
{reply}

Historical evidence:
{evidence}

Rate 1-5:

correctness: does the reply address the issue?
groundedness: is it supported by evidence?
helpfulness: does it help the customer?
brand_consistency: professional AppleSupport tone?
safety: avoids unsafe or invented advice?
overall: overall quality?

Return ONLY this JSON:

{{
"correctness": 4,
"groundedness": 4,
"helpfulness": 4,
"brand_consistency": 4,
"safety": 5,
"overall": 4
}}
"""

    raw = call_ollama(
        prompt
    )

    scores = parse_judge_response(
        raw
    )

    if scores is None:

        print()
        print(
            "    Could not parse judge output."
        )

        if raw:

            print(
                "    Raw output:"
            )

            # Print more of the output so
            # we can diagnose it if needed.
            print(
                "    "
                + raw[:1000]
                .replace(
                    "\n",
                    " "
                )
            )

        return {
            "correctness": 0,
            "groundedness": 0,
            "helpfulness": 0,
            "brand_consistency": 0,
            "safety": 0,
            "overall": 0,
            "reason":
                "LLM judge output could not be parsed."
        }

    return {
        "correctness":
            scores["correctness"],

        "groundedness":
            scores["groundedness"],

        "helpfulness":
            scores["helpfulness"],

        "brand_consistency":
            scores["brand_consistency"],

        "safety":
            scores["safety"],

        "overall":
            scores["overall"],

        "reason":
            "Scored by local Qwen3 LLM judge."
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)

    print(
        "APPLE SUPPORT — "
        "LLM REPLY QUALITY EVALUATION"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        print()
        print(
            "ERROR: Input file not found:"
        )

        print(
            INPUT_FILE
        )

        print()
        print(
            "Run this first:"
        )

        print(
            "python -m "
            "src.evaluation.generate_reply_eval"
        )

        return

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    print()
    print(
        "Loading reply evaluation results..."
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Total examples available: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # Verify columns
    # --------------------------------------------------------

    required_columns = [
        "tweet_id",
        "text",
        "true_intent",
        "predicted_intent",
        "reply",
        "evidence"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        print()
        print(
            "ERROR: Missing columns:"
        )

        for column in missing:

            print(
                f"  - {column}"
            )

        print()
        print(
            "Available columns:"
        )

        print(
            df.columns.tolist()
        )

        return

    # --------------------------------------------------------
    # Limit examples
    # --------------------------------------------------------

    df = df.head(
        MAX_EXAMPLES
    ).copy()

    print()
    print(
        f"Examples to evaluate: "
        f"{len(df)}"
    )

    print(
        f"Model: {OLLAMA_MODEL}"
    )

    print(
        f"Timeout: "
        f"{TIMEOUT_SECONDS} seconds"
    )

    print()

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------------

    for position, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        print(
            "-" * 60
        )

        print(
            f"Evaluating example "
            f"{position}/{len(df)}"
        )

        # ----------------------------------------------------
        # Get fields
        # ----------------------------------------------------

        tweet_id = row.get(
            "tweet_id",
            ""
        )

        customer_message = str(
            row.get(
                "text",
                ""
            )
        )

        true_intent = str(
            row.get(
                "true_intent",
                ""
            )
        )

        predicted_intent = str(
            row.get(
                "predicted_intent",
                ""
            )
        )

        reply = str(
            row.get(
                "reply",
                ""
            )
        )

        evidence = str(
            row.get(
                "evidence",
                ""
            )
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print(
            "Customer:",
            customer_message[:120]
        )

        print(
            "True intent:",
            true_intent
        )

        print(
            "Predicted:",
            predicted_intent
        )

        print(
            "Reply:",
            reply[:150]
        )

        # ----------------------------------------------------
        # Judge
        # ----------------------------------------------------

        judge = judge_reply(
            customer_message,
            true_intent,
            reply,
            evidence
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        results.append({

            "example":
                position,

            "tweet_id":
                tweet_id,

            "customer_message":
                customer_message,

            "true_intent":
                true_intent,

            "predicted_intent":
                predicted_intent,

            "reply":
                reply,

            "evidence":
                evidence,

            "correctness":
                judge[
                    "correctness"
                ],

            "groundedness":
                judge[
                    "groundedness"
                ],

            "helpfulness":
                judge[
                    "helpfulness"
                ],

            "brand_consistency":
                judge[
                    "brand_consistency"
                ],

            "safety":
                judge[
                    "safety"
                ],

            "overall":
                judge[
                    "overall"
                ],

            "reason":
                judge[
                    "reason"
                ]
        })

        # ----------------------------------------------------
        # Print scores
        # ----------------------------------------------------

        print()

        print(
            "Scores:"
        )

        print(
            f"  Correctness      : "
            f"{judge['correctness']}/5"
        )

        print(
            f"  Groundedness     : "
            f"{judge['groundedness']}/5"
        )

        print(
            f"  Helpfulness      : "
            f"{judge['helpfulness']}/5"
        )

        print(
            f"  Brand consistency: "
            f"{judge['brand_consistency']}/5"
        )

        print(
            f"  Safety           : "
            f"{judge['safety']}/5"
        )

        print(
            f"  Overall          : "
            f"{judge['overall']}/5"
        )

        print(
            "  Reason:",
            judge["reason"]
        )

        time.sleep(
            0.3
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)

    print(
        "REPLY QUALITY EVALUATION COMPLETE"
    )

    print("=" * 60)

    total = len(
        results_df
    )

    valid = results_df[
        results_df["overall"] > 0
    ]

    failed = results_df[
        results_df["overall"] == 0
    ]

    print()

    print(
        f"Total evaluated: "
        f"{total}"
    )

    print(
        f"Successful judge calls: "
        f"{len(valid)}/{total}"
    )

    print(
        f"Failed judge calls: "
        f"{len(failed)}/{total}"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    if len(valid) > 0:

        metrics = [
            "correctness",
            "groundedness",
            "helpfulness",
            "brand_consistency",
            "safety",
            "overall"
        ]

        print()
        print(
            "Average scores:"
        )

        for metric in metrics:

            average = valid[
                metric
            ].mean()

            print(
                f"{metric:20s}: "
                f"{average:.2f} / 5"
            )

        # ----------------------------------------------------
        # Good replies
        # ----------------------------------------------------

        good = (
            valid["overall"] >= 4
        ).sum()

        percentage = (
            good /
            len(valid)
        ) * 100

        print()

        print(
            f"Replies scoring >= 4/5: "
            f"{good}/{len(valid)} "
            f"({percentage:.1f}%)"
        )

        # ----------------------------------------------------
        # Distribution
        # ----------------------------------------------------

        print()

        print(
            "Overall score distribution:"
        )

        print(
            valid[
                "overall"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

    else:

        print()
        print(
            "WARNING: No valid LLM "
            "judge scores were produced."
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()

    print(
        "Saved results:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()