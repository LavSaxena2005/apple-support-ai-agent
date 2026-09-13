import streamlit as st
import pandas as pd
from pathlib import Path

RESULT_FILE = Path("evaluation/results/leakage_safe_eval_50.csv")
OUTPUT_FILE = Path("evaluation/human_reply_ratings.csv")

st.set_page_config(
    page_title="Human Reply Quality Evaluation",
    layout="wide"
)

st.title("AppleSupport AI Agent - Human Reply Quality Evaluation")

if not RESULT_FILE.exists():
    st.error(f"File not found: {RESULT_FILE}")
    st.stop()

df = pd.read_csv(RESULT_FILE)

if OUTPUT_FILE.exists():
    ratings = pd.read_csv(OUTPUT_FILE)
else:
    ratings = pd.DataFrame(columns=[
        "tweet_id",
        "text",
        "predicted_intent",
        "decision",
        "reply",
        "human_overall"
    ])

# Keep one row per example
rated_ids = set(ratings["tweet_id"].astype(str)) if len(ratings) else set()

remaining = df[~df["tweet_id"].astype(str).isin(rated_ids)].copy()

st.write(f"Total examples: {len(df)}")
st.write(f"Already rated: {len(ratings)}")
st.write(f"Remaining: {len(remaining)}")

if len(remaining) == 0:
    st.success("All examples have been rated!")
    st.stop()

row = remaining.iloc[0]

st.divider()

st.subheader(f"Example {len(ratings) + 1} / {len(df)}")

st.markdown("### Customer message")
st.info(str(row["text"]))

st.markdown("### Predicted intent")
st.write(row["predicted_intent"])

st.markdown("### Agent decision")
st.write(row["decision"])

st.markdown("### AI reply")
st.success(str(row["reply"]))

st.markdown("### Human rating")

rating = st.radio(
    "How good is this reply overall?",
    options=[1, 2, 3, 4, 5],
    horizontal=True,
    format_func=lambda x: {
        1: "1 - Very poor",
        2: "2 - Poor",
        3: "3 - Average",
        4: "4 - Good",
        5: "5 - Excellent"
    }[x]
)

if st.button("Save Rating & Next", type="primary"):

    new_row = {
        "tweet_id": row["tweet_id"],
        "text": row["text"],
        "predicted_intent": row["predicted_intent"],
        "decision": row["decision"],
        "reply": row["reply"],
        "human_overall": rating
    }

    ratings = pd.concat(
        [ratings, pd.DataFrame([new_row])],
        ignore_index=True
    )

    ratings.to_csv(OUTPUT_FILE, index=False)

    st.success("Rating saved!")

    st.rerun()