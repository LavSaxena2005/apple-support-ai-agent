import pandas as pd
from pathlib import Path


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = Path("data/processed/apple_support.csv")
RAW_FILE = Path("data/raw/twcs.csv")
OUTPUT_FILE = Path("data/processed/apple_support_cases.csv")


# ============================================================
# STEP 1: LOAD APPLESUPPORT CUSTOMER MESSAGES
# ============================================================

print("Loading AppleSupport data...")

df = pd.read_csv(INPUT_FILE)

print("Customer messages:", len(df))


# ============================================================
# STEP 2: SELECT REQUIRED CUSTOMER COLUMNS
# ============================================================

cases = df[
    [
        "tweet_id",
        "created_at",
        "text",
        "response_tweet_id",
    ]
].copy()


# Rename columns so they are easier to understand
cases = cases.rename(
    columns={
        "tweet_id": "customer_tweet_id",
        "text": "customer_text",
        "response_tweet_id": "brand_reply_id",
    }
)


# ============================================================
# STEP 3: LOAD ORIGINAL TWITTER DATASET
# ============================================================

print("Loading original dataset for AppleSupport replies...")

raw = pd.read_csv(RAW_FILE)


# ============================================================
# STEP 4: GET APPLESUPPORT REPLIES
# ============================================================

apple_replies = raw[
    (raw["author_id"] == "AppleSupport") &
    (raw["inbound"] == False)
][
    [
        "tweet_id",
        "text",
    ]
].copy()


# Rename columns
apple_replies = apple_replies.rename(
    columns={
        "tweet_id": "brand_reply_id",
        "text": "brand_reply",
    }
)


print("AppleSupport replies:", len(apple_replies))


# ============================================================
# STEP 5: FIX DATA TYPES
# ============================================================
# The error happened because one ID column was object/string
# and the other was int64.
#
# We convert both columns to string before merging.

cases["brand_reply_id"] = cases["brand_reply_id"].astype(str)

apple_replies["brand_reply_id"] = (
    apple_replies["brand_reply_id"].astype(str)
)


# ============================================================
# STEP 6: JOIN CUSTOMER MESSAGE WITH BRAND REPLY
# ============================================================

print("Matching customer messages with AppleSupport replies...")

cases = cases.merge(
    apple_replies,
    on="brand_reply_id",
    how="inner",
)


# ============================================================
# STEP 7: REMOVE INVALID DATA
# ============================================================

cases = cases.dropna(
    subset=[
        "customer_text",
        "brand_reply",
    ]
)


# Remove duplicate customer messages
cases = cases.drop_duplicates(
    subset=["customer_tweet_id"]
)


# ============================================================
# STEP 8: CLEAN TEXT
# ============================================================

cases["customer_text"] = (
    cases["customer_text"]
    .astype(str)
    .str.strip()
)

cases["brand_reply"] = (
    cases["brand_reply"]
    .astype(str)
    .str.strip()
)


# Remove empty messages
cases = cases[
    (cases["customer_text"] != "") &
    (cases["brand_reply"] != "")
]


# ============================================================
# STEP 9: SAVE HISTORICAL CASES
# ============================================================

cases.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# STEP 10: SHOW RESULTS
# ============================================================

print()
print("==============================================")
print("Historical AppleSupport cases created!")
print("==============================================")

print("Total cases:", len(cases))

print("Saved to:", OUTPUT_FILE)

print()
print("Columns:")
print(cases.columns.tolist())

print()
print("Sample historical cases:")
print("----------------------------------------------")

print(
    cases[
        [
            "customer_text",
            "brand_reply",
        ]
    ]
    .head(5)
    .to_string(index=False)
)

print()
print("==============================================")
print("DONE!")
print("==============================================")