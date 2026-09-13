import pandas as pd
from pathlib import Path

INPUT_FILE = "data/raw/twcs.csv"
OUTPUT_FILE = "data/processed/apple_support.csv"
BRAND = "AppleSupport"

REQUIRED_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

def main():
    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. Download twcs.csv and place it in data/raw/."
        )

    print("Loading dataset...")
    df = pd.read_csv(INPUT_FILE, usecols=REQUIRED_COLUMNS)
    print(f"Total tweets: {len(df):,}")

    # Brand replies are outbound messages authored by the selected brand.
    brand_replies = df[
        (df["author_id"] == BRAND) & (df["inbound"] == False)
    ].copy()

    print(f"{BRAND} replies: {len(brand_replies):,}")

    customer_ids = (
        pd.to_numeric(
            brand_replies["in_response_to_tweet_id"], errors="coerce"
        )
        .dropna()
        .astype("int64")
        .unique()
    )

    customer_messages = df[
        (df["inbound"] == True) & (df["tweet_id"].isin(customer_ids))
    ].copy()

    print(f"Linked customer messages: {len(customer_messages):,}")

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    customer_messages.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
