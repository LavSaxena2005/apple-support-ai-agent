import pandas as pd


INPUT = "evaluation/golden_set_200.csv"
OUTPUT = "evaluation/golden_set_200_v2.csv"


def normalize_intent(text):
    text = str(text).lower()

    # 1. Keyboard / typing
    if any(x in text for x in [
        "keyboard",
        "autocorrect",
        "auto correct",
        "typing",
        "type",
        "letter",
        "predictive text",
        "capitalizing",
        "question mark",
        '"i"',
        " i "
    ]):
        return "keyboard_input_issue"

    # 2. Battery / charging
    if any(x in text for x in [
        "battery",
        "battery life",
        "battery drain",
        "draining",
        "charge",
        "charging",
        "charger",
        "usb port"
    ]):
        return "battery_issue"

    # 3. Account / purchase / order
    if any(x in text for x in [
        "apple id",
        "account",
        "password",
        "login",
        "log in",
        "purchase",
        "refund",
        "paypal",
        "order",
        "payment",
        "billing",
        "iphone x order",
        "upgrade program"
    ]):
        return "account_purchase_issue"

    # 4. Hardware / device failure
    if any(x in text for x in [
        "screen",
        "cracked",
        "won't turn on",
        "wont turn on",
        "broken",
        "hardware",
        "defective",
        "overheats",
        "phone is dead",
        "unresponsive"
    ]):
        return "hardware_device_issue"

    # 5. Apps / photos / music / camera / media
    if any(x in text for x in [
        "app",
        "apps",
        "application",
        "camera",
        "photo",
        "photos",
        "picture",
        "pictures",
        "music",
        "itunes",
        "podcast",
        "video",
        "imessage",
        "airdrop",
        "maps",
        "icloud"
    ]):
        return "app_media_issue"

    # 6. iOS / software update
    if any(x in text for x in [
        "ios",
        "update",
        "updated",
        "updating",
        "software",
        "high sierra",
        "latest version"
    ]):
        return "ios_update_issue"

    # 7. General
    return "general_support"


print("Loading golden set...")

df = pd.read_csv(INPUT)

print("Original examples:", len(df))

df["intent_v2"] = df["text"].apply(normalize_intent)

df.to_csv(OUTPUT, index=False)

print("\nSaved:")
print(OUTPUT)

print("\nNew intent distribution:")
print(df["intent_v2"].value_counts())