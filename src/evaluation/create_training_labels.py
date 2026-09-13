import pandas as pd


INPUT = "data/processed/apple_support_training_5000.csv"
OUTPUT = "data/processed/apple_support_training_labeled.csv"


def assign_intent(text):
    text = str(text).lower()

    # 1. Keyboard / typing
    if any(x in text for x in [
        "keyboard",
        "typing",
        "autocorrect",
        "keypad",
        "predictive",
        "question mark",
        "question marks",
        "letter i",
        "replace my i"
    ]):
        return "keyboard_input_issue"

    # 2. Battery / charging
    if any(x in text for x in [
        "battery",
        "battery life",
        "battery drain",
        "draining",
        "charging",
        "charger",
        "overheating",
        "overheat"
    ]):
        return "battery_issue"

    # 3. Account / purchase / payment
    if any(x in text for x in [
        "apple id",
        "password",
        "login",
        "log in",
        "sign in",
        "account",
        "payment",
        "charged",
        "refund",
        "purchase",
        "purchased",
        "billing",
        "order",
        "activation",
        "activate",
        "subscription",
        "applecare"
    ]):
        return "account_purchase_issue"

    # 4. Hardware / physical device
    if any(x in text for x in [
        "screen",
        "display",
        "broken",
        "cracked",
        "hardware",
        "button",
        "won't turn on",
        "will not turn on",
        "dead phone",
        "defective",
        "damaged"
    ]):
        return "hardware_device_issue"

    # 5. iOS / software update
    if any(x in text for x in [
        "ios",
        "update",
        "updating",
        "software update",
        "software",
        "upgraded",
        "upgrade",
        "high sierra",
        "system update",
        "latest version",
        "version"
    ]):
        return "ios_update_issue"

    # 6. Apps / Apple services / media
    if any(x in text for x in [
        "app",
        "apps",
        "application",
        "apple music",
        "music",
        "itunes",
        "photo",
        "photos",
        "video",
        "picture",
        "icloud",
        "imessage",
        "messages",
        "message",
        "maps",
        "apple maps",
        "facetime",
        "safari",
        "app store",
        "pages",
        "backup"
    ]):
        return "app_media_issue"

    # 7. Everything unclear / miscellaneous
    return "general_support"


print("Loading training data...")

df = pd.read_csv(INPUT)

print("Messages:", len(df))

df["intent"] = df["text"].apply(assign_intent)

df.to_csv(OUTPUT, index=False)

print("\nSaved:")
print(OUTPUT)

print("\nTraining intent distribution:")
print(df["intent"].value_counts().to_string())