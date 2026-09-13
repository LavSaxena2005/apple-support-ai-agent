import streamlit as st
import pandas as pd
from pathlib import Path

FILE = Path("evaluation/golden_set_final.csv")

INTENTS = [
    "ios_update_issue",
    "battery_issue",
    "keyboard_input_issue",
    "app_media_issue",
    "account_purchase_issue",
    "hardware_device_issue",
    "general_support",
]

DESCRIPTIONS = {
    "ios_update_issue":
        "iOS/software update or software-version problem.",

    "battery_issue":
        "Battery drain, battery health, charging or charger problem.",

    "keyboard_input_issue":
        "Keyboard, typing, autocorrect, predictive text or input problem.",

    "app_media_issue":
        "Apps or Apple services such as Maps, Photos, Music, iMessage, etc.",

    "account_purchase_issue":
        "Apple ID, account, login, payment, purchase, refund or billing problem.",

    "hardware_device_issue":
        "Physical device, screen, button, hardware failure or device damage.",

    "general_support":
        "Unclear or miscellaneous support request."
}

st.set_page_config(
    page_title="AppleSupport Golden Set",
    layout="centered"
)

st.title("🍎 AppleSupport Golden Set Review")

df = pd.read_csv(FILE)

# Ensure required columns exist
if "reviewed" not in df.columns:
    df["reviewed"] = False

df["reviewed"] = df["reviewed"].fillna(False).astype(bool)

df["intent"] = df["intent"].fillna("").astype(str)
df["escalation_reason"] = df["escalation_reason"].fillna("").astype(str)

# Progress
total = len(df)
reviewed_count = int(df["reviewed"].sum())
remaining = total - reviewed_count

st.progress(reviewed_count / total)

st.write(
    f"### Progress: {reviewed_count}/{total} reviewed"
)

if remaining == 0:
    st.success("🎉 All 200 examples have been human-reviewed!")
    st.stop()

# First unreviewed example
idx = df.index[~df["reviewed"]][0]
row = df.loc[idx]

st.write(f"### Example {idx + 1} of {total}")

st.info(row["text"])

st.divider()

# Current suggested label
current_intent = row["intent"]

if current_intent not in INTENTS:
    current_intent = "general_support"

default_index = INTENTS.index(current_intent)

intent = st.selectbox(
    "Select the correct intent:",
    INTENTS,
    index=default_index
)

st.caption(DESCRIPTIONS[intent])

st.divider()

# Escalation
current_escalate = bool(row["escalate"])

escalate = st.radio(
    "Should this case be escalated to a human?",
    ["No", "Yes"],
    index=1 if current_escalate else 0,
    horizontal=True
)

if escalate == "Yes":
    reason = st.text_input(
        "Escalation reason:",
        value=row["escalation_reason"],
        placeholder="Example: Hardware failure may require human support."
    )
else:
    reason = ""

st.divider()

if st.button("✅ Confirm & Next", type="primary"):

    df.at[idx, "intent"] = intent
    df.at[idx, "escalate"] = escalate == "Yes"
    df.at[idx, "escalation_reason"] = reason
    df.at[idx, "reviewed"] = True

    df.to_csv(FILE, index=False)

    st.success("Saved!")

    st.rerun()