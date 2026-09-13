# AI Customer Support Agent — Apple Support

## 1. Project Overview

This project builds an AI customer-support agent for **Apple Support** using historical customer-support conversations from Twitter.

For a new customer message, the agent performs three tasks:

1. Classifies the customer's issue into an Apple-specific intent.
2. Retrieves historically similar Apple Support cases.
3. Drafts a grounded reply and decides whether the case should be:
   - `AUTO-HANDLE`
   - `ESCALATE`

The main design goal is to make the generated reply grounded in how the brand historically handled similar customer problems rather than generating a completely free-form answer.

---

# 2. Dataset

The project uses the Kaggle dataset:

**Customer Support on Twitter**

Dataset:

`thoughtvector/customer-support-on-twitter`

The original dataset contains approximately 2.8 million tweets and support interactions.

Important columns include:

- `tweet_id`
- `author_id`
- `inbound`
- `created_at`
- `text`
- `response_tweet_id`
- `in_response_to_tweet_id`

Only the **AppleSupport** brand was used for this project.

From the full dataset:

- AppleSupport outbound replies: **106,860**
- Linked customer messages: **106,623**

The complete dataset is not repeatedly processed during development. Subsamples are used for training, evaluation and experimentation.

---

# 3. Why AppleSupport?

I selected AppleSupport because it has a large number of customer-support interactions and contains a wide range of realistic technical-support problems.

Examples include:

- iOS update problems
- battery problems
- keyboard/input problems
- app and media problems
- account and purchase problems
- hardware/device problems
- general support requests

This provides enough historical evidence to build a brand-specific support agent.

---

# 4. Intent Taxonomy

A small AppleSupport-specific intent taxonomy was created from the data.

The final seven intents are:

| Intent | Description |
|---|---|
| `ios_update_issue` | iOS, macOS or software-update related problems |
| `battery_issue` | Battery drain, charging or battery-related problems |
| `keyboard_input_issue` | Keyboard, typing, autocorrect or input problems |
| `app_media_issue` | Apps, photos, messages, music, media or application problems |
| `account_purchase_issue` | Account, purchase, order or ownership problems |
| `hardware_device_issue` | Physical device, screen, power or hardware problems |
| `general_support` | General support, service or unclear requests |

The taxonomy is intentionally small so that the system can be evaluated reliably.

---

# 5. System Architecture

```text
Customer Message
       |
       v
+----------------------+
| Intent Classification|
+----------------------+
       |
       v
+----------------------+
| Historical Retrieval |
| AppleSupport Cases   |
+----------------------+
       |
       v
+----------------------+
| Grounded Reply       |
| Generation           |
+----------------------+
       |
       v
+----------------------+
| Escalation Decision  |
+----------------------+
       |
       +------------------+
       |                  |
       v                  v
 AUTO-HANDLE           ESCALATE



Deploy link - https://apple-support-ai-agent-1234.streamlit.app/
