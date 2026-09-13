from dataclasses import dataclass
from typing import List
import re
import json

import pandas as pd
import faiss
import requests
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CASES_FILE = "data/processed/apple_support_cases_demo.csv"

INDEX_FILE = "data/processed/apple_support_demo.faiss"

OLLAMA_URL = "http://localhost:11434/api/generate"

OLLAMA_MODEL = "qwen3:4b"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# RESULT STRUCTURE
# ============================================================

@dataclass
class AgentResult:
    intent: str
    confidence: float
    reply: str
    decision: str
    reason: str
    evidence: List[str]


# ============================================================
# LOAD HISTORICAL CASES
# ============================================================

print("Loading historical AppleSupport cases...")

cases_df = pd.read_csv(CASES_FILE)

print(
    f"Historical cases loaded: {len(cases_df)}"
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model ready.")


# ============================================================
# LOAD SAVED FAISS INDEX
# ============================================================

print("Loading saved FAISS index...")

faiss_index = faiss.read_index(
    INDEX_FILE
)

print(
    f"FAISS index loaded: {faiss_index.ntotal} cases"
)


# ============================================================
# FINAL APPLESUPPORT INTENT TAXONOMY
# ============================================================

INTENT_KEYWORDS = {

    # --------------------------------------------------------
    # 1. iOS / Software Update
    # --------------------------------------------------------

    "ios_update_issue": [
        "ios",
        "ios update",
        "software update",
        "system update",
        "updating",
        "upgrade",
        "upgraded",
        "high sierra",
        "software version",
        "latest version",
        "downgrade",
        "update",
        "updated",
        "after updating",
        "after the update",
        "after an update",
        "since updating",
        "since the update"
    ],

    # --------------------------------------------------------
    # 2. Battery
    # --------------------------------------------------------

    "battery_issue": [
        "battery",
        "battery drain",
        "battery draining",
        "battery life",
        "battery dies",
        "battery dying",
        "not holding a charge",
        "won't hold a charge",
        "charging problem",
        "charging issue",
        "charger",
        "overheating",
        "overheat",
        "phone is hot",
        "iphone is hot"
    ],

    # --------------------------------------------------------
    # 3. Keyboard / Input
    # --------------------------------------------------------

    "keyboard_input_issue": [
        "keyboard",
        "typing",
        "autocorrect",
        "auto correct",
        "keypad",
        "predictive text",
        "predictive",
        "question mark",
        "question marks",
        "letter i",
        "replace my i",
        "replacing my",
        "keys not working",
        "keyboard not working",
        "can't type",
        "cannot type",
        "unable to type"
    ],

    # --------------------------------------------------------
    # 4. Apps / Media
    # --------------------------------------------------------

    "app_media_issue": [
        "app store",
        "apple music",
        "itunes",
        "photos",
        "photo",
        "pictures",
        "picture",
        "videos",
        "video",
        "camera",
        "icloud",
        "imessage",
        "messages",
        "message",
        "apple maps",
        "maps",
        "facetime",
        "safari",
        "pages",
        "backup",
        "back up",
        "backup failed",
        "photos missing",
        "photos disappeared",
        "music disappeared",
        "music missing",
        "messages disappeared",
        "messages missing",
        "videos missing",
        "app crashing",
        "apps crashing",
        "application crashing",
        "application not working",
        "apps not working",
        "app not working"
    ],

    # --------------------------------------------------------
    # 5. Account / Purchase
    # --------------------------------------------------------

    "account_purchase_issue": [
        "apple id",
        "password",
        "login",
        "log in",
        "sign in",
        "account",
        "payment",
        "paid",
        "charged",
        "charged twice",
        "refund",
        "purchase",
        "purchased",
        "billing",
        "order",
        "activation",
        "activate",
        "subscription",
        "applecare",
        "buy again",
        "buy it again",
        "missing purchase",
        "purchase missing",
        "payment failed",
        "billing problem"
    ],

    # --------------------------------------------------------
    # 6. Hardware / Device
    # --------------------------------------------------------

    "hardware_device_issue": [
        "screen",
        "display",
        "broken",
        "cracked",
        "physical damage",
        "hardware",
        "button",
        "speaker",
        "won't turn on",
        "will not turn on",
        "can't turn on",
        "cannot turn on",
        "dead phone",
        "defective",
        "damaged",
        "phone is dead",
        "iphone is dead",
        "device is dead",
        "won't power on",
        "power button",
        "screen broken",
        "screen cracked"
    ],

    # --------------------------------------------------------
    # 7. General Support
    # --------------------------------------------------------

    "general_support": []
}


# ============================================================
# INTENT CLASSIFIER
# ============================================================

def classify_intent(message: str):

    """
    Rule-based AppleSupport intent classifier.

    The previous classifier used direct substring matching.
    That caused generic words such as "app" to produce many
    false positives.

    This version:
        1. Uses whole-word matching where appropriate.
        2. Uses specific phrases.
        3. Gives stronger weights to highly specific signals.
        4. Gives priority to battery, keyboard and hardware
           when those issues are explicit.
    """

    text = str(message).lower().strip()

    # --------------------------------------------------------
    # Helper functions
    # --------------------------------------------------------

    def has_word(word: str) -> bool:
        """
        Match a complete word instead of an arbitrary substring.
        Example:
            "app" matches "app"
            "app" does NOT match "happened"
        """

        pattern = r"\b" + re.escape(
            word.lower()
        ) + r"\b"

        return re.search(
            pattern,
            text
        ) is not None

    def has_phrase(phrase: str) -> bool:
        """
        Match a meaningful phrase.
        """

        return phrase.lower() in text

    # --------------------------------------------------------
    # Initialize scores
    # --------------------------------------------------------

    scores = {
        "ios_update_issue": 0,
        "battery_issue": 0,
        "keyboard_input_issue": 0,
        "app_media_issue": 0,
        "account_purchase_issue": 0,
        "hardware_device_issue": 0,
        "general_support": 0
    }

    # ========================================================
    # 1. BATTERY
    # ========================================================

    battery_phrases = [
        "battery",
        "battery drain",
        "battery draining",
        "battery life",
        "battery dies",
        "battery dying",
        "not holding a charge",
        "won't hold a charge",
        "charging problem",
        "charging issue",
        "charger",
        "overheating",
        "overheat",
        "phone is hot",
        "iphone is hot"
    ]

    for term in battery_phrases:

        if has_phrase(term):

            scores["battery_issue"] += 2

    # Strong explicit battery signal

    if has_word("battery"):

        scores["battery_issue"] += 4

    # ========================================================
    # 2. KEYBOARD / INPUT
    # ========================================================

    keyboard_phrases = [
        "keyboard",
        "typing",
        "autocorrect",
        "auto correct",
        "keypad",
        "predictive text",
        "predictive",
        "question mark",
        "question marks",
        "letter i",
        "replace my i",
        "replacing my",
        "keys not working",
        "keyboard not working",
        "can't type",
        "cannot type",
        "unable to type"
    ]

    for term in keyboard_phrases:

        if has_phrase(term):

            scores["keyboard_input_issue"] += 3

    # ========================================================
    # 3. ACCOUNT / PURCHASE
    # ========================================================

    account_phrases = [
        "apple id",
        "password",
        "login",
        "log in",
        "sign in",
        "account",
        "payment",
        "paid",
        "charged",
        "charged twice",
        "refund",
        "purchase",
        "purchased",
        "billing",
        "order",
        "activation",
        "activate",
        "subscription",
        "applecare",
        "buy again",
        "buy it again",
        "missing purchase",
        "purchase missing",
        "payment failed",
        "billing problem"
    ]

    for term in account_phrases:

        if has_phrase(term):

            scores["account_purchase_issue"] += 2

    # Strong account / purchase signals

    strong_account_terms = [
        "apple id",
        "refund",
        "charged twice",
        "purchase missing",
        "missing purchase",
        "payment failed",
        "billing problem",
        "buy again"
    ]

    for term in strong_account_terms:

        if has_phrase(term):

            scores["account_purchase_issue"] += 3

    # ========================================================
    # 4. HARDWARE / DEVICE
    # ========================================================

    hardware_phrases = [
        "screen",
        "display",
        "broken",
        "cracked",
        "physical damage",
        "hardware",
        "button",
        "speaker",
        "won't turn on",
        "will not turn on",
        "can't turn on",
        "cannot turn on",
        "dead phone",
        "defective",
        "damaged",
        "phone is dead",
        "iphone is dead",
        "device is dead",
        "won't power on",
        "power button",
        "screen broken",
        "screen cracked"
    ]

    for term in hardware_phrases:

        if has_phrase(term):

            scores["hardware_device_issue"] += 3

    # ========================================================
    # 5. iOS / SOFTWARE UPDATE
    # ========================================================

    update_phrases = [
        "ios",
        "ios update",
        "software update",
        "system update",
        "updating",
        "upgrade",
        "upgraded",
        "high sierra",
        "software version",
        "latest version",
        "downgrade",
        "update",
        "updated",
        "after updating",
        "after the update",
        "after an update",
        "since updating",
        "since the update"
    ]

    for term in update_phrases:

        if has_phrase(term):

            scores["ios_update_issue"] += 2

    # Explicit iOS gets additional weight

    if has_word("ios"):

        scores["ios_update_issue"] += 3

    # ========================================================
    # 6. APPS / MEDIA
    # ========================================================

    # IMPORTANT:
    #
    # Do NOT use:
    #
    #     if "app" in text
    #
    # because substring matching can produce false positives.
    #
    # We use specific application/media phrases instead.

    app_media_phrases = [
        "app store",
        "apple music",
        "itunes",
        "photos",
        "photo",
        "pictures",
        "picture",
        "videos",
        "video",
        "camera",
        "icloud",
        "imessage",
        "messages",
        "message",
        "apple maps",
        "maps",
        "facetime",
        "safari",
        "pages",
        "backup",
        "back up",
        "backup failed",
        "photos missing",
        "photos disappeared",
        "music disappeared",
        "music missing",
        "messages disappeared",
        "messages missing",
        "videos missing",
        "app crashing",
        "apps crashing",
        "application crashing",
        "application not working",
        "apps not working",
        "app not working"
    ]

    for term in app_media_phrases:

        if has_phrase(term):

            scores["app_media_issue"] += 2

    # Standalone app/apps only

    if has_word("app"):

        scores["app_media_issue"] += 1

    if has_word("apps"):

        scores["app_media_issue"] += 1

    # ========================================================
    # 7. GENERAL SUPPORT
    # ========================================================

    # General support intentionally remains the fallback.
    scores["general_support"] = 0

    # ========================================================
    # SPECIAL PRIORITY RULES
    # ========================================================

    # --------------------------------------------------------
    # Battery beats update
    #
    # Example:
    # "Battery draining after iOS update"
    # --------------------------------------------------------

    if (
        scores["battery_issue"] > 0
        and has_word("battery")
    ):

        scores["battery_issue"] += 3

    # --------------------------------------------------------
    # Keyboard beats generic app/media
    #
    # Example:
    # "Keyboard not working in an app"
    # --------------------------------------------------------

    if scores["keyboard_input_issue"] > 0:

        scores["keyboard_input_issue"] += 2

    # --------------------------------------------------------
    # Hardware beats generic app/media
    # --------------------------------------------------------

    if scores["hardware_device_issue"] > 0:

        scores["hardware_device_issue"] += 1

     # ========================================================
    # FIND BEST INTENT
    # ========================================================

    # Priority rules for specific customer problems.
    # A specific issue should win over a generic update/app signal.

    if scores["battery_issue"] > 0 and has_word("battery"):
        best_intent = "battery_issue"

    elif scores["keyboard_input_issue"] > 0:
        best_intent = "keyboard_input_issue"

    elif scores["hardware_device_issue"] > 0:
        best_intent = "hardware_device_issue"

    else:
        best_intent = max(
            scores,
            key=scores.get
        )

    best_score = scores[best_intent]


    # ========================================================
    # NOTHING MATCHED
    # ========================================================

    if best_score == 0:
        return (
            "general_support",
            0.40
        )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence = min(
        0.95,
        0.55 + (
            best_score * 0.08
        )
    )

    return (
        best_intent,
        confidence
    )


# ============================================================
# SEMANTIC RETRIEVAL
# ============================================================

def retrieve_cases(
    message: str,
    top_k: int = 5
):

    # --------------------------------------------------------
    # Convert customer message into embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [message],
        convert_to_numpy=True
    )

    # --------------------------------------------------------
    # Normalize embedding
    # --------------------------------------------------------

    faiss.normalize_L2(
        query_embedding
    )

    # --------------------------------------------------------
    # Search historical cases
    # --------------------------------------------------------

    scores, indices = faiss_index.search(
        query_embedding,
        top_k
    )

    results = []

    # --------------------------------------------------------
    # Convert FAISS results to dictionaries
    # --------------------------------------------------------

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:

            continue

        row = cases_df.iloc[
            int(idx)
        ]

        results.append({

            "score": float(score),

            "customer_text":
                str(
                    row["customer_text"]
                ),

            "brand_reply":
                str(
                    row["brand_reply"]
                )
        })

    return results


# ============================================================
# OLLAMA
# ============================================================

def call_ollama(prompt: str):
    """
    Send the prompt to the local Ollama model and return only
    the customer-facing reply.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "num_predict": 160
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=180
        )
        response.raise_for_status()
        data = response.json()
        raw = str(data.get("response", "")).strip()

        if not raw:
            print("WARNING: Ollama returned an empty response.")
            return None

        # Prefer structured JSON output.
        reply = None
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                reply = parsed.get("reply")
                # The model sometimes returns a structured copy of the
                # prompt instead of the requested reply. Never expose that.
                if not reply:
                    print("WARNING: Ollama returned structured reasoning instead of a reply.")
                    return None
        except Exception:
            # Some local model versions may still return surrounding text.
            pass

        if not reply:
            # Remove Qwen reasoning markers if present.
            cleaned = raw
            if "</think>" in cleaned:
                cleaned = cleaned.split("</think>", 1)[1].strip()
            if "<think>" in cleaned:
                cleaned = cleaned.split("<think>", 1)[-1].strip()

            # Try to extract the last JSON object containing "reply".
            matches = re.findall(r'\{.*?"reply"\s*:\s*".*?"\s*\}', cleaned, re.DOTALL)
            if matches:
                try:
                    parsed = json.loads(matches[-1])
                    reply = parsed.get("reply")
                except Exception:
                    pass

            if not reply:
                reply = cleaned

        reply = str(reply).strip()

        prefixes = [
            "CUSTOMER-FACING REPLY:",
            "Customer-facing reply:",
            "Reply:",
            "Assistant:",
            "Final answer:",
            "Final Reply:",
            "Response:"
        ]
        for prefix in prefixes:
            if reply.startswith(prefix):
                reply = reply[len(prefix):].strip()

        bad_starts = [
            "We are given",
            "We have a customer message",
            "The customer message",
            "We need to",
            "The user wants",
            "The task is",
            "We are asked to",
            "Historical Evidence:",
            "Classified intent:",
            "The classified intent is:",
            "Let's analyze",
            "Analysis:",
            "Reasoning:",
            "Looking at the historical",
            "Case 1:",
            "Case 2:"
        ]

        if reply.startswith("{") or reply.startswith("["):
            print("WARNING: Ollama returned structured data instead of a customer reply.")
            return None

        if any(reply.startswith(item) for item in bad_starts):
            print("WARNING: Ollama returned internal reasoning.")
            return None

        # Do not allow obvious multi-line reasoning to become a reply.
        if any(marker in reply for marker in [
            "Customer message:",
            "Classified intent:",
            "Historical AppleSupport responses:",
            "Important rules:"
        ]):
            print("WARNING: Ollama returned internal reasoning.")
            return None

        if len(reply) < 10:
            print("WARNING: Ollama reply is too short.")
            return None

        if len(reply) > 1000:
            print("WARNING: Ollama reply is too long.")
            return None

        return reply

    except Exception as error:
        print(f"Ollama error: {error}")
        return None


def generate_reply(
    message: str,
    intent: str,
    retrieved_cases
):
    """
    Generate a short customer-support reply using
    retrieved historical AppleSupport responses.
    """

    # --------------------------------------------------------
    # Fallback if retrieval finds nothing
    # --------------------------------------------------------

    if not retrieved_cases:
        return (
            "Thanks for contacting AppleSupport. "
            "We'd be happy to help. Please provide "
            "more details about the issue you're "
            "experiencing."
        )

    # --------------------------------------------------------
    # Prepare historical evidence
    # --------------------------------------------------------

    evidence_text = ""

    for i, case in enumerate(
        retrieved_cases[:5],
        start=1
    ):

        if isinstance(case, dict):

            customer_text = str(
                case.get(
                    "customer_text",
                    ""
                )
            ).strip()

            brand_reply = str(
                case.get(
                    "brand_reply",
                    ""
                )
            ).strip()

            score = case.get(
                "score",
                0.0
            )

        else:

            customer_text = ""

            brand_reply = str(
                case
            ).strip()

            score = 0.0

        if not brand_reply:
            continue

        evidence_text += f"""
HISTORICAL CASE {i}

Customer issue:
{customer_text}

AppleSupport response:
{brand_reply}

Similarity score:
{float(score):.4f}

--------------------------------
"""

    # --------------------------------------------------------
    # If evidence is empty
    # --------------------------------------------------------

    if not evidence_text:

        return (
            "Thanks for contacting AppleSupport. "
            "Could you provide more details about "
            "the issue you're experiencing?"
        )

    # --------------------------------------------------------
    # Grounded LLM prompt
    # --------------------------------------------------------

    prompt = f"""
You are an AI customer-support assistant for AppleSupport.

Write ONE short, helpful customer-facing reply.

CUSTOMER MESSAGE:
{message}

CLASSIFIED INTENT:
{intent}

HISTORICAL APPLESUPPORT RESPONSES:
{evidence_text}

IMPORTANT RULES:

1. Use the historical AppleSupport responses as the main source.

2. Adapt the historical response to the customer's actual problem.

3. Do NOT invent Apple policies.

4. Do NOT invent refund or compensation policies.

5. Do NOT invent troubleshooting steps that are not supported
   by the historical responses.

6. Do NOT invent URLs, phone numbers, prices, warranty
   information, or promises.

7. Do NOT claim Apple has already performed an action.

8. If the historical evidence does not provide enough information
   for a specific solution, ask the customer for the minimum useful
   information needed.

9. Be polite and professional.

10. Keep the response to 1-3 sentences.

11. Do not mention the dataset.

12. Do not mention historical cases.

13. Do not mention similarity scores.

14. Do not mention that you are an AI.

15. Do not explain your reasoning.

16. Return ONLY the customer-facing reply.

CUSTOMER-FACING REPLY:
""".strip()

    # --------------------------------------------------------
    # Call Ollama
    # --------------------------------------------------------

    reply = call_ollama(prompt)

    # --------------------------------------------------------
    # Clean response
    # --------------------------------------------------------

    if reply:
        reply = str(reply).strip()

        prefixes = [
            "CUSTOMER-FACING REPLY:",
            "Reply:",
            "Assistant:",
            "Final answer:",
            "Final Reply:",
            "Response:"
        ]

        for prefix in prefixes:
            if reply.startswith(prefix):
                reply = reply[len(prefix):].strip()

    # --------------------------------------------------------
    # Final fallback
    # --------------------------------------------------------

    if not reply:
        # Use the strongest historical AppleSupport response as a
        # grounded fallback instead of inventing troubleshooting.
        fallback_reply = ""
        if retrieved_cases:
            fallback_reply = str(
                retrieved_cases[0].get("brand_reply", "")
            ).strip()

        # Remove a leading customer handle when present.
        fallback_reply = re.sub(
            r"^@[A-Za-z0-9_]+\s*",
            "",
            fallback_reply
        ).strip()

        if fallback_reply:
            reply = fallback_reply
        else:
            reply = (
                "Thanks for contacting AppleSupport. "
                "We'd be happy to help. Please provide more details "
                "about the issue you're experiencing."
            )

    return reply



# ============================================================
# ESCALATION DECISION
# ============================================================

# ============================================================
# ESCALATION DECISION
# ============================================================

def decide_escalation(
    message: str,
    intent: str,
    confidence: float,
    retrieved_cases: list
):
    """
    Conservative escalation policy for AppleSupport.

    The agent escalates cases involving:
    1. Account / payment / security risk
    2. Serious hardware or device failure
    3. Data loss
    4. Repeated or unresolved problems
    5. Severe software-update failures
    6. Multiple simultaneous technical problems
    7. Strong customer dissatisfaction with a technical issue
    8. Low-confidence concrete problems
    9. Missing historical evidence
    10. Weak retrieval similarity

    Return format:
        ("ESCALATE", reason)
        ("AUTO-HANDLE", reason)
    """

    text = str(message).lower().strip()

    # ========================================================
    # 1. HIGH-RISK ACCOUNT / PAYMENT / SECURITY
    # ========================================================

    high_risk_terms = [
        "security breach",
        "hacked",
        "hack",
        "stole",
        "stolen",
        "fraud",
        "unauthorized",
        "account compromised",

        "charged",
        "charge",
        "charged me",
        "charged twice",
        "money taken",

        "payment",
        "payment issue",
        "payment problem",
        "payment failed",

        "purchase missing",
        "missing purchase",
        "purchase isn't showing",
        "purchase isnt showing",
        "buy again",
        "buy it again",

        "can't verify",
        "cannot verify",
        "unable to verify",
        "verification",

        "activation",
        "activate",
        "activation issue",
        "activation impossible",
        "can't activate",
        "cannot activate",
        "unable to activate",

        "upgrade program",
        "order",
        "refund",

        "account locked",
        "can't access my account",
        "cannot access my account",
        "unable to access my account"
    ]

    if any(term in text for term in high_risk_terms):
        return (
            "ESCALATE",
            "High-risk account, payment, purchase, activation, or security issue."
        )

    # ========================================================
    # 2. SERIOUS HARDWARE / DEVICE FAILURE
    # ========================================================

    serious_device_terms = [
        "won't turn on",
        "wont turn on",
        "will not turn on",
        "can't turn on",
        "cannot turn on",

        "won't power on",
        "wont power on",
        "will not power on",

        "phone is dead",
        "iphone is dead",
        "ipad is dead",
        "mac is dead",
        "device is dead",
        "completely dead",
        "dead",

        "unresponsive",
        "device hung",

        "cracked screen",
        "cracked iphone",
        "screen is cracked",
        "screen has cracked",
        "screen is broken",
        "screen has broken",
        "broken screen",
        "screen broke",
        "screen broken",
        "needs screen repair",
        "need screen repair",
        "screen repair",
        "screen replacement",
        "need repair",
        "needs repair",

        "headphones broke",
        "charger broke",
        "charger stopped",
        "cable broke",

        "physical damage",
        "defective",
        "damaged"
    ]

    if any(term in text for term in serious_device_terms):
        return (
            "ESCALATE",
            "Serious device or hardware failure requires human support."
        )

    # ========================================================
    # 3. DATA LOSS / MISSING USER CONTENT
    # ========================================================

    data_loss_terms = [
        "photos disappeared",
        "photos are gone",
        "photos gone",
        "photos missing",

        "pictures disappeared",
        "pictures are gone",
        "pictures gone",
        "pictures missing",

        "text messages are gone",
        "text messages disappeared",
        "messages are gone",
        "messages disappeared",
        "messages missing",

        "data disappeared",
        "data is gone",
        "lost my data",
        "lost all my",

        "backup incomplete",
        "backup is incomplete",
        "backup failed",
        "backup missing",
        "backup hasn't completed",
        "backup hasnt completed",

        "not transferred",
        "only 200 of",
        "only 200"
    ]

    if any(term in text for term in data_loss_terms):
        return (
            "ESCALATE",
            "Potential data loss or missing user content requires human support."
        )

    # ========================================================
    # 4. REPEATED / UNRESOLVED SUPPORT PROBLEMS
    # ========================================================

    repeated_terms = [
        "again",
        "third time",
        "twice",
        "happened twice",
        "happened again",

        "multiple times",
        "several times",
        "repeatedly",

        "still not",
        "still doesn't",
        "still doesnt",
        "still won't",
        "still wont",

        "still not fixed",
        "still not working",
        "still broken",

        "doesn't help",
        "doesnt help",

        "not responding",
        "no response",
        "no answer",
        "nobody answers",
        "no one answers",

        "haven't helped",
        "havent helped",
        "no help",

        "months",
        "weeks",
        "for hours",

        "called multiple times",
        "called several times",
        "called many times",

        "contacted multiple times",
        "contacted several times",

        "multiple calls",
        "four calls",

        "nothing works",
        "nothing has been done"
    ]

    if any(term in text for term in repeated_terms):
        return (
            "ESCALATE",
            "Repeated or unresolved support issue."
        )

    # ========================================================
    # 5. SEVERE iOS / SOFTWARE UPDATE FAILURE
    # ========================================================

    update_terms = [
        "ios 11",
        "ios 11.0",
        "ios 11.0.3",
        "ios 11.1",
        "ios 11.1.1",

        "latest update",
        "after updating",
        "after the update",
        "after an update",
        "since updating",
        "since the update",
        "updated and now",

        "update has",
        "update ruined",
        "update wrecked",
        "update broke",

        "downgrade"
    ]

    severe_failure_terms = [
        "crash",
        "crashes",
        "crashing",

        "freeze",
        "freezes",
        "freezing",
        "frozen",

        "restart",
        "restarts",
        "restarting",
        "restarting over",

        "won't work",
        "wont work",
        "can't use",
        "cant use",

        "unresponsive",
        "dead",

        "slow",
        "hangs",

        "heats up",
        "overheating",

        "battery life is awful",

        "apps don't work",
        "apps dont work",
        "apps freezing",
        "apps shutting down",

        "phone is ruined",
        "ruined my phone",
        "wrecked my phone",

        "dumpster fire",
        "mess",
        "joke",
        "crap",
        "junk"
    ]

    has_update = any(
        term in text
        for term in update_terms
    )

    has_severe_failure = any(
        term in text
        for term in severe_failure_terms
    )

    if has_update and has_severe_failure:
        return (
            "ESCALATE",
            "Update-related failure is causing significant device or software problems."
        )

    # ========================================================
    # 6. MULTIPLE TECHNICAL FAILURES
    # ========================================================

    technical_terms = [
        "freezing",
        "freezes",
        "crashing",
        "crashes",

        "restarting",
        "restarts",

        "slow",
        "heats up",

        "battery",

        "wifi",
        "wi-fi",
        "bluetooth",

        "camera",

        "apps",
        "app",

        "can't connect",
        "cant connect",
        "won't connect",
        "wont connect",

        "not working",
        "doesn't work",
        "doesnt work"
    ]

    technical_count = sum(
        1
        for term in technical_terms
        if term in text
    )

    if technical_count >= 3:
        return (
            "ESCALATE",
            "Multiple simultaneous technical problems indicate a complex case."
        )

    # ========================================================
    # 7. STRONG CUSTOMER DISTRESS + TECHNICAL PROBLEM
    # ========================================================

    distress_terms = [
        "wtf",
        "shit",
        "f**king",
        "fuck",
        "fucked",

        "ridiculous",
        "terrible",
        "awful",
        "disappointed",
        "hate",

        "do better",
        "fix this now",
        "what a mess",
        "joke",
        "crap",
        "junk"
    ]

    has_distress = any(
        term in text
        for term in distress_terms
    )

    has_technical_problem = any(
        term in text
        for term in technical_terms
    )

    if has_distress and has_technical_problem:
        return (
            "ESCALATE",
            "Strong customer dissatisfaction combined with a technical problem."
        )

    # ========================================================
    # 8. LOW CLASSIFICATION CONFIDENCE
    # ========================================================

    problem_terms = [
        "problem",
        "issue",
        "error",
        "broken",
        "not working",
        "doesn't work",
        "doesnt work",

        "won't",
        "wont",
        "can't",
        "cant",

        "help",
        "failed",
        "failure"
    ]

    if (
        confidence < 0.45
        and len(text.split()) >= 5
        and any(term in text for term in problem_terms)
    ):
        return (
            "ESCALATE",
            "Low classification confidence for a concrete customer problem."
        )

    # ========================================================
    # 9. NO HISTORICAL EVIDENCE
    # ========================================================

    if not retrieved_cases:
        return (
            "ESCALATE",
            "No relevant historical support evidence was retrieved."
        )

    # ========================================================
    # 10. LOW RETRIEVAL SIMILARITY
    # ========================================================

    try:
        top_score = float(
            retrieved_cases[0].get(
                "score",
                0.0
            )
        )

        if top_score < 0.55:
            return (
                "ESCALATE",
                "Historical evidence is too weak for confident automated handling."
            )

    except (
        ValueError,
        TypeError,
        AttributeError,
        IndexError
    ):
        pass

    # ========================================================
    # 11. DEFAULT: AUTO-HANDLE
    # ========================================================

    return (
        "AUTO-HANDLE",
        "Relevant historical evidence supports automated handling."
    )

# ============================================================
# COMPLETE AGENT PIPELINE
# ============================================================

def run_agent(message: str) -> AgentResult:
    """
    Run the complete AppleSupport AI agent.

    Steps:
        1. Classify the customer message.
        2. Retrieve similar historical AppleSupport cases.
        3. Generate a grounded reply using Ollama.
        4. Decide AUTO-HANDLE or ESCALATE.
        5. Return all results in one AgentResult object.
    """

    message = str(message).strip()

    # 1. Intent classification
    intent, confidence = classify_intent(message)

    # 2. Historical case retrieval
    retrieved_cases = retrieve_cases(message, top_k=5)

    # 3. Grounded reply generation
    reply = generate_reply(
        message=message,
        intent=intent,
        retrieved_cases=retrieved_cases
    )

    # 4. Escalation decision
    decision, reason = decide_escalation(
        message=message,
        intent=intent,
        confidence=confidence,
        retrieved_cases=retrieved_cases
    )

    # 5. Prepare short evidence strings for display/evaluation
    evidence = []

    for case in retrieved_cases:
        customer_text = str(case.get("customer_text", "")).strip()
        brand_reply = str(case.get("brand_reply", "")).strip()
        score = float(case.get("score", 0.0))

        evidence.append(
            f"Similarity {score:.4f} | "
            f"Customer: {customer_text} | "
            f"AppleSupport: {brand_reply}"
        )

    return AgentResult(
        intent=intent,
        confidence=float(confidence),
        reply=str(reply),
        decision=decision,
        reason=reason,
        evidence=evidence
    )


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():
    print()
    print("=" * 60)
    print("APPLE SUPPORT AI AGENT")
    print("=" * 60)
    print("Type a customer message.")
    print("Type 'exit' to stop.")
    print()

    while True:
        message = input("Customer: ").strip()

        if message.lower() == "exit":
            print("Exiting agent...")
            break

        if not message:
            print("Please enter a customer message.")
            continue

        print()
        print("Processing...")

        result = run_agent(message)

        print()
        print("-" * 60)
        print("AI AGENT RESULT")
        print("-" * 60)

        print(f"Intent      : {result.intent}")
        print(f"Confidence  : {result.confidence:.2f}")
        print(f"Decision    : {result.decision}")
        print(f"Reason      : {result.reason}")

        print()
        print("Reply:")
        print(result.reply)

        print()
        print("Historical Evidence:")

        if result.evidence:
            for i, evidence in enumerate(result.evidence, 1):
                print(f"{i}. {evidence}")
        else:
            print("No historical evidence found.")

        print("-" * 60)
        print()


if __name__ == "__main__":
    main()
