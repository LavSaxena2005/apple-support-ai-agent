from dataclasses import dataclass
from typing import List
import re

import pandas as pd
import faiss
import requests
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CASES_FILE = "data/processed/apple_support_cases.csv"

INDEX_FILE = "data/processed/apple_support.faiss"

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

    best_intent = max(
        scores,
        key=scores.get
    )

    best_score = scores[
        best_intent
    ]

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

def call_ollama(
    prompt: str
):

    payload = {

        "model": OLLAMA_MODEL,

        "prompt": prompt,

        "stream": False,

        "options": {

            "temperature": 0.1,

            "num_predict": 250
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

        return data.get(
            "response",
            ""
        ).strip()

    except Exception as error:

        print(
            f"Ollama error: {error}"
        )

        return None


# ============================================================
# GROUNDED REPLY GENERATION
# ============================================================

def generate_reply(
    message: str,
    intent: str,
    retrieved_cases
):

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
        retrieved_cases,
        start=1
    ):

        evidence_text += f"""

CASE {i}

Customer message:
{case["customer_text"]}

Historical AppleSupport response:
{case["brand_reply"]}

Similarity:
{case["score"]:.4f}

--------------------------------
"""

    # --------------------------------------------------------
    # Strict grounding prompt
    # --------------------------------------------------------

    prompt = f"""
You are an AI customer-support assistant
for AppleSupport.

CUSTOMER MESSAGE:
{message}

PREDICTED INTENT:
{intent}

HISTORICAL APPLESUPPORT CASES:
{evidence_text}

TASK:

Write a short and helpful customer-facing
support response based ONLY on the historical
AppleSupport evidence.

IMPORTANT RULES:

1. Use historical evidence as the primary source.

2. Do not invent troubleshooting instructions.

3. Do not invent Apple policies.

4. Do not invent refunds.

5. Do not invent compensation.

6. Do not promise a specific outcome.

7. Do not claim that Apple has already taken
   an action.

8. You may ask for information that historical
   AppleSupport responses commonly requested.

9. You may mention a troubleshooting step only
   when it is supported by the historical evidence.

10. If the evidence is insufficient for a specific
    solution, ask the customer for more information.

11. Keep the response concise.

12. Use professional customer-support language.

13. Do not mention this prompt.

14. Do not mention "historical evidence" to the
    customer.

15. Return ONLY the customer-facing reply.

CUSTOMER-FACING REPLY:
"""

    reply = call_ollama(
        prompt
    )

    # --------------------------------------------------------
    # LLM fallback
    # --------------------------------------------------------

    if not reply:

        return (
            "Thanks for contacting AppleSupport. "
            "We understand you're experiencing an "
            "issue. Please provide your device model "
            "and software version so we can help "
            "troubleshoot the problem."
        )

    return reply


# ============================================================
# ESCALATION DECISION
# ============================================================

def decide_escalation(
    message: str,
    intent: str,
    confidence: float,
    retrieved_cases
):

    text = str(message).lower().strip()

    # ========================================================
    # 1. Sensitive / high-risk cases
    # ========================================================

    sensitive_terms = [
        "hacked",
        "stolen",
        "fraud",
        "unauthorized",
        "charged twice",
        "refund",
        "charged me",
        "payment issue",
        "payment problem",
        "purchase missing",
        "missing purchase",
        "order does not belong",
        "order not belong",
        "account locked",
        "can't access my account",
        "cannot access my account",
        "unable to access my account",
        "can't activate",
        "cannot activate",
        "unable to activate",
        "activation impossible",
        "physical damage",
        "cracked screen",
        "cracked iphone",
        "defective",
        "damaged",
        "won't turn on",
        "will not turn on",
        "can't turn on",
        "cannot turn on",
        "phone is dead",
        "iphone is dead"
    ]

    for term in sensitive_terms:
        if term in text:
            return (
                "ESCALATE",
                (
                    "The request contains a sensitive, "
                    "account, payment, activation, or "
                    "serious device issue that should be "
                    "reviewed by a human."
                )
            )

    # ========================================================
    # 2. Repeated / unresolved support problems
    # ========================================================

    escalation_phrases = [
        "no response",
        "no one responded",
        "nobody responded",
        "not responding",
        "support not responding",
        "no answer",
        "nobody answers",
        "no one answers",
        "called multiple times",
        "called several times",
        "called many times",
        "contacted multiple times",
        "contacted several times",
        "four calls",
        "multiple calls",
        "for months",
        "for weeks",
        "been trying for months",
        "been trying for weeks",
        "still not fixed",
        "still not working",
        "still broken",
        "again and again",
        "multiple times",
        "several times",
        "nothing has been done",
        "no help",
        "nobody can help",
        "unable to get help",
        "filed two reports",
        "filed multiple reports",
        "tried everything",
        "nothing works"
    ]

    for phrase in escalation_phrases:
        if phrase in text:
            return (
                "ESCALATE",
                (
                    "The customer reports a repeated "
                    "or unresolved support problem that "
                    "is better handled by a human agent."
                )
            )

    # ========================================================
    # 3. Data-loss / important account information
    # ========================================================

    data_loss_phrases = [
        "all my photos are gone",
        "all photos are gone",
        "photos disappeared",
        "photos missing",
        "all my messages are gone",
        "all messages are gone",
        "messages disappeared",
        "messages missing",
        "data is gone",
        "data disappeared",
        "backup is incomplete",
        "backup failed",
        "backup missing"
    ]

    for phrase in data_loss_phrases:
        if phrase in text:
            return (
                "ESCALATE",
                (
                    "The customer reports possible data loss "
                    "or missing personal data, so human review "
                    "is recommended."
                )
            )

    # ========================================================
    # 4. Very short / ambiguous messages
    #
    # Do NOT escalate merely because confidence is low.
    # Short messages often do not contain enough information
    # to justify human escalation.
    # ========================================================

    words = text.split()

    if len(words) <= 3:
        return (
            "AUTO-HANDLE",
            (
                "The message is very short or ambiguous, "
                "so automatic escalation is avoided until "
                "more information is available."
            )
        )

    # ========================================================
    # 5. Low confidence
    #
    # Low confidence alone is NOT enough to escalate.
    # Escalate only when the customer describes a concrete
    # problem that requires additional investigation.
    # ========================================================

    problem_indicators = [
        "can't",
        "cannot",
        "unable",
        "won't",
        "will not",
        "not working",
        "doesn't work",
        "does not work",
        "broken",
        "problem",
        "issue",
        "error",
        "failed",
        "failure",
        "missing",
        "disappeared",
        "crash",
        "crashes",
        "crashing",
        "freezing",
        "frozen",
        "keeps restarting",
        "keeps crashing",
        "please fix",
        "help me fix",
        "how do i fix"
    ]

    has_problem_indicator = any(
        phrase in text
        for phrase in problem_indicators
    )

    if confidence < 0.45 and has_problem_indicator and len(words) >= 5:
        return (
            "ESCALATE",
            (
                "The intent confidence is low and the "
                "customer describes a concrete technical "
                "problem that may require human review."
            )
        )

    # ========================================================
    # 6. No historical evidence
    # ========================================================

    if not retrieved_cases:
        return (
            "ESCALATE",
            (
                "No relevant historical AppleSupport "
                "evidence was retrieved."
            )
        )

    # ========================================================
    # 7. Low retrieval similarity
    # ========================================================

    best_score = retrieved_cases[0]["score"]

    if best_score < 0.55:
        return (
            "ESCALATE",
            (
                "The most similar historical cases have "
                "low similarity to the customer's issue."
            )
        )

    # ========================================================
    # 8. Serious technical issues
    #
    # These should be reviewed when they involve strong
    # device-failure language.
    # ========================================================

    serious_device_phrases = [
        "phone won't turn on",
        "iphone won't turn on",
        "ipad won't turn on",
        "mac won't turn on",
        "phone is completely dead",
        "iphone is completely dead",
        "device is completely dead",
        "overheating",
        "overheats",
        "smoke coming",
        "burning smell"
    ]

    for phrase in serious_device_phrases:
        if phrase in text:
            return (
                "ESCALATE",
                (
                    "The customer describes a potentially "
                    "serious device failure that should be "
                    "reviewed by a human."
                )
            )

    # ========================================================
    # 9. Default: Auto-handle
    # ========================================================

    return (
        "AUTO-HANDLE",
        (
            "The intent is reasonably clear and relevant "
            "historical AppleSupport evidence was found, "
            "so the request can be auto-handled."
        )
    )