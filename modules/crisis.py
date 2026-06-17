"""
modules/crisis.py — Crisis Detection & Ethical Safeguards

WHY: This is the most important module in the entire system.
A mental health AI that fails to detect suicidal ideation and responds
with a generic wellness tip is not just unhelpful — it could be harmful.

HOW: Two-layer detection.
  Layer 1 — Keyword matching: fast, zero-latency, high recall.
             Catches explicit phrases even if the LLM misses them.
  Layer 2 — Semantic scoring: VADER compound score cross-referenced
             with distress keyword density for context-aware detection.

When triggered, the normal LLM response is OVERRIDDEN entirely with a
structured safe-messaging response that follows international crisis
communication guidelines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()

# ── Tiered keyword lists ──────────────────────────────────────────────────────
# Tier 1: Immediate crisis — override response unconditionally
_TIER1_PATTERNS = [
    r"\b(kill|end|take)\s+(my(self)?|my life|it all)\b",
    r"\bsuicid(e|al|ally)?\b",
    r"\bwant\s+to\s+die\b",
    r"\bdon'?t\s+want\s+to\s+(be\s+here|live|exist)\b",
    r"\bself[-\s]?harm\b",
    r"\bcut(ting)?\s+(my)?self\b",
    r"\boverdos(e|ing)\b",
    r"\bjump\s+off\b",
    r"\bhang\s+(my)?self\b",
    r"\bno\s+reason\s+to\s+live\b",
    r"\bno\s+way\s+out\b",
    r"\beveryone\s+(would\s+be\s+better|is\s+better)\s+off\s+without\s+me\b",
]

# Tier 2: High distress — prepend crisis resources but allow LLM to respond
_TIER2_KEYWORDS = {
    "hopeless", "worthless", "burden", "trapped", "exhausted from living",
    "can't go on", "give up", "nothing matters", "no point",
    "tired of everything", "can't do this anymore", "want to disappear",
    "falling apart", "breaking down", "end the pain",
}

_tier1_compiled = [re.compile(p, re.IGNORECASE) for p in _TIER1_PATTERNS]


@dataclass
class CrisisAssessment:
    level: int           # 0 = safe, 1 = distress, 2 = crisis
    triggered: bool
    matched_pattern: str
    safe_response: str   # pre-written safe message (overrides LLM for level 2)


_SAFE_RESPONSE_TEMPLATE = """
I hear you, and I want you to know: **what you're feeling is real, and you matter**.

Right now, the most important thing is that you're not alone in this moment.

🆘 **Please reach out to a crisis line right now — they're free, confidential, and available 24/7:**

- 🇺🇸 **USA** — Call or text **988** (Suicide & Crisis Lifeline)
- 🇬🇧 **UK** — Call **116 123** (Samaritans, free)
- 🇮🇳 **India** — Call **9152987821** (iCall) or **1860-2662-345** (Vandrevala, 24/7)
- 🌐 **Find your country** → [findahelpline.com](https://findahelpline.com)
- 📱 **Crisis Text** — Text **HOME** to **741741** (USA, UK, Canada, Ireland)

If you are in immediate danger, **please call emergency services (911 / 999 / 112)** or go to your nearest emergency room.

I'm here with you. Can you tell me more about what's been happening? You don't have to face this alone. 💙
""".strip()

_DISTRESS_PREPEND = """
💙 I want to start by saying — it takes real courage to share how you're feeling.

Before we talk further, please know that if things ever feel truly unbearable, you can call a crisis line:
- 🇺🇸 **988** (USA) | 🇬🇧 **116 123** (Samaritans) | 🇮🇳 **9152987821** (iCall India)
- 🌐 Find your country: [findahelpline.com](https://findahelpline.com)

---
"""


def assess_crisis(text: str) -> CrisisAssessment:
    """
    Assess text for crisis indicators. Returns CrisisAssessment.

    Level 0 → no action needed
    Level 1 → prepend helpline info, allow LLM response
    Level 2 → override LLM entirely with safe crisis response
    """
    lower = text.lower()

    # Tier 1 — explicit crisis language
    for pattern in _tier1_compiled:
        if pattern.search(text):
            return CrisisAssessment(
                level=2,
                triggered=True,
                matched_pattern=pattern.pattern,
                safe_response=_SAFE_RESPONSE_TEMPLATE,
            )

    # Tier 2 — high distress keywords
    tier2_hit = any(kw in lower for kw in _TIER2_KEYWORDS)

    # Semantic check: very negative VADER + subjective
    scores = _vader.polarity_scores(text)
    semantic_distress = scores["compound"] < -0.7 and scores["neg"] > 0.4

    if tier2_hit or semantic_distress:
        return CrisisAssessment(
            level=1,
            triggered=True,
            matched_pattern="distress_tier2",
            safe_response=_DISTRESS_PREPEND,
        )

    return CrisisAssessment(
        level=0,
        triggered=False,
        matched_pattern="",
        safe_response="",
    )


def apply_crisis_guardrails(text: str, llm_response: str) -> tuple[str, bool]:
    """
    Given user text and a draft LLM response, apply crisis logic.

    Returns:
        (final_response, was_overridden)
    """
    assessment = assess_crisis(text)

    if assessment.level == 2:
        # Full override — LLM response discarded
        return assessment.safe_response, True

    if assessment.level == 1:
        # Prepend safety info to LLM response
        return assessment.safe_response + "\n\n" + llm_response, False

    return llm_response, False
