"""
modules/sentiment.py — Emotion-Aware Intelligence (FINAL FIXED VERSION)
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


# ─────────────────────────────────────────────────────────────
# EMOTION TYPES
# ─────────────────────────────────────────────────────────────

class Emotion(str, Enum):
    HAPPY = "happy"
    NEUTRAL = "neutral"
    SAD = "sad"
    ANXIOUS = "anxious"
    DISTRESSED = "distressed"


@dataclass
class EmotionResult:
    emotion: Emotion
    compound: float
    subjectivity: float
    label: str
    emoji: str
    color: str


# ─────────────────────────────────────────────────────────────
# KEYWORDS
# ─────────────────────────────────────────────────────────────

_ANXIETY_KEYWORDS = {
    "anxious", "anxiety", "panic", "nervous", "worry", "worried",
    "overwhelmed", "stressed", "stress", "scared", "fear", "afraid",
}

_DISTRESS_KEYWORDS = {
    "hopeless", "worthless", "can't go on", "give up", "end it",
    "don't want to live", "no point", "tired of living",
    "kill myself", "suicide", "self-harm",
}


# ─────────────────────────────────────────────────────────────
# EMOTION MAP
# ─────────────────────────────────────────────────────────────

_EMOTION_MAP = {
    Emotion.HAPPY: ("Positive 😊", "😊", "#4CAF50"),
    Emotion.NEUTRAL: ("Neutral 😐", "😐", "#9E9E9E"),
    Emotion.SAD: ("Feeling Down 😔", "😔", "#5C6BC0"),
    Emotion.ANXIOUS: ("Anxious 😰", "😰", "#FF9800"),
    Emotion.DISTRESSED: ("In Distress 🆘", "🆘", "#F44336"),
}


_vader = SentimentIntensityAnalyzer()


# ─────────────────────────────────────────────────────────────
# EMOTION DETECTION
# ─────────────────────────────────────────────────────────────

def detect_emotion(text: str) -> EmotionResult:
    lower = text.lower()

    scores = _vader.polarity_scores(text)
    compound = scores["compound"]

    blob = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity

    # Priority rules
    if any(k in lower for k in _DISTRESS_KEYWORDS):
        emotion = Emotion.DISTRESSED
    elif any(k in lower for k in _ANXIETY_KEYWORDS):
        emotion = Emotion.ANXIOUS
    elif compound < -0.45:
        emotion = Emotion.SAD
    elif compound < -0.1 and subjectivity > 0.5:
        emotion = Emotion.ANXIOUS
    elif compound > 0.3:
        emotion = Emotion.HAPPY
    else:
        emotion = Emotion.NEUTRAL

    label, emoji, color = _EMOTION_MAP[emotion]

    return EmotionResult(
        emotion=emotion,
        compound=round(compound, 3),
        subjectivity=round(subjectivity, 3),
        label=label,
        emoji=emoji,
        color=color,
    )


# ─────────────────────────────────────────────────────────────
# 🔥 SYSTEM PROMPT (FINAL FIX)
# ─────────────────────────────────────────────────────────────

def build_system_prompt(emotion: Emotion, retrieved_context: str = "") -> str:
    """
    FINAL FIX:
    - Forces English output
    - Controls LLM randomness
    - Keeps emotional intelligence
    """

    base = (
        "You are MindBridge, a compassionate AI mental health assistant.\n\n"

        "🚨 STRICT RULES:\n"
        "1. ALWAYS respond in English.\n"
        "2. NEVER switch language.\n"
        "3. NEVER output non-English text.\n\n"

        "You are supportive, warm, and emotionally intelligent.\n"
        "You use CBT, DBT, and mindfulness-based techniques.\n"
        "You are not a replacement for therapy.\n\n"
    )

    tone_map = {
        Emotion.HAPPY: (
            "User is happy. Encourage positivity and reinforce good habits."
        ),
        Emotion.NEUTRAL: (
            "User is neutral. Ask gentle questions and explore feelings."
        ),
        Emotion.SAD: (
            "User is sad. Show empathy. Validate feelings. Be supportive."
        ),
        Emotion.ANXIOUS: (
            "User is anxious. Provide grounding techniques and calm tone."
        ),
        Emotion.DISTRESSED: (
            "User is in distress. PRIORITIZE safety. Encourage seeking help immediately."
        ),
    }

    context_block = (
        f"\nRelevant Context:\n{retrieved_context}"
        if retrieved_context.strip()
        else ""
    )

    return f"{base}\nTone Guidance: {tone_map[emotion]}{context_block}"