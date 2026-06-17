"""
modules/domain_guard.py — Domain Restriction & Guardrails

WHY: A mental health assistant that answers coding questions or does maths
is not just unhelpful — it erodes user trust, dilutes the product identity,
and creates liability. Every off-topic response is a missed opportunity to
gently bring the user back to what matters: their wellbeing.

DESIGN PHILOSOPHY:
  Never be harsh. A user who asks "can you solve this equation?" may be
  procrastinating because they're anxious. A user who asks "write me code"
  might be using task-focus to cope with stress. The refusal must always:
    1. Acknowledge the request warmly (no cold rejection)
    2. Explain the scope gently
    3. Offer a bridge back to emotional support
    4. Leave the door open

THREE-LAYER DETECTION (ordered by cost, fast → slow):
  Layer 1 — Hard blocklist: explicit off-topic keywords/patterns (regex, ~0ms)
  Layer 2 — Allowlist scoring: count mental-health domain signals (word matching, ~1ms)
  Layer 3 — LLM classifier: call LLM with a binary classification prompt (~500ms)
             Only runs when Layers 1+2 are ambiguous (saves tokens + latency).

Each layer can independently return BLOCK, ALLOW, or UNSURE.
Final decision: first non-UNSURE verdict wins, Layer 3 as tiebreaker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import random


# ══════════════════════════════════════════════════════════════════════════════
# Decision types
# ══════════════════════════════════════════════════════════════════════════════

class Verdict(str, Enum):
    ALLOW  = "allow"
    BLOCK  = "block"
    UNSURE = "unsure"


@dataclass
class DomainCheckResult:
    verdict: Verdict
    layer: int                   # which layer made the final call (1/2/3)
    reason: str                  # internal reason string for logging
    redirect_response: str       # pre-written empathetic redirect (empty if ALLOW)
    is_blocked: bool             # convenience alias


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — Hard blocklist (explicit off-topic patterns)
# ══════════════════════════════════════════════════════════════════════════════

# These patterns are unambiguously outside mental health scope.
# Tuned to be specific — avoid false positives on "python of emotions", etc.
_BLOCK_PATTERNS: list[tuple[str, str]] = [
    # Programming / technical
    (r"\b(write|create|build|make|generate|debug|fix|explain|implement)\s+(a\s+)?"
     r"(code|script|program|function|class|algorithm|api|database|query|sql|html|css|"
     r"javascript|python|java|c\+\+|ruby|react|node|flask|django|app|website|webpage)\b",
     "programming_request"),

    (r"\b(def |import |print\(|for .+ in |while |if __name__|async def|await )\b",
     "code_syntax"),

    (r"\b(bug|syntax error|stack overflow|null pointer|segfault|compile|runtime error|"
     r"git (commit|push|pull|merge|clone)|docker|kubernetes|devops|CI/CD)\b",
     "dev_ops"),

    # Mathematics / academic
    (r"\b(solve|calculate|compute|differentiate|integrate|factorise|factorise|simplify)\s+"
     r"(this\s+)?(equation|expression|polynomial|matrix|integral|derivative|formula)\b",
     "math_problem"),

    (r"\b(\d+\s*[\+\-\*\/\^]\s*\d+|\bsin\b|\bcos\b|\btan\b|\blog\b|\bln\b|\bsqrt\b)\b",
     "math_expression"),

    (r"\b(algebra|calculus|trigonometry|geometry|statistics|probability theory|"
     r"linear algebra|differential equation)\b",
     "math_subject"),

    # General knowledge / trivia
    (r"\b(capital (city|of)|population of|history of|who (invented|discovered|wrote|"
     r"created|founded)|when (was|did|were)|what (year|century)|geography|"
     r"recipe for|how (to cook|to make food|to bake)|convert \d+ (kg|lb|km|miles))\b",
     "general_knowledge"),

    # Finance / legal / technical domains
    (r"\b(stock (price|market|ticker)|crypto(currency)?|bitcoin|ethereum|nft|"
     r"investment (advice|portfolio)|tax (return|filing)|legal (advice|document)|"
     r"patent|trademark|copyright law|immigration)\b",
     "finance_legal"),

    # Entertainment requests (not emotional discussion)
    (r"\b(write (me a|a) (song|poem|story|essay|novel|joke|rap|screenplay)|"
     r"translate (this text|to (french|spanish|german|hindi|arabic))|"
     r"summarize (this article|this text|this document))\b",
     "creative_task"),

    # Sports / games / trivia
    (r"\b(who (won|scored|played in)|sports (score|result|standings)|"
     r"game (walkthrough|cheat code|strategy guide))\b",
     "sports_games"),
]

_block_compiled = [
    (re.compile(pattern, re.IGNORECASE), label)
    for pattern, label in _BLOCK_PATTERNS
]


def _layer1_check(text: str) -> tuple[Verdict, str]:
    for regex, label in _block_compiled:
        if regex.search(text):
            return Verdict.BLOCK, label
    return Verdict.UNSURE, ""


# ══════════════════════════════════════════════════════════════════════════════
# Layer 2 — Allowlist scoring (mental health domain signals)
# ══════════════════════════════════════════════════════════════════════════════

# Positive signals: words that strongly suggest mental health context
_MENTAL_HEALTH_SIGNALS = {
    # Emotional states
    "sad", "happy", "anxious", "anxiety", "depressed", "depression", "lonely",
    "lonely", "hopeless", "hopeful", "angry", "frustrated", "overwhelmed",
    "stressed", "stress", "burnout", "exhausted", "numb", "empty", "hurt",
    "scared", "afraid", "nervous", "worry", "worried", "grief", "grieving",
    "trauma", "traumatic", "ptsd", "panic", "panic attack",

    # Relationships & life
    "relationship", "breakup", "divorce", "family", "parents", "partner",
    "friend", "friendship", "loneliness", "isolation", "rejection", "abandoned",
    "trust", "boundary", "toxic", "abuse", "conflict", "communication",
    "attachment", "love", "heartbreak", "loss", "death", "grief",

    # Mental health terms
    "mental health", "therapy", "therapist", "counseling", "counsellor",
    "psychiatrist", "medication", "antidepressant", "bipolar", "ocd",
    "adhd", "eating disorder", "self-esteem", "confidence", "self-worth",
    "mindfulness", "meditation", "coping", "healing", "recovery",

    # Life challenges
    "work stress", "job loss", "unemployment", "purpose", "meaning",
    "motivation", "procrastination", "sleep", "insomnia", "fatigue",
    "addiction", "sobriety", "substance", "alcohol", "identity",
    "career", "life", "feeling", "feel", "emotion", "mind", "heart",

    # Help-seeking
    "help", "support", "talk", "listen", "understand", "advice", "cope",
    "better", "improve", "struggling", "difficult", "hard time",
}

# Context bridging: ambiguous words that could be mental health in context
_AMBIGUOUS_SIGNALS = {
    "thinking", "thought", "feeling", "wonder", "lost", "stuck",
    "confused", "unsure", "overwhelmed", "pressure", "problem",
    "issue", "situation", "life", "day", "today", "lately",
}


def _layer2_score(text: str) -> tuple[Verdict, str, int]:
    """
    Score text against mental health signal vocabulary.
    Returns (verdict, reason, score).

    Scoring:
      strong signal  (+2 each)  → from _MENTAL_HEALTH_SIGNALS
      ambiguous      (+1 each)  → from _AMBIGUOUS_SIGNALS

    Thresholds:
      score >= 2 → ALLOW  (has meaningful mental health content)
      score == 1 → UNSURE (ambiguous, pass to Layer 3)
      score == 0 → UNSURE (no signals found, but not explicitly blocked)
    """
    lower = text.lower()
    score = 0
    matched: list[str] = []

    for signal in _MENTAL_HEALTH_SIGNALS:
        if signal in lower:
            score += 2
            matched.append(signal)

    for signal in _AMBIGUOUS_SIGNALS:
        if signal in lower:
            score += 1

    if score >= 2:
        return Verdict.ALLOW, f"mental_health_signals:{','.join(matched[:3])}", score
    return Verdict.UNSURE, f"low_signal_score:{score}", score


# ══════════════════════════════════════════════════════════════════════════════
# Layer 3 — LLM-based classifier (optional, runs only when ambiguous)
# ══════════════════════════════════════════════════════════════════════════════

_CLASSIFICATION_SYSTEM = """You are a domain classifier for a mental health support chatbot.
Your ONLY job is to classify whether a user message is within scope.

IN-SCOPE topics (respond YES):
- Mental health, emotional wellbeing, mood, feelings
- Stress, anxiety, depression, grief, trauma
- Relationships, family, friendship, loneliness, heartbreak
- Life challenges, purpose, identity, self-esteem
- Sleep problems, burnout, motivation, coping strategies
- Asking for emotional support, someone to talk to
- Vague/unclear messages that could reflect emotional state

OUT-OF-SCOPE topics (respond NO):
- Programming, coding, software, algorithms
- Mathematics, science, academic subjects
- General knowledge, trivia, history, geography
- Recipes, cooking, sports scores
- Legal or financial advice
- Translation of neutral text
- Creative writing unrelated to emotions

Respond with ONLY one word: YES or NO"""


def _layer3_llm_classify(text: str, llm_client) -> tuple[Verdict, str]:
    """
    Call the LLM with a cheap binary classification prompt.
    Used only when Layers 1+2 are ambiguous — saves tokens.
    Returns (verdict, reason).
    """
    if llm_client is None:
        return Verdict.ALLOW, "llm_unavailable_defaulting_allow"

    try:
        result = llm_client.chat(
            messages=[{"role": "user", "content": f'Message: "{text}"\n\nIs this in scope?'}],
            system_prompt=_CLASSIFICATION_SYSTEM,
            max_tokens=5,
            temperature=0.0,
        )
        answer = result.strip().upper()
        if "YES" in answer:
            return Verdict.ALLOW, "llm_classifier:yes"
        elif "NO" in answer:
            return Verdict.BLOCK, "llm_classifier:no"
        return Verdict.ALLOW, "llm_classifier:unclear_defaulting_allow"
    except Exception as e:
        # Fail open — don't block on classifier errors
        return Verdict.ALLOW, f"llm_classifier_error:{e}"


# ══════════════════════════════════════════════════════════════════════════════
# Redirect responses — warm, empathetic, varied
# ══════════════════════════════════════════════════════════════════════════════

_REDIRECT_RESPONSES = [
    (
        "That's a bit outside what I'm able to help with — I'm here specifically "
        "for emotional support and mental wellbeing. 💙\n\n"
        "But I'm curious — is there something on your mind *beyond* that question? "
        "Sometimes when we're stressed or anxious, we focus on tasks to distract ourselves. "
        "How are you actually feeling today?"
    ),
    (
        "I appreciate you reaching out! That particular topic is outside my lane — "
        "I'm a mental health companion, so I'm best placed to support you with "
        "feelings, relationships, stress, and life challenges.\n\n"
        "Is there anything going on emotionally that brought you here today? "
        "I'm all ears, no judgment. 🌿"
    ),
    (
        "Hmm, that's not quite something I can help with — my focus is entirely "
        "on your mental and emotional wellbeing.\n\n"
        "I'd gently ask though: how are *you* doing right now? "
        "Sometimes the questions we ask on the surface aren't the ones weighing on us most. "
        "Feel free to share what's really going on. 💬"
    ),
    (
        "That one's a little outside my expertise — I'm specifically designed to "
        "support mental health, emotional wellbeing, and life's tougher moments.\n\n"
        "What I *can* do is listen, help you process difficult emotions, explore "
        "coping strategies, or just be a calm presence when things feel heavy. "
        "What would feel most helpful for you right now? 🌱"
    ),
    (
        "I'm not the right tool for that, I'm afraid — but I don't want to just "
        "turn you away. 💙\n\n"
        "I'm here for the emotional stuff: stress, anxiety, relationships, "
        "feeling lost or overwhelmed. If any of that resonates, I'm listening. "
        "What's been on your heart lately?"
    ),
]


def _pick_redirect(category: str) -> str:
    """Pick a redirect response. Category-specific variants > random fallback."""
    _CATEGORY_SPECIFIC: dict[str, str] = {
        "programming_request": (
            "I can't help with coding, but I notice that sometimes people bury "
            "themselves in technical work when something deeper is bothering them. 💙\n\n"
            "Is everything okay with you? I'm here if you want to talk about what's "
            "really going on beneath the surface."
        ),
        "math_problem": (
            "Maths is a bit beyond my world — I live in the realm of feelings and "
            "mental wellbeing! 😊\n\n"
            "How are *you* doing today, though? Any stress, worries, or things "
            "weighing on you? I'm genuinely here to listen."
        ),
        "general_knowledge": (
            "That's more of a search-engine question than a heart question! "
            "I'm here for the emotional and mental health side of life.\n\n"
            "But since you're here — how are you feeling today? "
            "Any worries, frustrations, or things on your mind? 🌿"
        ),
    }
    if category in _CATEGORY_SPECIFIC:
        return _CATEGORY_SPECIFIC[category]
    return random.choice(_REDIRECT_RESPONSES)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

class DomainGuard:
    """
    Three-layer domain restriction guard.

    Usage:
        guard = DomainGuard(llm_client=client)  # pass None to skip Layer 3
        result = guard.check("how do I sort a list in python?")
        if result.is_blocked:
            return result.redirect_response
    """

    def __init__(self, llm_client=None, use_llm_classifier: bool = True) -> None:
        self._llm = llm_client
        self._use_llm = use_llm_classifier

    def check(self, text: str) -> DomainCheckResult:
        """
        Run all three layers and return a DomainCheckResult.

        Fast-path: Layer 1 block exits immediately (no LLM call).
        Slow-path: Layer 3 LLM only fires when 1+2 are ambiguous.
        """
        # ── Layer 1 ───────────────────────────────────────────────────────────
        l1_verdict, l1_reason = _layer1_check(text)
        if l1_verdict == Verdict.BLOCK:
            return DomainCheckResult(
                verdict=Verdict.BLOCK,
                layer=1,
                reason=l1_reason,
                redirect_response=_pick_redirect(l1_reason),
                is_blocked=True,
            )

        # ── Layer 2 ───────────────────────────────────────────────────────────
        l2_verdict, l2_reason, _ = _layer2_score(text)
        if l2_verdict == Verdict.ALLOW:
            return DomainCheckResult(
                verdict=Verdict.ALLOW,
                layer=2,
                reason=l2_reason,
                redirect_response="",
                is_blocked=False,
            )

        # ── Layer 3 (LLM classifier) ──────────────────────────────────────────
        if self._use_llm and self._llm is not None:
            l3_verdict, l3_reason = _layer3_llm_classify(text, self._llm)
            if l3_verdict == Verdict.BLOCK:
                return DomainCheckResult(
                    verdict=Verdict.BLOCK,
                    layer=3,
                    reason=l3_reason,
                    redirect_response=_pick_redirect("general"),
                    is_blocked=True,
                )
            return DomainCheckResult(
                verdict=Verdict.ALLOW,
                layer=3,
                reason=l3_reason,
                redirect_response="",
                is_blocked=False,
            )

        # Default: ALLOW when uncertain and no LLM available (fail-open)
        return DomainCheckResult(
            verdict=Verdict.ALLOW,
            layer=2,
            reason="fail_open_no_classifier",
            redirect_response="",
            is_blocked=False,
        )
