"""
tests/test_crisis.py — Unit Tests for Crisis Detection & Safeguards

These tests are CRITICAL. A failure here could mean a real user in crisis
receives a generic wellness tip instead of emergency resources.

Run:  pytest tests/test_crisis.py -v
"""

import pytest
from modules.crisis import (
    assess_crisis,
    apply_crisis_guardrails,
    CrisisAssessment,
)


# ══════════════════════════════════════════════════════════════════════════════
# Tier 2 (Level 1) — High distress, prepend helplines
# ══════════════════════════════════════════════════════════════════════════════

class TestTier2Detection:

    @pytest.mark.parametrize("text", [
        "I feel completely hopeless about everything",
        "I feel like such a burden to everyone around me",
        "I feel so trapped and can't see a way out",
        "I'm so tired of everything, I just want it to stop",
        "nothing matters anymore, what's the point",
        "I feel worthless and nobody would care if I was gone",
    ])
    def test_detects_high_distress(self, text):
        result = assess_crisis(text)
        assert result.level >= 1, f"Expected level >= 1 for: '{text}'"
        assert result.triggered


# ══════════════════════════════════════════════════════════════════════════════
# Tier 1 (Level 2) — Explicit crisis, full LLM override
# ══════════════════════════════════════════════════════════════════════════════

class TestTier1Detection:

    @pytest.mark.parametrize("text", [
        "I want to kill myself",
        "I'm thinking about suicide",
        "I want to end my life",
        "I don't want to live anymore",
        "I'm going to hurt myself",
        "I want to overdose on my medication",
        "I've been cutting myself",
        "I want to hang myself",
        "everyone would be better off without me",
        "I have no reason to live",
    ])
    def test_detects_explicit_crisis(self, text):
        result = assess_crisis(text)
        assert result.level == 2, f"Expected level 2 (CRISIS) for: '{text}'"
        assert result.triggered
        assert result.safe_response, "Safe response must not be empty on level 2"

    def test_safe_response_contains_helpline(self):
        result = assess_crisis("I want to end my life")
        assert result.level == 2
        response = result.safe_response
        # Must contain at least one crisis number
        assert any(n in response for n in ["988", "116 123", "9152987821", "741741"]), \
            "Safe response must include crisis helpline numbers"

    def test_safe_response_is_warm_not_clinical(self):
        result = assess_crisis("I don't want to live anymore")
        response = result.safe_response.lower()
        # Should contain empathetic language
        warm_words = ["you matter", "you're not alone", "hear you", "care", "here with you"]
        has_warmth = any(w in response for w in warm_words)
        assert has_warmth, "Safe response must contain warm, empathetic language"


# ══════════════════════════════════════════════════════════════════════════════
# Level 0 — Normal messages should NOT trigger
# ══════════════════════════════════════════════════════════════════════════════

class TestNoFalsePositives:

    @pytest.mark.parametrize("text", [
        "I feel a bit sad today but I'm managing",
        "I'm stressed about work but nothing serious",
        "I had a difficult conversation with my partner",
        "I've been feeling anxious about my job interview",
        "I'm going through a tough time but I have support",
        "I feel happy most days but sometimes low",
        "I want to improve my mental health",
        "how do I cope with stress better",
        "I want to talk about my relationship",
    ])
    def test_no_false_positive(self, text):
        result = assess_crisis(text)
        assert result.level == 0, f"False positive crisis for: '{text}' (level={result.level})"

    def test_word_kill_in_neutral_context(self):
        # "killing it" or "kill time" should NOT trigger
        result = assess_crisis("I'm totally killing it at the gym today!")
        # Regex should not match "killing it" as crisis
        # This tests pattern specificity
        assert result.level < 2, "'Killing it' should not be a crisis trigger"


# ══════════════════════════════════════════════════════════════════════════════
# apply_crisis_guardrails integration
# ══════════════════════════════════════════════════════════════════════════════

class TestApplyGuardrails:

    def test_level2_overrides_llm_completely(self):
        crisis_text = "I want to kill myself right now"
        llm_draft = "Have you tried deep breathing? It might help with stress."
        final, was_overridden = apply_crisis_guardrails(crisis_text, llm_draft)

        assert was_overridden, "Level 2 crisis must override LLM"
        assert llm_draft not in final, "LLM draft must not appear in final response"
        assert "988" in final or "116" in final, "Final must include helplines"

    def test_level1_prepends_not_replaces(self):
        distress_text = "I feel so hopeless and worthless lately"
        llm_draft = "It sounds like you're going through a really tough time."
        final, was_overridden = apply_crisis_guardrails(distress_text, llm_draft)

        assert not was_overridden, "Level 1 should not fully override"
        assert llm_draft in final, "LLM draft should appear in level 1 response"
        # Helplines should still be prepended
        assert "988" in final or "116" in final or "9152987821" in final

    def test_level0_passes_through_unchanged(self):
        normal_text = "I've been feeling a bit stressed lately"
        llm_draft = "That sounds tough. Can you tell me more about what's been stressful?"
        final, was_overridden = apply_crisis_guardrails(normal_text, llm_draft)

        assert not was_overridden
        assert final == llm_draft, "Level 0 should return LLM draft unchanged"


# ══════════════════════════════════════════════════════════════════════════════
# CrisisAssessment dataclass
# ══════════════════════════════════════════════════════════════════════════════

class TestCrisisAssessmentDataclass:

    def test_level_0_not_triggered(self):
        result = assess_crisis("I feel okay today")
        assert isinstance(result, CrisisAssessment)
        assert not result.triggered
        assert result.level == 0
        assert result.safe_response == ""

    def test_level_2_has_matched_pattern(self):
        result = assess_crisis("I want to end my life")
        assert result.matched_pattern != ""
