"""
tests/test_domain_guard.py — Unit Tests for Domain Guard

Run:  pytest tests/test_domain_guard.py -v
"""

import pytest
from modules.domain_guard import (
    DomainGuard,
    Verdict,
    _layer1_check,
    _layer2_score,
    _pick_redirect,
)


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 tests — blocklist
# ══════════════════════════════════════════════════════════════════════════════

class TestLayer1:

    @pytest.mark.parametrize("text,expected_label", [
        # Programming
        ("write me a Python function to sort a list", "programming_request"),
        ("create a React component for login", "programming_request"),
        ("debug my code please", "programming_request"),
        ("explain how algorithms work", "programming_request"),
        # Code syntax
        ("def fibonacci(n):", "code_syntax"),
        ("import pandas as pd", "code_syntax"),
        ("for item in my_list:", "code_syntax"),
        # DevOps
        ("I have a git merge conflict", "dev_ops"),
        ("help me with docker deployment", "dev_ops"),
        # Maths
        ("solve this equation: x^2 + 5x + 6 = 0", "math_problem"),
        ("calculate the integral of sin(x)", "math_problem"),
        ("simplify this polynomial expression", "math_problem"),
        ("what is sin(45 degrees)", "math_expression"),
        # General knowledge
        ("what is the capital of France", "general_knowledge"),
        ("who invented the telephone", "general_knowledge"),
        ("what year was the Eiffel Tower built", "general_knowledge"),
        ("recipe for chocolate cake", "general_knowledge"),
        # Finance
        ("what is the Bitcoin price today", "finance_legal"),
        ("should I invest in crypto", "finance_legal"),
        # Creative unrelated
        ("write me a poem about autumn", "creative_task"),
        ("translate this text to Spanish", "creative_task"),
        ("summarize this article for me", "creative_task"),
    ])
    def test_blocks_off_topic(self, text, expected_label):
        verdict, reason = _layer1_check(text)
        assert verdict == Verdict.BLOCK, f"Expected BLOCK for: '{text}'"
        assert reason == expected_label, f"Expected reason '{expected_label}', got '{reason}'"

    @pytest.mark.parametrize("text", [
        "I feel so anxious about my relationship",
        "I can't stop crying and I don't know why",
        "my partner and I keep fighting",
        "I've been feeling really hopeless lately",
        "how do I deal with panic attacks",
        "I'm struggling with depression",
        "I feel lost and have no purpose",
        "my family situation is very stressful",
        "I just want someone to talk to",
        "everything feels overwhelming right now",
    ])
    def test_allows_mental_health(self, text):
        verdict, _ = _layer1_check(text)
        assert verdict == Verdict.UNSURE, f"Expected UNSURE (not block) for: '{text}'"


# ══════════════════════════════════════════════════════════════════════════════
# Layer 2 tests — signal scoring
# ══════════════════════════════════════════════════════════════════════════════

class TestLayer2:

    @pytest.mark.parametrize("text", [
        "I feel so anxious and overwhelmed",
        "I'm struggling with depression and hopelessness",
        "my relationship is falling apart and I feel worthless",
        "I've been having panic attacks and can't sleep",
        "I need help coping with grief after losing someone",
        "feeling really stressed and burned out at work",
        "I have low self-esteem and struggle with confidence",
        "I'm dealing with trauma from my childhood",
    ])
    def test_scores_mental_health_high(self, text):
        verdict, reason, score = _layer2_score(text)
        assert verdict == Verdict.ALLOW, f"Expected ALLOW for: '{text}' (score={score})"
        assert score >= 2

    @pytest.mark.parametrize("text", [
        "the weather is nice today",
        "I like pizza",
        "2 + 2 = 4",
        "hello",
    ])
    def test_scores_neutral_low(self, text):
        verdict, reason, score = _layer2_score(text)
        assert verdict == Verdict.UNSURE, f"Expected UNSURE for: '{text}' (score={score})"
        assert score < 2


# ══════════════════════════════════════════════════════════════════════════════
# DomainGuard full pipeline tests (no LLM — Layer 3 disabled)
# ══════════════════════════════════════════════════════════════════════════════

class TestDomainGuard:

    @pytest.fixture
    def guard(self):
        """Guard with Layer 3 disabled — deterministic, no API calls."""
        return DomainGuard(llm_client=None, use_llm_classifier=False)

    # ── Should BLOCK ──────────────────────────────────────────────────────────

    @pytest.mark.parametrize("text", [
        "write me a Python script to parse JSON",
        "how do I fix this JavaScript error",
        "what is 15% of 340",
        "solve for x: 2x + 4 = 10",
        "what year did World War 2 end",
        "who is the president of France",
        "recipe for pasta carbonara",
        "translate 'hello' to French",
    ])
    def test_blocks_explicit_off_topic(self, guard, text):
        result = guard.check(text)
        assert result.is_blocked, f"Should have blocked: '{text}'"
        assert result.redirect_response, "Redirect response should not be empty"
        assert len(result.redirect_response) > 50, "Redirect should be substantive"

    # ── Should ALLOW ──────────────────────────────────────────────────────────

    @pytest.mark.parametrize("text", [
        "I've been feeling really anxious lately",
        "I'm going through a difficult breakup",
        "I can't stop overthinking everything",
        "my depression has been getting worse",
        "I feel so lonely and isolated",
        "how do I cope with grief",
        "I have panic attacks at work",
        "I feel overwhelmed and stressed",
        "I need someone to talk to about my feelings",
        "I'm struggling with low self-esteem",
    ])
    def test_allows_mental_health(self, guard, text):
        result = guard.check(text)
        assert not result.is_blocked, f"Should have allowed: '{text}'"
        assert result.redirect_response == ""

    # ── Redirect quality ──────────────────────────────────────────────────────

    def test_redirect_is_empathetic(self, guard):
        result = guard.check("write me a sorting algorithm in C++")
        assert result.is_blocked
        # Should not contain harsh language
        redirect = result.redirect_response.lower()
        for harsh_word in ["cannot", "refuse", "not allowed", "forbidden", "invalid"]:
            assert harsh_word not in redirect, f"Found harsh word '{harsh_word}' in redirect"

    def test_redirect_bridges_back(self, guard):
        result = guard.check("explain machine learning to me")
        assert result.is_blocked
        # Should contain an invitation to emotional discussion
        redirect = result.redirect_response.lower()
        bridge_words = ["feeling", "feel", "how are you", "what's on your", "emotional", "heart"]
        has_bridge = any(word in redirect for word in bridge_words)
        assert has_bridge, f"Redirect missing emotional bridge: {result.redirect_response}"

    def test_layer_reported_correctly(self, guard):
        # Programming request → Layer 1
        result = guard.check("write a Python function")
        assert result.layer == 1

        # Mental health → Layer 2 ALLOW
        result = guard.check("I feel so anxious and depressed")
        assert result.layer == 2

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_empty_string(self, guard):
        result = guard.check("")
        # Empty message should not be blocked (could be emotional pause)
        assert not result.is_blocked

    def test_very_short_message(self, guard):
        result = guard.check("hi")
        assert not result.is_blocked

    def test_mixed_topic(self, guard):
        # Code + emotional context — the emotional part should tip the balance
        result = guard.check("I'm so stressed about this coding interview")
        # "stressed" is a strong mental health signal — should allow
        assert not result.is_blocked, "Mixed message with emotional content should be allowed"

    def test_block_returns_layer_1_not_3(self, guard):
        # Layer 3 is disabled; explicit blocks should still come from Layer 1
        result = guard.check("debug my Python code")
        assert result.layer == 1
        assert result.is_blocked


# ══════════════════════════════════════════════════════════════════════════════
# Redirect response quality tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRedirectResponses:

    def test_programming_redirect_is_category_specific(self):
        redirect = _pick_redirect("programming_request")
        assert "coding" in redirect.lower() or "technical" in redirect.lower()

    def test_math_redirect_is_category_specific(self):
        redirect = _pick_redirect("math_problem")
        assert "maths" in redirect.lower() or "math" in redirect.lower()

    def test_general_redirect_randomises(self):
        """Should not always return the same response."""
        responses = set(_pick_redirect("general") for _ in range(20))
        assert len(responses) > 1, "Redirects should be randomised"

    def test_all_redirects_end_with_invitation(self):
        """Every redirect must close with an open question or invitation."""
        for _ in range(10):
            redirect = _pick_redirect("general")
            # Should end with question mark or contain an invitation phrase
            has_question = "?" in redirect
            assert has_question, f"Redirect should contain a question: {redirect}"
