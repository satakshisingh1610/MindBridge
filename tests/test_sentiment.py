"""
tests/test_sentiment.py — Unit Tests for Emotion Detection

Run:  pytest tests/test_sentiment.py -v
"""

import pytest
from modules.sentiment import detect_emotion, build_system_prompt, Emotion, EmotionResult


class TestDetectEmotion:

    # ── Happy ─────────────────────────────────────────────────────────────────
    @pytest.mark.parametrize("text", [
        "I feel amazing today, everything is going great!",
        "I'm so happy and grateful for everything in my life",
        "Things are really looking up, I feel wonderful",
    ])
    def test_detects_happy(self, text):
        result = detect_emotion(text)
        assert result.emotion == Emotion.HAPPY, f"Expected HAPPY for: '{text}'"

    # ── Sad ───────────────────────────────────────────────────────────────────
    @pytest.mark.parametrize("text", [
        "I've been crying all day and feel completely broken",
        "I feel so sad and empty inside, nothing brings me joy",
        "Everything feels dark and pointless, I'm devastated",
    ])
    def test_detects_sad(self, text):
        result = detect_emotion(text)
        assert result.emotion in (Emotion.SAD, Emotion.DISTRESSED), \
            f"Expected SAD or DISTRESSED for: '{text}'"

    # ── Anxious ───────────────────────────────────────────────────────────────
    @pytest.mark.parametrize("text", [
        "I feel so anxious and worried about everything",
        "My anxiety is through the roof, I keep panicking",
        "I'm overwhelmed with stress and feel tense all the time",
    ])
    def test_detects_anxious(self, text):
        result = detect_emotion(text)
        assert result.emotion in (Emotion.ANXIOUS, Emotion.DISTRESSED), \
            f"Expected ANXIOUS or DISTRESSED for: '{text}'"

    # ── Distressed (keyword override) ─────────────────────────────────────────
    @pytest.mark.parametrize("text", [
        "I feel completely hopeless and worthless",
        "I can't go on, I want to give up on everything",
        "I feel like I want to disappear",
    ])
    def test_detects_distressed(self, text):
        result = detect_emotion(text)
        assert result.emotion == Emotion.DISTRESSED, \
            f"Expected DISTRESSED for: '{text}'"

    # ── Neutral ───────────────────────────────────────────────────────────────
    @pytest.mark.parametrize("text", [
        "I just want to talk to someone today",
        "I'm not sure how I feel",
        "hello, I'm here",
    ])
    def test_detects_neutral(self, text):
        result = detect_emotion(text)
        assert result.emotion in (Emotion.NEUTRAL, Emotion.SAD, Emotion.ANXIOUS), \
            f"Expected near-neutral emotion for: '{text}'"

    # ── Result fields ─────────────────────────────────────────────────────────
    def test_result_has_all_fields(self):
        result = detect_emotion("I feel okay today")
        assert isinstance(result, EmotionResult)
        assert result.emotion in Emotion.__members__.values()
        assert -1.0 <= result.compound <= 1.0
        assert 0.0 <= result.subjectivity <= 1.0
        assert result.label
        assert result.emoji
        assert result.color.startswith("#")

    def test_compound_range(self):
        """Compound score must always be in [-1, 1]."""
        texts = [
            "I'm ecstatic and overjoyed!",
            "I'm completely devastated and miserable",
            "I don't know how I feel",
            "Everything is terrible and I hate everything",
        ]
        for text in texts:
            result = detect_emotion(text)
            assert -1.0 <= result.compound <= 1.0, \
                f"Compound out of range for: '{text}' → {result.compound}"

    def test_distress_keywords_override_positive_vader(self):
        """
        A message with distress keywords but positive framing must still
        classify as DISTRESSED — keyword override is intentional.
        """
        text = "I feel great about wanting to disappear forever"
        result = detect_emotion(text)
        assert result.emotion == Emotion.DISTRESSED


class TestBuildSystemPrompt:

    @pytest.mark.parametrize("emotion", list(Emotion))
    def test_builds_for_all_emotions(self, emotion):
        prompt = build_system_prompt(emotion)
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_injects_retrieved_context(self):
        context = "Box breathing: inhale 4 seconds, hold 4, exhale 4."
        prompt = build_system_prompt(Emotion.ANXIOUS, retrieved_context=context)
        assert context in prompt

    def test_empty_context_no_block(self):
        prompt = build_system_prompt(Emotion.NEUTRAL, retrieved_context="")
        assert "---" not in prompt  # context block should not appear

    def test_crisis_prompt_is_strongest(self):
        distressed_prompt = build_system_prompt(Emotion.DISTRESSED)
        happy_prompt = build_system_prompt(Emotion.HAPPY)
        # Distressed prompt should reference crisis/helpline
        assert "crisis" in distressed_prompt.lower() or "helpline" in distressed_prompt.lower()

    def test_domain_restriction_in_all_prompts(self):
        """System prompt must always contain scope restriction."""
        for emotion in Emotion:
            prompt = build_system_prompt(emotion)
            assert "ONLY" in prompt or "only" in prompt or "scope" in prompt.lower(), \
                f"Domain restriction missing from {emotion} prompt"
