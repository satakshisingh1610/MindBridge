"""
tests/test_orchestrator_integration.py — Integration Tests for Pipeline

These tests mock the LLM and external services but run the full
orchestrator pipeline to catch integration bugs between modules.

Run:  pytest tests/test_orchestrator_integration.py -v
"""

import pytest
from unittest.mock import patch, MagicMock
from modules.orchestrator import Orchestrator, OrchestratorResponse


def _make_orchestrator(llm_response: str = "I'm here to support you."):
    """
    Create an Orchestrator with mocked LLM + RAG to avoid API calls.
    """
    orch = Orchestrator.__new__(Orchestrator)

    # Real modules (no network needed)
    from modules.memory import ShortTermMemory, MoodTracker, Journal
    import tempfile, os
    tmp = tempfile.mkdtemp()
    orch._memory  = ShortTermMemory()
    orch._mood    = MoodTracker(storage_path=os.path.join(tmp, "mood.json"))
    orch._journal = Journal(storage_path=os.path.join(tmp, "journal.json"))
    orch._history = []

    # Mock LLM
    mock_llm = MagicMock()
    mock_llm.chat.return_value = llm_response
    orch._llm = mock_llm

    # Mock RAG (lightweight — avoids loading sentence-transformers)
    mock_rag = MagicMock()
    mock_rag.retrieve.return_value = "Box breathing: inhale 4s, hold 4s, exhale 4s."
    mock_rag.retrieve_crisis_resources.return_value = "Call 988 for crisis support."
    orch._rag = mock_rag

    # Real domain guard (no LLM classifier)
    from modules.domain_guard import DomainGuard
    orch._domain = DomainGuard(llm_client=None, use_llm_classifier=False)

    return orch


# ══════════════════════════════════════════════════════════════════════════════
# Normal flow
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalFlow:

    def test_returns_orchestrator_response(self):
        orch = _make_orchestrator()
        result = orch.chat("I've been feeling really anxious lately")
        assert isinstance(result, OrchestratorResponse)
        assert result.reply
        assert result.metadata
        assert isinstance(result.history, list)

    def test_reply_is_string(self):
        orch = _make_orchestrator("Here is my empathetic response.")
        result = orch.chat("I feel overwhelmed by stress")
        assert isinstance(result.reply, str)
        assert len(result.reply) > 0

    def test_history_accumulates(self):
        orch = _make_orchestrator()
        orch.chat("I feel anxious")
        orch.chat("I'm also struggling with sleep")
        result = orch.chat("Everything feels too much")
        assert len(result.history) == 3

    def test_memory_fed_to_llm(self):
        orch = _make_orchestrator()
        orch.chat("I feel anxious today")
        orch.chat("The anxiety is getting worse")
        # LLM should have been called with conversation history
        call_args = orch._llm.chat.call_args
        messages = call_args[1]["messages"] if call_args[1] else call_args[0][0]
        # Should contain prior messages
        assert len(messages) >= 2

    def test_metadata_populated(self):
        orch = _make_orchestrator()
        result = orch.chat("I've been feeling sad and lonely")
        meta = result.metadata
        assert meta.emotion is not None
        assert meta.language_code
        assert isinstance(meta.crisis_level, int)
        assert isinstance(meta.domain_blocked, bool)
        assert isinstance(meta.was_overridden, bool)


# ══════════════════════════════════════════════════════════════════════════════
# Domain guard integration
# ══════════════════════════════════════════════════════════════════════════════

class TestDomainGuardIntegration:

    def test_off_topic_does_not_call_llm(self):
        orch = _make_orchestrator()
        orch.chat("write me a Python function to sort a list")
        # LLM should NOT have been called for off-topic
        orch._llm.chat.assert_not_called()

    def test_off_topic_returns_redirect(self):
        orch = _make_orchestrator()
        result = orch.chat("solve this math equation: x^2 + 4 = 0")
        assert result.metadata.domain_blocked
        assert "?" in result.reply  # redirect should ask an open question
        assert len(result.reply) > 50

    def test_off_topic_appears_in_history(self):
        orch = _make_orchestrator()
        result = orch.chat("how do I fix a bug in my code")
        assert len(result.history) == 1
        assert result.history[0][0] == "how do I fix a bug in my code"

    def test_mental_health_topic_calls_llm(self):
        orch = _make_orchestrator()
        orch.chat("I feel so depressed and hopeless")
        orch._llm.chat.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# Crisis integration
# ══════════════════════════════════════════════════════════════════════════════

class TestCrisisIntegration:

    def test_crisis_overrides_llm(self):
        orch = _make_orchestrator(llm_response="Have you tried journaling?")
        result = orch.chat("I want to kill myself")
        assert result.metadata.was_overridden or result.metadata.crisis_level == 2
        # LLM response should not appear
        assert "journaling" not in result.reply.lower()

    def test_crisis_reply_contains_helplines(self):
        orch = _make_orchestrator()
        result = orch.chat("I don't want to live anymore")
        assert any(n in result.reply for n in ["988", "116", "9152987821"])

    def test_crisis_level_in_metadata(self):
        orch = _make_orchestrator()
        result = orch.chat("I want to end my life")
        assert result.metadata.crisis_level == 2


# ══════════════════════════════════════════════════════════════════════════════
# Emotion detection integration
# ══════════════════════════════════════════════════════════════════════════════

class TestEmotionIntegration:

    def test_emotion_detected_and_stored(self):
        orch = _make_orchestrator()
        result = orch.chat("I feel absolutely wonderful and joyful today!")
        from modules.sentiment import Emotion
        assert result.metadata.emotion.emotion == Emotion.HAPPY

    def test_anxious_triggers_rag(self):
        orch = _make_orchestrator()
        orch.chat("I have terrible panic attacks and anxiety")
        orch._rag.retrieve.assert_called()

    def test_emotion_influences_system_prompt(self):
        orch = _make_orchestrator()
        orch.chat("I am devastated, hopeless and cannot go on")
        call_kwargs = orch._llm.chat.call_args
        if call_kwargs:
            # Either kwarg or positional
            kw = call_kwargs[1] if call_kwargs[1] else {}
            system_prompt = kw.get("system_prompt", "")
            assert len(system_prompt) > 100  # should be a rich dynamic prompt


# ══════════════════════════════════════════════════════════════════════════════
# Session reset
# ══════════════════════════════════════════════════════════════════════════════

class TestSessionReset:

    def test_reset_clears_history(self):
        orch = _make_orchestrator()
        orch.chat("I feel anxious")
        orch.chat("Still feeling anxious")
        orch.reset_session()
        result = orch.chat("Hello again")
        assert len(result.history) == 1  # fresh session

    def test_reset_clears_memory(self):
        orch = _make_orchestrator()
        orch.chat("First message")
        orch.reset_session()
        assert len(orch._memory) == 0
