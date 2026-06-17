"""
tests/test_llm_client.py — Unit Tests for LLM API Client

Uses unittest.mock to avoid real API calls — tests retry logic,
error handling, and message format without spending tokens.

Run:  pytest tests/test_llm_client.py -v
"""

import pytest
from unittest.mock import patch, MagicMock, call
from modules.llm_client import LLMClient, LLMError


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mock_response(status_code: int, content: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    resp.text = content
    return resp


# ══════════════════════════════════════════════════════════════════════════════
# Happy path
# ══════════════════════════════════════════════════════════════════════════════

class TestLLMClientSuccess:

    @patch("modules.llm_client.requests.post")
    def test_successful_chat(self, mock_post):
        mock_post.return_value = _mock_response(200, "I'm here to help you.")
        client = LLMClient(api_key="test_key")
        result = client.chat(messages=[{"role": "user", "content": "Hello"}])
        assert result == "I'm here to help you."
        mock_post.assert_called_once()

    @patch("modules.llm_client.requests.post")
    def test_system_prompt_prepended(self, mock_post):
        mock_post.return_value = _mock_response(200, "response")
        client = LLMClient(api_key="test_key")
        client.chat(
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="You are MindBridge."
        )
        payload = mock_post.call_args[1]["json"]
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "You are MindBridge."

    @patch("modules.llm_client.requests.post")
    def test_respects_max_tokens(self, mock_post):
        mock_post.return_value = _mock_response(200, "ok")
        client = LLMClient(api_key="test_key")
        client.chat(messages=[], max_tokens=500)
        payload = mock_post.call_args[1]["json"]
        assert payload["max_tokens"] == 500

    @patch("modules.llm_client.requests.post")
    def test_respects_temperature(self, mock_post):
        mock_post.return_value = _mock_response(200, "ok")
        client = LLMClient(api_key="test_key")
        client.chat(messages=[], temperature=0.0)
        payload = mock_post.call_args[1]["json"]
        assert payload["temperature"] == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Retry logic
# ══════════════════════════════════════════════════════════════════════════════

class TestRetryLogic:

    @patch("modules.llm_client.time.sleep")
    @patch("modules.llm_client.requests.post")
    def test_retries_on_429(self, mock_post, mock_sleep):
        """Should retry on rate limit, then succeed."""
        mock_post.side_effect = [
            _mock_response(429, "rate limited"),
            _mock_response(429, "rate limited"),
            _mock_response(200, "success after retry"),
        ]
        client = LLMClient(api_key="test_key")
        result = client.chat(messages=[{"role": "user", "content": "hi"}])
        assert result == "success after retry"
        assert mock_post.call_count == 3
        assert mock_sleep.call_count == 2  # slept between retries

    @patch("modules.llm_client.time.sleep")
    @patch("modules.llm_client.requests.post")
    def test_retries_on_503(self, mock_post, mock_sleep):
        mock_post.side_effect = [
            _mock_response(503, "service unavailable"),
            _mock_response(200, "recovered"),
        ]
        client = LLMClient(api_key="test_key")
        result = client.chat(messages=[])
        assert result == "recovered"

    @patch("modules.llm_client.time.sleep")
    @patch("modules.llm_client.requests.post")
    def test_raises_after_max_retries(self, mock_post, mock_sleep):
        """Should raise LLMError after all retries exhausted."""
        mock_post.return_value = _mock_response(429, "always rate limited")
        client = LLMClient(api_key="test_key")
        with pytest.raises(LLMError):
            client.chat(messages=[])

    @patch("modules.llm_client.requests.post")
    def test_raises_immediately_on_400(self, mock_post):
        """400 Bad Request is not retryable."""
        mock_post.return_value = _mock_response(400, "bad request")
        client = LLMClient(api_key="test_key")
        with pytest.raises(LLMError, match="400"):
            client.chat(messages=[])

    @patch("modules.llm_client.requests.post")
    def test_raises_on_connection_error(self, mock_post):
        import requests as req
        mock_post.side_effect = req.exceptions.ConnectionError("no connection")
        client = LLMClient(api_key="test_key")
        with pytest.raises(LLMError, match="connect"):
            client.chat(messages=[])


# ══════════════════════════════════════════════════════════════════════════════
# Init validation
# ══════════════════════════════════════════════════════════════════════════════

class TestInitValidation:

    def test_raises_without_api_key(self):
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            LLMClient(api_key="")

    def test_auth_header_set(self):
        client = LLMClient(api_key="sk_test_123")
        assert client._headers["Authorization"] == "Bearer sk_test_123"
