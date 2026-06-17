"""
modules/llm_client.py — Groq / OpenAI-compatible LLM API Client

WHY: Isolating the API call into its own module means:
  - Swapping LLM providers (Groq → OpenAI → local Ollama) changes ONE file.
  - Retry logic, rate-limit handling, and error normalisation live here only.
  - The rest of the codebase never touches raw HTTP.

RETRY STRATEGY: Exponential backoff with jitter.
  Attempt 1 → immediate
  Attempt 2 → wait ~1.5s
  Attempt 3 → wait ~2.3s
  This handles transient 429 (rate limit) and 503 (service unavailable) errors
  without hammering the API.
"""

from __future__ import annotations

import time
import random
from typing import Optional

import requests

from config import (
    GROQ_API_KEY,
    GROQ_API_URL,
    GROQ_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    MAX_RETRIES,
    RETRY_BACKOFF,
)


class LLMError(Exception):
    """Raised when the LLM API fails after all retries."""
    pass


class LLMClient:
    """
    Thin wrapper around the Groq chat completions endpoint.
    Supports full conversation history (multi-turn).
    """

    def __init__(
        self,
        api_key: str = GROQ_API_KEY,
        api_url: str = GROQ_API_URL,
        model: str = GROQ_MODEL,
    ) -> None:
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file or "
                "export it as an environment variable."
            )
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._api_url = api_url
        self._model = model

    def chat(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
    ) -> str:
        """
        Send a list of messages to the LLM and return the assistant reply.

        Args:
            messages: OpenAI-compatible list of {"role": ..., "content": ...}
            system_prompt: Injected as the first "system" message if provided.
            max_tokens: Upper bound on response length.
            temperature: Sampling temperature (0 = deterministic, 1 = creative).

        Returns:
            The assistant's reply as a plain string.

        Raises:
            LLMError: If all retries are exhausted.
        """
        full_messages: list[dict] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload = {
            "model": self._model,
            "messages": full_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        last_error: Optional[Exception] = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(
                    self._api_url,
                    headers=self._headers,
                    json=payload,
                    timeout=30,
                )

                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]

                # Retryable errors
                if response.status_code in (429, 500, 502, 503, 504):
                    wait = RETRY_BACKOFF ** attempt + random.uniform(0, 0.5)
                    print(f"[LLM] HTTP {response.status_code} on attempt {attempt}. Retrying in {wait:.1f}s…")
                    time.sleep(wait)
                    last_error = LLMError(f"HTTP {response.status_code}: {response.text[:200]}")
                    continue

                # Non-retryable error
                raise LLMError(f"HTTP {response.status_code}: {response.text[:200]}")

            except requests.exceptions.Timeout:
                wait = RETRY_BACKOFF ** attempt
                print(f"[LLM] Timeout on attempt {attempt}. Retrying in {wait:.1f}s…")
                time.sleep(wait)
                last_error = LLMError("Request timed out.")

            except requests.exceptions.ConnectionError as e:
                raise LLMError(f"Could not connect to LLM API: {e}") from e

        raise LLMError(f"LLM API failed after {MAX_RETRIES} attempts. Last error: {last_error}")
