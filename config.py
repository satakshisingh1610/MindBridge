"""
config.py — Central configuration using environment variables.

WHY: Never hardcode secrets. .env files + os.getenv() keep keys out of
version control and make deployment to any cloud provider trivial.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file if present

# ── LLM ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", 1000))
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.7))

# ── Memory ────────────────────────────────────────────────────────────────────
MAX_SHORT_TERM_TURNS: int = int(os.getenv("MAX_SHORT_TERM_TURNS", 10))

# ── RAG ───────────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", 3))

# ── Translation ───────────────────────────────────────────────────────────────
LIBRETRANSLATE_URL: str = os.getenv("LIBRETRANSLATE_URL", "https://libretranslate.com")

# ── Retry ─────────────────────────────────────────────────────────────────────
MAX_RETRIES: int = 3
RETRY_BACKOFF: float = 1.5  # seconds, exponential
