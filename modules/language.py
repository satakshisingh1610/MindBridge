"""
modules/language.py — Multilingual Support (PRODUCTION IMPROVED)

Improvements:
✔ Better short-text detection
✔ Script-aware language detection
✔ More supported languages
✔ Safe fallback logic
✔ Translation guardrails
✔ MindBridge-compatible
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from langdetect import detect as _langdetect
    from langdetect.lang_detect_exception import LangDetectException
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator
    _TRANSLATOR_AVAILABLE = True
except ImportError:
    _TRANSLATOR_AVAILABLE = False


LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",

    # New languages
    "ja": "Japanese",
    "ko": "Korean",
    "zh-cn": "Chinese",
    "tr": "Turkish",
    "nl": "Dutch",
    "pl": "Polish",
    "uk": "Ukrainian",
}


@dataclass
class LanguageResult:
    detected_code: str
    detected_name: str
    is_english: bool
    translated_to_english: str


def _contains_non_latin(text: str) -> bool:
    """
    Detect supported non-Latin scripts.
    """

    for ch in text:
        code = ord(ch)

        # Devanagari (Hindi)
        if 0x0900 <= code <= 0x097F:
            return True

        # Arabic
        if 0x0600 <= code <= 0x06FF:
            return True

        # Bengali
        if 0x0980 <= code <= 0x09FF:
            return True

        # Tamil
        if 0x0B80 <= code <= 0x0BFF:
            return True

        # Telugu
        if 0x0C00 <= code <= 0x0C7F:
            return True

        # Japanese Hiragana/Katakana
        if 0x3040 <= code <= 0x30FF:
            return True

        # Chinese
        if 0x4E00 <= code <= 0x9FFF:
            return True

        # Korean
        if 0xAC00 <= code <= 0xD7AF:
            return True

    return False


def detect_and_translate_to_english(text: str) -> LanguageResult:

    text_clean = text.strip()

    if not text_clean:
        return LanguageResult(
            "en",
            "English",
            True,
            text,
        )

    # --------------------------------------------------
    # RULE 1
    # Short ASCII text is almost always English.
    # Prevents random language detection.
    # --------------------------------------------------
    if (
        len(text_clean) <= 15
        and not _contains_non_latin(text_clean)
    ):
        return LanguageResult(
            "en",
            "English",
            True,
            text,
        )

    # --------------------------------------------------
    # RULE 2
    # Pure ASCII longer text → English
    # --------------------------------------------------
    if all(ord(c) < 128 for c in text_clean):
        return LanguageResult(
            "en",
            "English",
            True,
            text,
        )

    # --------------------------------------------------
    # RULE 3
    # Langdetect unavailable
    # --------------------------------------------------
    if not _LANGDETECT_AVAILABLE:
        return LanguageResult(
            "en",
            "English",
            True,
            text,
        )

    try:
        code = _langdetect(text_clean)
    except (LangDetectException, Exception):
        code = "en"

    # --------------------------------------------------
    # RULE 4
    # Unsupported language → English fallback
    # --------------------------------------------------
    if code not in LANGUAGE_NAMES:
        code = "en"

    name = LANGUAGE_NAMES.get(code, "English")

    # --------------------------------------------------
    # RULE 5
    # Skip translation for English
    # --------------------------------------------------
    if code == "en":
        return LanguageResult(
            "en",
            "English",
            True,
            text,
        )

    if not _TRANSLATOR_AVAILABLE:
        return LanguageResult(
            code,
            name,
            False,
            text,
        )

    # --------------------------------------------------
    # Translation
    # --------------------------------------------------
    try:
        translated = GoogleTranslator(
            source="auto",
            target="en",
        ).translate(text)

        return LanguageResult(
            code,
            name,
            False,
            translated or text,
        )

    except Exception:
        return LanguageResult(
            "en",
            "English",
            True,
            text,
        )


def translate_to_language(
    text: str,
    target_code: str,
) -> str:

    if target_code not in LANGUAGE_NAMES:
        return text

    if target_code == "en":
        return text

    if not _TRANSLATOR_AVAILABLE:
        return text

    try:
        translated = GoogleTranslator(
            source="en",
            target=target_code,
        ).translate(text)

        return translated or text

    except Exception:
        return text