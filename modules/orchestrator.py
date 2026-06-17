from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from modules.sentiment import detect_emotion, build_system_prompt, EmotionResult
from modules.memory import ShortTermMemory, MoodTracker, Journal
from modules.retrieval import RetrievalEngine
from modules.crisis import apply_crisis_guardrails, assess_crisis
from modules.language import detect_and_translate_to_english, translate_to_language
from modules.llm_client import LLMClient, LLMError
from modules.domain_guard import DomainGuard


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class MessageMetadata:
    emotion: EmotionResult
    language_code: str
    language_name: str
    was_translated: bool
    crisis_level: int
    was_overridden: bool
    retrieved_chunks: int
    domain_blocked: bool = False
    domain_layer: int = 0
    error: Optional[str] = None


@dataclass
class OrchestratorResponse:
    reply: str
    metadata: MessageMetadata
    history: list[tuple[str, str]]


# ─────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────

class Orchestrator:

    def __init__(self) -> None:
        self._memory = ShortTermMemory()
        self._mood = MoodTracker()
        self._journal = Journal()
        self._rag = RetrievalEngine()
        self._llm = LLMClient()
        self._domain = DomainGuard(llm_client=self._llm, use_llm_classifier=True)
        self._history: list[tuple[str, str]] = []

    def chat(self, user_message: str) -> OrchestratorResponse:

        # ── STEP 0: LANGUAGE ──
        lang_result = detect_and_translate_to_english(user_message)

        print(
            f"[LANGUAGE] "
            f"{lang_result.detected_name} "
            f"({lang_result.detected_code})"
        )

        english_input = lang_result.translated_to_english

        # ── STEP 1: DOMAIN ──
        domain_check = self._domain.check(english_input)

        if domain_check.is_blocked:
            from modules.sentiment import EmotionResult, Emotion

            neutral_emotion = EmotionResult(
                emotion=Emotion.NEUTRAL,
                compound=0.0,
                subjectivity=0.0,
                label="Neutral 😐",
                emoji="😐",
                color="#9E9E9E",
            )

            redirect = domain_check.redirect_response

            if not lang_result.is_english:
                try:
                    redirect = translate_to_language(
                        redirect,
                        lang_result.detected_code
                    )
                except Exception:
                    pass

            self._history.append((user_message, redirect))

            metadata = MessageMetadata(
                emotion=neutral_emotion,
                language_code=lang_result.detected_code,
                language_name=lang_result.detected_name,
                was_translated=not lang_result.is_english,
                crisis_level=0,
                was_overridden=False,
                retrieved_chunks=0,
                domain_blocked=True,
                domain_layer=domain_check.layer,
            )

            return OrchestratorResponse(
                redirect,
                metadata,
                self._history.copy()
            )

        # ── STEP 2: EMOTION ──
        emotion = detect_emotion(english_input)

        # ── STEP 3: CRISIS ──
        crisis = assess_crisis(english_input)

        # ── STEP 4: RAG ──
        if crisis.level == 2:
            retrieved_context = self._rag.retrieve_crisis_resources()
        else:
            retrieved_context = self._rag.retrieve(english_input)

        # ── STEP 5: PROMPT ──
        system_prompt = build_system_prompt(
            emotion.emotion,
            retrieved_context
        )

        # ── STEP 6: MEMORY ──
        self._memory.add(
            "user",
            english_input,
            emotion=emotion.emotion.value
        )

        # ── STEP 7: LLM ──
        llm_reply = ""
        error_msg = None

        if crisis.level < 2:
            try:
                llm_reply = self._llm.chat(
                    messages=self._memory.to_messages(),
                    system_prompt=system_prompt,
                )
            except LLMError as e:
                error_msg = str(e)
                llm_reply = (
                    "I'm having a small technical issue, "
                    "but I'm here for you 💙"
                )

        # ── STEP 8: GUARDRAILS ──
        final_reply, was_overridden = apply_crisis_guardrails(
            english_input,
            llm_reply
        )

        if crisis.level == 2:
            final_reply = crisis.safe_response
            was_overridden = True

        # ── STEP 9: MEMORY ──
        self._memory.add(
            "assistant",
            final_reply
        )

        # ── STEP 10: MOOD ──
        self._mood.record(
            emotion.emotion.value,
            emotion.compound,
            english_input
        )

        # ── STEP 11: TRANSLATION ──
        display_reply = final_reply

        should_translate = (
            not lang_result.is_english
        )

        if should_translate:
            try:
                display_reply = translate_to_language(
                    final_reply,
                    lang_result.detected_code
                )
            except Exception:
                display_reply = final_reply

        # ── STEP 12: HISTORY ──
        self._history.append(
            (user_message, display_reply)
        )

        metadata = MessageMetadata(
            emotion=emotion,
            language_code=lang_result.detected_code,
            language_name=lang_result.detected_name,
            was_translated=not lang_result.is_english,
            crisis_level=crisis.level,
            was_overridden=was_overridden,
            retrieved_chunks=(
                len(retrieved_context.split("\n\n"))
                if retrieved_context
                else 0
            ),
            error=error_msg,
        )

        return OrchestratorResponse(
            display_reply,
            metadata,
            self._history.copy()
        )

    def reset_session(self):
        self._memory.clear()
        self._history.clear()