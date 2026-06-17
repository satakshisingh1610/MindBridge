"""
modules/memory.py — Conversational Memory Management

WHY: Without memory, every message is stateless — the model cannot
reference "what you said earlier" or track how mood evolves. Short-term
memory (sliding window of N turns) keeps context within token limits.
Long-term memory (persistent JSON) enables mood tracking over sessions.

HOW:
  - ShortTermMemory  → list of {role, content} dicts, max N turns, sent to LLM
  - LongTermMemory   → JSON file on disk, stores mood trend + journal entries
  - MoodTracker      → records emotion per turn, computes weekly stats
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Deque

from config import MAX_SHORT_TERM_TURNS


# ── Short-Term Memory ─────────────────────────────────────────────────────────

@dataclass
class Turn:
    role: str      # "user" | "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    emotion: str = "neutral"


class ShortTermMemory:
    """
    Sliding-window conversation buffer.

    WHY: LLM APIs charge per token. Sending the full history grows costs
    exponentially. A window of MAX_SHORT_TERM_TURNS keeps context recent
    while bounding token spend.
    """

    def __init__(self, max_turns: int = MAX_SHORT_TERM_TURNS) -> None:
        self._buffer: Deque[Turn] = deque(maxlen=max_turns * 2)  # user+assistant pairs

    def add(self, role: str, content: str, emotion: str = "neutral") -> None:
        self._buffer.append(Turn(role=role, content=content, emotion=emotion))

    def to_messages(self) -> list[dict]:
        """Return OpenAI-compatible message list for LLM payload."""
        return [{"role": t.role, "content": t.content} for t in self._buffer]

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


# ── Mood Tracker ──────────────────────────────────────────────────────────────

@dataclass
class MoodEntry:
    timestamp: str
    emotion: str
    compound_score: float
    message_snippet: str   # first 80 chars for context


class MoodTracker:
    """
    Persists mood history and computes wellness insights.

    WHY: Showing a user "you've felt anxious 4 times this week" is
    clinically meaningful and turns a chatbot into a genuine wellness tool.
    """

    def __init__(self, storage_path: str = "data/mood_log.json") -> None:
        self.storage_path = storage_path
        self._entries: list[MoodEntry] = []
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path) as f:
                    raw = json.load(f)
                self._entries = [MoodEntry(**e) for e in raw]
            except (json.JSONDecodeError, TypeError):
                self._entries = []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump([asdict(e) for e in self._entries], f, indent=2)

    # ── Public API ────────────────────────────────────────────────────────────

    def record(self, emotion: str, compound: float, text: str) -> None:
        entry = MoodEntry(
            timestamp=datetime.utcnow().isoformat(),
            emotion=emotion,
            compound_score=round(compound, 3),
            message_snippet=text[:80],
        )
        self._entries.append(entry)
        self._save()

    def weekly_summary(self) -> dict:
        """
        Return emotion frequency + average compound score for last 7 days.
        Used to generate the 'Weekly Wellness Insight' card in the UI.
        """
        from datetime import timedelta
        cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
        recent = [e for e in self._entries if e.timestamp >= cutoff]

        if not recent:
            return {"message": "Not enough data yet. Keep chatting! 🌱", "entries": []}

        freq: dict[str, int] = {}
        total_score = 0.0
        for e in recent:
            freq[e.emotion] = freq.get(e.emotion, 0) + 1
            total_score += e.compound_score

        dominant = max(freq, key=lambda k: freq[k])
        avg_score = round(total_score / len(recent), 3)

        insight = _generate_insight(dominant, avg_score, freq)
        return {
            "message": insight,
            "dominant_emotion": dominant,
            "avg_score": avg_score,
            "frequency": freq,
            "total_entries": len(recent),
            "entries": [asdict(e) for e in recent[-20:]],  # last 20 for chart
        }

    def all_entries(self) -> list[dict]:
        return [asdict(e) for e in self._entries]


def _generate_insight(dominant: str, avg_score: float, freq: dict) -> str:
    """Craft a human-friendly wellness insight string."""
    msgs = {
        "happy": "🌟 You've had a predominantly positive week. Keep nurturing what's working!",
        "neutral": "😌 Your week looks balanced. A little mindfulness can help maintain that equilibrium.",
        "sad": "💙 It's been a tough week emotionally. Be gentle with yourself — small acts of self-care matter.",
        "anxious": "🌬️ Anxiety has been a recurring theme this week. Try to schedule even 5 minutes of deep breathing daily.",
        "distressed": "🆘 You've been through significant distress recently. Please consider reaching out to a mental health professional.",
    }
    base = msgs.get(dominant, "Keep checking in — self-awareness is a superpower.")
    return f"{base} (Avg mood score: {avg_score:+.2f})"


# ── Journal ───────────────────────────────────────────────────────────────────

class Journal:
    """
    Simple append-only journal for text entries.

    WHY: Journaling is a first-line CBT intervention. Providing it in-app
    removes friction and keeps users engaged with their mental health.
    """

    def __init__(self, storage_path: str = "data/journal.json") -> None:
        self.storage_path = storage_path
        self._entries: list[dict] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path) as f:
                    self._entries = json.load(f)
            except (json.JSONDecodeError, TypeError):
                self._entries = []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(self._entries, f, indent=2)

    def add_entry(self, text: str, emotion: str = "neutral") -> dict:
        entry = {
            "id": len(self._entries) + 1,
            "timestamp": datetime.utcnow().isoformat(),
            "text": text,
            "emotion": emotion,
        }
        self._entries.append(entry)
        self._save()
        return entry

    def get_entries(self, limit: int = 10) -> list[dict]:
        return self._entries[-limit:][::-1]  # most recent first
