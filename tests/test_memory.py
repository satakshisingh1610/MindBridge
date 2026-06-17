"""
tests/test_memory.py — Unit Tests for Memory Management

Run:  pytest tests/test_memory.py -v
"""

import json
import os
import tempfile
import pytest
from modules.memory import ShortTermMemory, MoodTracker, Journal, Turn


class TestShortTermMemory:

    def test_add_and_retrieve(self):
        mem = ShortTermMemory(max_turns=5)
        mem.add("user", "Hello")
        mem.add("assistant", "Hi there!")
        messages = mem.to_messages()
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi there!"}

    def test_sliding_window_evicts_oldest(self):
        mem = ShortTermMemory(max_turns=2)  # max 4 messages (2 turns × 2)
        for i in range(5):
            mem.add("user", f"message {i}")
            mem.add("assistant", f"reply {i}")

        messages = mem.to_messages()
        # Should only contain last 2 turns (4 messages)
        assert len(messages) <= 4
        # Most recent messages should be present
        contents = [m["content"] for m in messages]
        assert "message 4" in contents
        assert "reply 4" in contents
        # Oldest should be evicted
        assert "message 0" not in contents

    def test_clear_empties_buffer(self):
        mem = ShortTermMemory()
        mem.add("user", "test")
        mem.add("assistant", "response")
        mem.clear()
        assert len(mem) == 0
        assert mem.to_messages() == []

    def test_to_messages_format(self):
        """Output must be OpenAI-compatible message list."""
        mem = ShortTermMemory()
        mem.add("user", "I feel sad", emotion="sad")
        msgs = mem.to_messages()
        assert len(msgs) == 1
        msg = msgs[0]
        assert "role" in msg
        assert "content" in msg
        assert msg["role"] == "user"
        assert msg["content"] == "I feel sad"
        # emotion should NOT be in the message dict (not part of API spec)
        assert "emotion" not in msg

    def test_len(self):
        mem = ShortTermMemory()
        assert len(mem) == 0
        mem.add("user", "hi")
        assert len(mem) == 1
        mem.add("assistant", "hello")
        assert len(mem) == 2


class TestMoodTracker:

    @pytest.fixture
    def tracker(self, tmp_path):
        path = str(tmp_path / "mood_log.json")
        return MoodTracker(storage_path=path)

    def test_record_and_persist(self, tracker):
        tracker.record("anxious", -0.45, "I feel really anxious today")
        entries = tracker.all_entries()
        assert len(entries) == 1
        assert entries[0]["emotion"] == "anxious"
        assert entries[0]["compound_score"] == -0.45

    def test_multiple_records(self, tracker):
        tracker.record("happy", 0.8, "I feel great!")
        tracker.record("sad", -0.6, "I feel down")
        tracker.record("anxious", -0.3, "I'm worried")
        entries = tracker.all_entries()
        assert len(entries) == 3

    def test_weekly_summary_empty(self, tracker):
        summary = tracker.weekly_summary()
        assert "message" in summary
        assert "Not enough data" in summary["message"]

    def test_weekly_summary_with_data(self, tracker):
        tracker.record("happy", 0.8, "great")
        tracker.record("happy", 0.7, "wonderful")
        tracker.record("anxious", -0.4, "worried")
        summary = tracker.weekly_summary()
        assert "dominant_emotion" in summary
        assert summary["dominant_emotion"] == "happy"
        assert "frequency" in summary
        assert summary["frequency"]["happy"] == 2
        assert "avg_score" in summary

    def test_snippet_truncated_to_80(self, tracker):
        long_text = "x" * 200
        tracker.record("neutral", 0.0, long_text)
        entries = tracker.all_entries()
        assert len(entries[0]["message_snippet"]) <= 80

    def test_persistence_across_instances(self, tmp_path):
        path = str(tmp_path / "mood.json")
        t1 = MoodTracker(storage_path=path)
        t1.record("happy", 0.7, "feeling good")

        t2 = MoodTracker(storage_path=path)
        entries = t2.all_entries()
        assert len(entries) == 1
        assert entries[0]["emotion"] == "happy"


class TestJournal:

    @pytest.fixture
    def journal(self, tmp_path):
        path = str(tmp_path / "journal.json")
        return Journal(storage_path=path)

    def test_add_entry(self, journal):
        entry = journal.add_entry("Today I felt anxious about work", "anxious")
        assert entry["text"] == "Today I felt anxious about work"
        assert entry["emotion"] == "anxious"
        assert "timestamp" in entry
        assert entry["id"] == 1

    def test_sequential_ids(self, journal):
        e1 = journal.add_entry("First entry")
        e2 = journal.add_entry("Second entry")
        assert e2["id"] == e1["id"] + 1

    def test_get_entries_most_recent_first(self, journal):
        journal.add_entry("Entry 1")
        journal.add_entry("Entry 2")
        journal.add_entry("Entry 3")
        entries = journal.get_entries(limit=3)
        assert entries[0]["text"] == "Entry 3"
        assert entries[1]["text"] == "Entry 2"
        assert entries[2]["text"] == "Entry 1"

    def test_get_entries_respects_limit(self, journal):
        for i in range(10):
            journal.add_entry(f"Entry {i}")
        entries = journal.get_entries(limit=3)
        assert len(entries) == 3

    def test_persistence_across_instances(self, tmp_path):
        path = str(tmp_path / "journal.json")
        j1 = Journal(storage_path=path)
        j1.add_entry("I feel much better today", "happy")

        j2 = Journal(storage_path=path)
        entries = j2.get_entries()
        assert len(entries) == 1
        assert entries[0]["text"] == "I feel much better today"

    def test_corrupted_file_recovers(self, tmp_path):
        path = str(tmp_path / "journal.json")
        with open(path, "w") as f:
            f.write("THIS IS NOT JSON {{{")
        # Should not raise, should start fresh
        journal = Journal(storage_path=path)
        assert journal.get_entries() == []
