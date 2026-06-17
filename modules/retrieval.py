"""
modules/retrieval.py — Retrieval-Augmented Generation (RAG) Engine

WHY RAG: LLMs hallucinate. For mental health — a safety-critical domain —
we cannot tolerate invented coping strategies or wrong crisis numbers.
RAG grounds responses in verified, curated knowledge by:
  1. Embedding each knowledge chunk into a dense vector space.
  2. At query time, finding the semantically nearest chunks.
  3. Injecting those chunks into the system prompt before the LLM responds.

This is the same architecture used by production systems like Bing Chat,
Perplexity, and enterprise RAG pipelines.

HOW:
  - sentence-transformers 'all-MiniLM-L6-v2' produces 384-dim embeddings
    (fast, accurate, runs on CPU without GPU).
  - FAISS IndexFlatL2 stores vectors and performs millisecond nearest-neighbour search.
  - The index is built once and cached to disk; subsequent launches load from disk.
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np

try:
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

from config import FAISS_INDEX_PATH, TOP_K_RESULTS
from data.knowledge_base import KNOWLEDGE_BASE


class RetrievalEngine:
    """
    Manages a FAISS vector index over the mental health knowledge base.

    Usage:
        engine = RetrievalEngine()
        context = engine.retrieve("I keep having panic attacks")
    """

    _INDEX_FILE = FAISS_INDEX_PATH + ".faiss"
    _META_FILE  = FAISS_INDEX_PATH + ".pkl"

    def __init__(self) -> None:
        if not _DEPS_AVAILABLE:
            print("[RAG] faiss / sentence-transformers not installed. RAG disabled.")
            self._enabled = False
            return

        self._enabled = True
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._documents: list[dict] = KNOWLEDGE_BASE
        self._index: Optional[faiss.Index] = None
        self._load_or_build_index()

    # ── Index management ──────────────────────────────────────────────────────

    def _load_or_build_index(self) -> None:
        """Load cached index from disk or build and cache a new one."""
        if os.path.exists(self._INDEX_FILE) and os.path.exists(self._META_FILE):
            print("[RAG] Loading FAISS index from cache…")
            self._index = faiss.read_index(self._INDEX_FILE)
            with open(self._META_FILE, "rb") as f:
                self._documents = pickle.load(f)
        else:
            print("[RAG] Building FAISS index…")
            self._build_index()

    def _build_index(self) -> None:
        texts = [f"{d['title']}. {d['content']}" for d in self._documents]
        embeddings = self._model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dim)
        self._index.add(embeddings)  # type: ignore[arg-type]

        # Persist to disk
        os.makedirs(os.path.dirname(self._INDEX_FILE), exist_ok=True)
        faiss.write_index(self._index, self._INDEX_FILE)
        with open(self._META_FILE, "wb") as f:
            pickle.dump(self._documents, f)
        print(f"[RAG] Index built with {len(self._documents)} documents.")

    def rebuild(self) -> None:
        """Force-rebuild the index (call when knowledge base changes)."""
        if os.path.exists(self._INDEX_FILE):
            os.remove(self._INDEX_FILE)
        if os.path.exists(self._META_FILE):
            os.remove(self._META_FILE)
        self._documents = KNOWLEDGE_BASE
        self._build_index()

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = TOP_K_RESULTS) -> str:
        """
        Return a formatted string of the top-k most relevant knowledge chunks.

        The string is injected into the LLM system prompt so the model can
        cite and expand upon evidence-based techniques.
        """
        if not self._enabled or self._index is None:
            return ""

        query_vec = self._model.encode([query], show_progress_bar=False)
        query_vec = np.array(query_vec, dtype="float32")

        distances, indices = self._index.search(query_vec, k)  # type: ignore[arg-type]

        results: list[str] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            doc = self._documents[idx]
            relevance = max(0.0, 1.0 - dist / 10.0)  # normalise to 0–1
            if relevance < 0.1:
                continue  # skip very distant results
            results.append(
                f"[{doc['category'].upper()}] {doc['title']}\n{doc['content']}"
            )

        return "\n\n".join(results)

    def retrieve_crisis_resources(self) -> str:
        """Always return crisis resources — bypasses vector search for safety."""
        for doc in self._documents:
            if doc["id"] == "crisis_resources":
                return doc["content"]
        return "Please call emergency services or a crisis line immediately."
