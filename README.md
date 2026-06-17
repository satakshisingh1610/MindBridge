# 🧠 MindBridge — Production AI Mental Health Companion

> An intelligent, empathetic AI companion with emotion awareness, crisis detection, RAG, and multilingual support.

---

## 🏗️ Architecture Overview

```
mindbridge/
├── app.py                    # Entry point
├── config.py                 # Centralised config via env vars
├── requirements.txt
├── .env.example
│
├── modules/
│   ├── domain_guard.py       # ★ Domain restriction & guardrails (3-layer)
│   ├── sentiment.py          # Emotion detection (VADER + TextBlob)
│   ├── memory.py             # Short-term memory, mood tracker, journal
│   ├── retrieval.py          # RAG engine (FAISS + sentence-transformers)
│   ├── crisis.py             # Crisis detection & ethical safeguards
│   ├── language.py           # Multilingual support
│   ├── llm_client.py         # Groq API client with retry logic
│   └── orchestrator.py       # Pipeline coordinator (Step 0 = domain guard)
│
├── data/
│   └── knowledge_base.py     # Curated mental health knowledge (CBT, DBT, grounding)
│
└── ui/
    └── interface.py          # Gradio Blocks UI (chat, journal, insights, about)
```

---

