# data/knowledge_base.py
"""
Mental health knowledge base entries.
Each entry: {"id": str, "category": str, "title": str, "content": str}

WHY RAG: Instead of training a custom model, we inject curated,
evidence-based knowledge directly into prompts at inference time.
This keeps responses accurate, controllable, and updatable without
re-training — a critical property for a safety-sensitive domain.
"""

KNOWLEDGE_BASE = [
    # ── Grounding Techniques ──────────────────────────────────────────────────
    {
        "id": "ground_54321",
        "category": "grounding",
        "title": "5-4-3-2-1 Sensory Grounding",
        "content": (
            "The 5-4-3-2-1 technique anchors you in the present moment by engaging all five senses. "
            "Name 5 things you can see, 4 things you can physically feel (e.g., your feet on the floor), "
            "3 things you can hear, 2 things you can smell, and 1 thing you can taste. "
            "This technique interrupts anxiety spirals by redirecting attention to the physical world. "
            "Best used during panic attacks or overwhelming anxiety episodes."
        ),
    },
    {
        "id": "ground_box_breathing",
        "category": "grounding",
        "title": "Box Breathing (4-4-4-4)",
        "content": (
            "Box breathing activates the parasympathetic nervous system, reducing cortisol and heart rate. "
            "Inhale for 4 seconds, hold for 4 seconds, exhale for 4 seconds, hold for 4 seconds. "
            "Repeat 4–6 cycles. Used by military, athletes, and therapists for acute stress management. "
            "Can be practiced anywhere and shows measurable anxiety reduction in 2 minutes."
        ),
    },
    {
        "id": "ground_safe_place",
        "category": "grounding",
        "title": "Safe Place Visualisation",
        "content": (
            "Close your eyes and vividly imagine a place — real or imaginary — where you feel completely safe. "
            "Engage all senses: what do you see, hear, smell, feel? "
            "This technique is a core EMDR and trauma therapy tool. "
            "Practice for 5 minutes daily to build the neural pathway so it's accessible under stress."
        ),
    },

    # ── CBT Techniques ────────────────────────────────────────────────────────
    {
        "id": "cbt_thought_record",
        "category": "cbt",
        "title": "Cognitive Restructuring / Thought Record",
        "content": (
            "CBT thought records help identify and challenge cognitive distortions. "
            "Steps: (1) Identify the automatic negative thought (ANT). "
            "(2) Rate how much you believe it (0–100%). "
            "(3) Identify the cognitive distortion (catastrophising, all-or-nothing, mind reading, etc). "
            "(4) Generate evidence FOR and AGAINST the thought. "
            "(5) Write a balanced, realistic alternative thought. "
            "(6) Re-rate belief in the original thought. "
            "Regular practice physically rewires neural pathways over 8–12 weeks."
        ),
    },
    {
        "id": "cbt_behavioral_activation",
        "category": "cbt",
        "title": "Behavioural Activation for Depression",
        "content": (
            "Depression creates a vicious cycle: low mood → reduced activity → less positive reinforcement → lower mood. "
            "Behavioural activation breaks this by scheduling small, achievable activities even before motivation returns. "
            "Start with a 10-minute walk, calling a friend, or cooking a meal. "
            "Track mood before and after each activity to build awareness of what genuinely lifts your mood. "
            "One of the most evidence-backed interventions for mild-to-moderate depression."
        ),
    },
    {
        "id": "cbt_worry_time",
        "category": "cbt",
        "title": "Scheduled Worry Time",
        "content": (
            "Paradoxically, scheduling a fixed 20-minute 'worry time' each day reduces overall anxiety. "
            "When a worry arises outside this window, note it briefly and postpone it. "
            "During worry time, focus on problem-solving: what's within your control? What's the smallest action? "
            "This technique prevents rumination from bleeding into the whole day and restores a sense of control."
        ),
    },

    # ── Mindfulness ───────────────────────────────────────────────────────────
    {
        "id": "mindful_body_scan",
        "category": "mindfulness",
        "title": "Body Scan Meditation",
        "content": (
            "A body scan progressively relaxes each body part, releasing stored tension. "
            "Lie down comfortably, close your eyes. Bring attention to your feet — notice any sensation without judgment. "
            "Slowly move up: ankles, calves, knees, thighs, abdomen, chest, arms, neck, face. "
            "At each region, breathe into any tension and consciously release it on the exhale. "
            "10–20 minutes daily significantly reduces cortisol and improves sleep quality."
        ),
    },
    {
        "id": "mindful_urge_surfing",
        "category": "mindfulness",
        "title": "Urge Surfing",
        "content": (
            "Urge surfing treats cravings or emotional impulses like waves — they rise, peak, and fall. "
            "When an urge arises (to self-harm, use substances, react impulsively), don't act or suppress. "
            "Instead, observe the sensation: where do you feel it? How intense is it? Watch it change. "
            "Most urges peak within 20–30 minutes if not acted upon. "
            "This DBT technique builds distress tolerance without reinforcing avoidance."
        ),
    },

    # ── Self-Compassion ───────────────────────────────────────────────────────
    {
        "id": "self_compassion_break",
        "category": "self_compassion",
        "title": "Self-Compassion Break (Kristin Neff)",
        "content": (
            "When struggling, place your hand on your heart and say: "
            "(1) 'This is a moment of suffering' — mindfulness; acknowledge, don't suppress. "
            "(2) 'Suffering is part of the shared human experience' — common humanity; you're not alone. "
            "(3) 'May I be kind to myself' — self-kindness; offer yourself what you'd offer a good friend. "
            "Research by Dr. Kristin Neff shows self-compassion outperforms self-esteem for resilience "
            "because it doesn't depend on success or comparison."
        ),
    },

    # ── Sleep Hygiene ─────────────────────────────────────────────────────────
    {
        "id": "sleep_hygiene",
        "category": "sleep",
        "title": "Sleep Hygiene Essentials",
        "content": (
            "Sleep and mental health are bidirectionally linked — poor sleep worsens every mental health condition. "
            "Key practices: keep a consistent sleep/wake time (even weekends), "
            "avoid screens 1 hour before bed (blue light suppresses melatonin), "
            "keep your bedroom cool (18°C/65°F is optimal), "
            "avoid caffeine after 2pm, "
            "use your bed only for sleep and intimacy (strengthens the mental association). "
            "If anxious at bedtime, try the cognitive shuffle: imagine random, unconnected images to confuse the prefrontal cortex into sleep."
        ),
    },

    # ── Crisis Resources ──────────────────────────────────────────────────────
    {
        "id": "crisis_resources",
        "category": "crisis",
        "title": "Crisis Helplines and Emergency Resources",
        "content": (
            "If you or someone you know is in immediate danger, call emergency services (911 in US, 999 in UK, 112 in EU). "
            "Crisis helplines: "
            "🇺🇸 USA — 988 Suicide & Crisis Lifeline: call or text 988 (free, 24/7). "
            "🇬🇧 UK — Samaritans: 116 123 (free, 24/7). "
            "🇮🇳 India — iCall: 9152987821 | Vandrevala Foundation: 1860-2662-345 (24/7). "
            "🌐 International — findahelpline.com lists resources for 50+ countries. "
            "Crisis Text Line (USA/UK/Canada/Ireland): text HOME to 741741. "
            "You are not alone. Reaching out is an act of courage."
        ),
    },

    # ── Coping Strategies ─────────────────────────────────────────────────────
    {
        "id": "cope_tipp",
        "category": "coping",
        "title": "TIPP Skills (DBT)",
        "content": (
            "TIPP is a DBT crisis survival skill for rapidly reducing intense emotions. "
            "T — Temperature: hold ice cubes or splash cold water on face to activate the dive reflex, slowing heart rate. "
            "I — Intense exercise: 20 minutes of vigorous movement burns off stress hormones. "
            "P — Paced breathing: slow exhale (longer than inhale) activates the vagus nerve. "
            "P — Paired muscle relaxation: tense each muscle group for 5 seconds, then release. "
            "Use when emotions feel unbearable and you need fast physiological relief."
        ),
    },
    {
        "id": "cope_dear_man",
        "category": "coping",
        "title": "DEAR MAN (Interpersonal Effectiveness - DBT)",
        "content": (
            "DEAR MAN is a DBT script for assertively communicating needs without aggression or passivity. "
            "D — Describe the situation factually. "
            "E — Express how you feel using 'I' statements. "
            "A — Assert what you need clearly. "
            "R — Reinforce by explaining the positive outcome if your request is met. "
            "M — stay Mindful (return to your request if distracted). "
            "A — Appear confident (posture, eye contact, steady voice). "
            "N — Negotiate; be willing to give to get. "
            "Useful for setting boundaries, making requests, and navigating conflict."
        ),
    },
]
