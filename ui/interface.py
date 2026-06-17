from __future__ import annotations

import gradio as gr
from modules.orchestrator import Orchestrator


# ==========================================================
# CUSTOM CSS
# ==========================================================

CUSTOM_CSS = """
.gradio-container {
    background: linear-gradient(135deg, #020617, #0f172a) !important;
    color: #e2e8f0 !important;
    font-family: Inter, sans-serif;
}

#mb-header {
    text-align: center;
    padding: 20px;
}

#mb-header h1 {
    font-size: 38px;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sidebar-box {
    background: rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 10px;
}

#emotion-badge {
    padding: 10px;
    border-radius: 12px;
    text-align: center;
    font-weight: bold;
    color: white;
}

.gradio-chatbot {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 16px !important;
}

textarea {
    background: rgba(255,255,255,0.05) !important;
    color: white !important;
    border-radius: 12px !important;
}

button {
    background: linear-gradient(
        135deg,
        #6366f1,
        #8b5cf6
    ) !important;

    color: white !important;
    border: none !important;
    border-radius: 10px !important;
}

button:hover {
    transform: scale(1.03);
}
"""


# ==========================================================
# HEADER
# ==========================================================

HEADER_HTML = """
<div id="mb-header">
    <h1>🧠 MindBridge</h1>
    <p>
        Multilingual AI Mental Health Assistant
        <br>
        Emotion Analysis • Crisis Detection • RAG Support
    </p>
</div>
"""


DISCLAIMER_HTML = """
<div style="text-align:center;color:#facc15;">
⚠️ MindBridge provides emotional support and guidance,
but is not a substitute for professional mental health care.
</div>
"""


# ==========================================================
# EMOTION BADGE
# ==========================================================

def make_badge(label: str):

    label_lower = label.lower()

    if "sad" in label_lower:
        color = "#3b82f6"

    elif "happy" in label_lower:
        color = "#22c55e"

    elif "angry" in label_lower:
        color = "#ef4444"

    elif "anxious" in label_lower:
        color = "#f59e0b"

    else:
        color = "#818cf8"

    return (
        f"<div id='emotion-badge' "
        f"style='background:{color}'>"
        f"{label}"
        f"</div>"
    )


# ==========================================================
# INTERFACE
# ==========================================================

def create_interface():

    def get_orchestrator(state):

        if "orch" not in state:
            state["orch"] = Orchestrator()

        return state["orch"], state

    def chat_fn(user_msg, history, state):

        if not user_msg.strip():
            return history, "", "", state

        orch, state = get_orchestrator(state)

        response = orch.chat(user_msg)

        messages = []

        for user_text, bot_text in response.history:

            messages.append(
                {
                    "role": "user",
                    "content": str(user_text)
                }
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": str(bot_text)
                }
            )

        emotion_label = response.metadata.emotion.label

        emotion_html = make_badge(
            emotion_label
        )

        language_html = (
            "<div class='sidebar-box'>"
            f"🌍 Language: "
            f"{response.metadata.language_name}"
            "</div>"
        )

        return (
            messages,
            emotion_html,
            language_html,
            state
        )

    def reset_fn(state):

        if "orch" in state:
            state["orch"].reset_session()

        return (
            [],
            make_badge("Neutral 😐"),
            "<div class='sidebar-box'>🌍 Language: English</div>",
            state
        )

    with gr.Blocks(
        title="MindBridge",
        css=CUSTOM_CSS
    ) as demo:

        state = gr.State({})

        gr.HTML(HEADER_HTML)

        gr.HTML(DISCLAIMER_HTML)

        with gr.Row():

            # ==================================
            # SIDEBAR
            # ==================================

            with gr.Column(scale=1):

                emotion = gr.HTML(
                    make_badge("Neutral 😐")
                )

                language = gr.HTML(
                    "<div class='sidebar-box'>"
                    "🌍 Language: English"
                    "</div>"
                )

                reset_btn = gr.Button(
                    "🔄 Reset Session"
                )

            # ==================================
            # CHAT
            # ==================================

            with gr.Column(scale=4):

                chatbot = gr.Chatbot(
                    value=[],
                    height=500
                )

                msg = gr.Textbox(
                    placeholder=(
                        "How are you feeling today?"
                    ),
                    label=""
                )

                send_btn = gr.Button(
                    "Send ✨"
                )

        send_btn.click(
            fn=chat_fn,
            inputs=[
                msg,
                chatbot,
                state
            ],
            outputs=[
                chatbot,
                emotion,
                language,
                state
            ]
        ).then(
            lambda: "",
            outputs=[msg]
        )

        msg.submit(
            fn=chat_fn,
            inputs=[
                msg,
                chatbot,
                state
            ],
            outputs=[
                chatbot,
                emotion,
                language,
                state
            ]
        ).then(
            lambda: "",
            outputs=[msg]
        )

        reset_btn.click(
            fn=reset_fn,
            inputs=[state],
            outputs=[
                chatbot,
                emotion,
                language,
                state
            ]
        )

    return demo