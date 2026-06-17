import os
import gradio as gr
from fastapi import FastAPI
from ui.interface import create_interface

demo = create_interface()

app = FastAPI()

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)