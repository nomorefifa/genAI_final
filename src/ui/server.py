"""
FastAPI 서버 - Gradio UI 마운트

Gradio 인터페이스를 FastAPI 앱에 마운트하여 서빙
"""
from fastapi import FastAPI
import os
import uvicorn
import gradio as gr

from src.ui.gradio_app import create_gradio_interface

# FastAPI 앱 생성
app = FastAPI(title="ReAct Agent Server")

# Gradio 인터페이스 생성
gradio_demo = create_gradio_interface()

# Gradio 마운트 경로
MOUNT_PATH = os.getenv("GRADIO_MOUNT_PATH", "/")

# Gradio 앱 마운트
try:
    if hasattr(gr, "mount_gradio_app"):
        gr.mount_gradio_app(app, gradio_demo, path=MOUNT_PATH)
    else:
        app.mount(MOUNT_PATH, gradio_demo.app)
except Exception:
    app.mount(MOUNT_PATH, gradio_demo.app)


@app.get("/health")
def health():
    """헬스 체크 엔드포인트"""
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("src.ui.server:app", host=host, port=port, reload=True)