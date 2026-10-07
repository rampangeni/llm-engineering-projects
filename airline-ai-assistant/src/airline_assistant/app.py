"""Gradio multimodal UI: chatbot + city image + spoken replies."""

from __future__ import annotations

import os
import tempfile

import gradio as gr

from .assistant import Assistant
from .factory import build_assistant


def save_audio(data: bytes | None) -> str | None:
    """gr.Audio needs a file path; write the mp3 bytes to a temp file."""
    if not data:
        return None
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(data)
        return f.name


def put_message_in_chatbot(message: str, history: list[dict]):
    return "", history + [{"role": "user", "content": message}]


def make_chat_fn(assistant: Assistant):
    def chat(history: list[dict]):
        try:
            reply = assistant.respond(history)
            text = reply.text
            audio, image = save_audio(reply.audio), reply.image
        except Exception as exc:  # API/network failure shown in the chat, not a stack trace
            text, audio, image = f"Sorry, something went wrong: {exc}", None, None
        return history + [{"role": "assistant", "content": text}], audio, image

    return chat


def build_ui(assistant: Assistant) -> gr.Blocks:
    with gr.Blocks(title="FlightAI") as ui:
        gr.Markdown("# FlightAI assistant\nAsk about ticket prices or book a flight.")
        with gr.Row():
            chatbot = gr.Chatbot(height=500, type="messages")
            image_output = gr.Image(height=500, interactive=False)
        with gr.Row():
            audio_output = gr.Audio(autoplay=True, visible=assistant.media is not None)
        with gr.Row():
            message = gr.Textbox(label="Chat with our AI Assistant:")
        message.submit(put_message_in_chatbot, [message, chatbot], [message, chatbot]).then(
            make_chat_fn(assistant), chatbot, [chatbot, audio_output, image_output])
    return ui


def launch(assistant: Assistant | None = None) -> None:
    assistant = assistant or build_assistant()
    user, password = os.getenv("GRADIO_USER"), os.getenv("GRADIO_PASSWORD")
    auth = (user, password) if user and password else None
    build_ui(assistant).launch(inbrowser=True, auth=auth)
