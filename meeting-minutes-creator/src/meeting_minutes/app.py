"""Gradio UI: upload audio (or paste a transcript), stream minutes."""

from __future__ import annotations

import gradio as gr

from meeting_minutes.config import cuda_available, resolve_backend
from meeting_minutes.pipeline import create_minutes


def _handle(audio, transcript_text, backend, description):
    if not audio and not transcript_text:
        yield "", "Upload audio or paste a transcript."
        return
    try:
        for tr, minutes in create_minutes(
            audio_path=audio, transcript=transcript_text or None, backend=backend,
            meeting_description=description or "a meeting",
        ):
            yield tr, minutes or "_Generating minutes..._"
    except Exception as e:  # show setup problems (missing key, no GPU) in the UI
        yield "", f"**Error:** {e}"


def build_ui() -> gr.Blocks:
    default = resolve_backend("auto")
    with gr.Blocks(title="Meeting minutes creator") as ui:
        gr.Markdown(
            f"# Meeting minutes creator\nDetected default backend: **{default}** "
            f"(CUDA: {cuda_available()}). `local` = Whisper + Llama 4-bit on GPU, `api` = OpenAI."
        )
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(type="filepath", label="Meeting audio")
                transcript_in = gr.Textbox(label="...or paste a transcript (skips transcription)", lines=6)
                backend = gr.Radio(["auto", "local", "api"], value="auto", label="Backend")
                description = gr.Textbox(value="a Denver City council meeting", label="Meeting description")
                go = gr.Button("Create minutes", variant="primary")
            with gr.Column():
                transcript_out = gr.Textbox(label="Transcript", lines=10)
        minutes_out = gr.Markdown(label="Minutes")
        go.click(_handle, [audio, transcript_in, backend, description], [transcript_out, minutes_out])
    return ui
