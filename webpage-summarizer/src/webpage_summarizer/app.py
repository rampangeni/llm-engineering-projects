"""Gradio UI with a backend switch (OpenAI / Ollama)."""

from __future__ import annotations

import gradio as gr

from .config import BACKENDS, load_settings
from .scraper import ScrapeError
from .summarizer import stream_summary


def run(url: str, backend: str, model: str):
    """Gradio handler: streams markdown, reports errors in the output pane."""
    try:
        settings = load_settings(backend, model.strip() or None)
        yield from stream_summary(url, settings=settings)
    except (ScrapeError, RuntimeError, ValueError) as exc:
        yield f"**Error:** {exc}"
    except Exception as exc:  # network/API errors from the LLM client
        yield f"**LLM call failed:** {exc}"


def build_ui(default_backend: str | None = None, default_model: str | None = None) -> gr.Blocks:
    with gr.Blocks(title="Webpage Summarizer") as ui:
        gr.Markdown("# Webpage Summarizer\nSnarky summaries of any server-rendered page.")
        url = gr.Textbox(label="URL (include https://)", value="https://huggingface.co")
        with gr.Row():
            backend = gr.Dropdown(list(BACKENDS), value=default_backend or "openai", label="Backend")
            model = gr.Textbox(label="Model (blank = default for backend)", value=default_model or "")
        out = gr.Markdown(label="Summary")
        gr.Button("Summarize", variant="primary").click(run, [url, backend, model], out)
    return ui


def launch(backend: str | None = None, model: str | None = None) -> None:
    build_ui(backend, model).launch(inbrowser=True)
