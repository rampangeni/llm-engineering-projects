"""Gradio streaming UI with model choice."""

from __future__ import annotations

import gradio as gr

from .brochure import stream_brochure
from .prompts import TONES
from .providers import PROVIDERS
from .scraper import ScrapeError


def generate(company_name: str, url: str, model: str, tone: str):
    """Gradio generator: errors are rendered into the output pane."""
    yield ""
    if not url.startswith(("http://", "https://")):
        yield "**Error:** URL must start with http:// or https://"
        return
    try:
        yield from stream_brochure(company_name, url, model, tone)
    except (ScrapeError, RuntimeError, ValueError) as exc:
        yield f"**Error:** {exc}"
    except Exception as exc:
        yield f"**LLM call failed:** {exc}"


def build_ui() -> gr.Interface:
    return gr.Interface(
        fn=generate,
        title="Brochure Generator",
        inputs=[
            gr.Textbox(label="Company name:"),
            gr.Textbox(label="Landing page URL including http:// or https://"),
            gr.Dropdown(list(PROVIDERS), label="Select model", value="GPT"),
            gr.Dropdown(list(TONES), label="Tone", value="professional"),
        ],
        outputs=[gr.Markdown(label="Brochure:")],
        examples=[
            ["Hugging Face", "https://huggingface.co", "GPT", "professional"],
            ["Hugging Face", "https://huggingface.co", "Claude", "humorous"],
        ],
        flagging_mode="never",
    )


def launch() -> None:
    build_ui().launch(inbrowser=True)
