"""Gradio chat UI with a pipeline selector and a retrieved-context panel."""

from __future__ import annotations

import gradio as gr

from .config import Settings
from .pipelines import get_pipeline

_cache: dict[str, object] = {}


def _pipeline(name: str):
    if name not in _cache:
        _cache[name] = get_pipeline(name, Settings.from_env())
    return _cache[name]


def format_context(context) -> str:
    out = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in context:
        out += f"<span style='color: #ff7800;'>Source: {doc.metadata['source']}</span>\n\n{doc.page_content}\n\n"
    return out


def chat(history: list[dict], pipeline_name: str = "basic"):
    """Answer the last user message in ``history``; returns (history, context markdown)."""
    question, prior = history[-1]["content"], history[:-1]
    try:
        answer, context = _pipeline(pipeline_name).answer(question, prior)
    except Exception as e:  # surface setup problems (empty DB, missing key) in the UI
        history.append({"role": "assistant", "content": f"Error: {e}"})
        return history, "*No context (error)*"
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)


def build_ui(default_pipeline: str = "basic") -> gr.Blocks:
    def put_message(message, history):
        return "", history + [{"role": "user", "content": message}]

    with gr.Blocks(title="Insurellm Expert Assistant", theme=gr.themes.Soft()) as ui:
        gr.Markdown("# Insurellm Expert Assistant\nAsk me anything about Insurellm!")
        pipeline = gr.Radio(["basic", "pro"], value=default_pipeline, label="Pipeline")
        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(label="Conversation", height=600, type="messages", show_copy_button=True)
                message = gr.Textbox(label="Your Question", placeholder="Ask anything about Insurellm...", show_label=False)
            with gr.Column():
                context_md = gr.Markdown(value="*Retrieved context will appear here*", container=True, height=600)
        message.submit(put_message, [message, chatbot], [message, chatbot]).then(
            chat, [chatbot, pipeline], [chatbot, context_md]
        )
    return ui


def main(pipeline: str = "basic", **launch_kwargs) -> None:
    build_ui(pipeline).launch(**launch_kwargs)
