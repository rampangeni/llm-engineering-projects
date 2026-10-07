"""Gradio UI: model dropdown, streaming port, run Python / run native."""

from __future__ import annotations

import gradio as gr

from code_generator.generator import stream_port
from code_generator.models import default_models
from code_generator.prompts import LANGUAGES
from code_generator.runner import compile_and_run, compile_command_for, run_python, speedup
from code_generator.samples import SAMPLE_FILES, load_sample
from code_generator.styles import CSS
from code_generator.system_info import retrieve_system_info

_state: dict = {"py_seconds": None}


def _fmt_time(res) -> str:
    return f"\n\n[wall time: {res.seconds:.4f}s]"


def do_port(model: str, python: str, language: str):
    """Streaming generator for the Convert button."""
    try:
        cmd = compile_command_for(language, LANGUAGES[language]["filename"], "main")
    except RuntimeError:
        cmd = ["(compiler not found; command unknown)"]
    try:
        yield from stream_port(model, python, language, cmd, retrieve_system_info())
    except Exception as e:  # surface provider/key errors in the UI
        yield f"// Error: {e}"


def do_run_python(code: str) -> str:
    res = run_python(code)
    _state["py_seconds"] = res.seconds if res.ok else None
    return res.render() + _fmt_time(res)


def do_run_native(code: str, language: str) -> str:
    res = compile_and_run(code, language)
    text = res.render()
    if res.ok:
        text += _fmt_time(res)
        s = speedup(_state["py_seconds"] or 0, res.seconds)
        if s:
            text += f"\n[speedup vs last Python run: {s:,.0f}x]"
    return text


def build_ui(language: str = "C++") -> gr.Blocks:
    models = [m.name for m in default_models()]
    with gr.Blocks(css=CSS, theme=gr.themes.Monochrome(), title="Port from Python") as ui:
        gr.Markdown(
            "**Warning:** Run buttons execute code on this machine (generated code included). "
            "Use a container/VM for untrusted input."
        )
        with gr.Row():
            lang = gr.Radio(list(LANGUAGES), value=language, label="Target language")
            sample = gr.Dropdown(list(SAMPLE_FILES), value=list(SAMPLE_FILES)[1], label="Sample program")
        with gr.Row(equal_height=True):
            with gr.Column(scale=6):
                python = gr.Code(
                    label="Python (original)", value=load_sample(list(SAMPLE_FILES)[1]), language="python", lines=26
                )
            with gr.Column(scale=6):
                native = gr.Code(label="Generated code", value="", language="cpp", lines=26)
        with gr.Row(elem_classes=["controls"]):
            python_run = gr.Button("Run Python", elem_classes=["run-btn", "py"])
            model = gr.Dropdown(models, value=models[0], show_label=False)
            convert = gr.Button("Port code", elem_classes=["convert-btn"])
            native_run = gr.Button("Compile & run", elem_classes=["run-btn", "cpp"])
        with gr.Row(equal_height=True):
            python_out = gr.TextArea(label="Python result", lines=8, elem_classes=["py-out"])
            native_out = gr.TextArea(label="Native result", lines=8, elem_classes=["cpp-out"])

        sample.change(lambda s: load_sample(s), inputs=sample, outputs=python)
        lang.change(lambda l: gr.Code(language="rust" if l == "Rust" else "cpp"), inputs=lang, outputs=native)
        convert.click(do_port, inputs=[model, python, lang], outputs=native)
        python_run.click(do_run_python, inputs=python, outputs=python_out)
        native_run.click(do_run_native, inputs=[native, lang], outputs=native_out)
    return ui
