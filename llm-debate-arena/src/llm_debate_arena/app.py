"""Gradio UI: multi-bot conversation tab + model comparison tab."""

from __future__ import annotations

import json

import gradio as gr

from .compare import PROMPTS, compare_models
from .debate import DebateEngine, config_from_dict, config_to_dict, format_turn
from .llm import LLMRouter
from .presets import PRESETS, get_preset


def run_debate(config_json: str, rounds: int):
    """Generator streaming the transcript as markdown after every turn."""
    try:
        data = json.loads(config_json)
        data["rounds"] = int(rounds)
        cfg = config_from_dict(data)
        engine = DebateEngine(cfg, LLMRouter().text)
    except (ValueError, KeyError, TypeError) as exc:
        yield f"**Invalid configuration:** {exc}"
        return
    text = "".join(format_turn(t) for t in engine.seed_turns())
    yield text
    try:
        for turn in engine.run():
            text += "\n" + format_turn(turn)
            yield text
    except Exception as exc:
        yield text + f"\n\n**Stopped:** {exc}"


def load_preset(name: str):
    cfg = get_preset(name)
    return json.dumps(config_to_dict(cfg), indent=2), cfg.rounds


def run_compare(prompt: str, models: str):
    specs = [m.strip() for m in models.replace("\n", ",").split(",") if m.strip()]
    rows = compare_models(prompt, specs)
    return [[r.spec, r.error or r.text, r.input_tokens, r.output_tokens, round(r.seconds, 1)] for r in rows]


def build_ui() -> gr.Blocks:
    first = next(iter(PRESETS))
    first_json, first_rounds = load_preset(first)
    with gr.Blocks(title="LLM Debate Arena") as ui:
        gr.Markdown("# LLM Debate Arena")
        with gr.Tab("Conversation"):
            preset = gr.Dropdown(list(PRESETS), value=first, label="Preset")
            cfg = gr.Code(value=first_json, language="json", label="Config (bots: name/model/system_prompt)")
            rounds = gr.Slider(1, 12, value=first_rounds, step=1, label="Rounds")
            out = gr.Markdown()
            preset.change(load_preset, preset, [cfg, rounds])
            gr.Button("Start", variant="primary").click(run_debate, [cfg, rounds], out)
        with gr.Tab("Compare models"):
            prompt = gr.Textbox(label="Prompt", lines=5, value=PROMPTS["joke"])
            models = gr.Textbox(label="Models (comma separated provider:model)",
                                value="openai:gpt-4.1-mini, anthropic:claude-haiku-4-5, ollama:llama3.2")
            table = gr.Dataframe(headers=["model", "answer / error", "in tok", "out tok", "seconds"],
                                 wrap=True)
            gr.Button("Compare", variant="primary").click(run_compare, [prompt, models], table)
    return ui


def launch() -> None:
    build_ui().launch(inbrowser=True)
