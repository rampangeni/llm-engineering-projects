"""CLI: ``airline-assistant`` (Gradio UI) or ``airline-assistant --chat`` (terminal)."""

from __future__ import annotations

import argparse
import sys

from .config import load_settings
from .factory import build_assistant


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="airline-assistant", description="FlightAI airline assistant.")
    p.add_argument("--backend", choices=["openai", "ollama"], help="Default from ASSISTANT_BACKEND")
    p.add_argument("--model", help="Override chat model")
    p.add_argument("--chat", action="store_true", help="Terminal chat instead of the web UI")
    p.add_argument("--no-media", action="store_true", help="Disable image generation and speech")
    return p


def terminal_chat(assistant) -> None:
    history: list[dict[str, str]] = []
    print("FlightAI - type 'quit' to exit.")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line.lower() in {"quit", "exit"}:
            break
        if not line:
            continue
        history.append({"role": "user", "content": line})
        text, _ = assistant.complete(history)
        history.append({"role": "assistant", "content": text})
        print(f"flightai> {text}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.backend, args.model)
        assistant = build_assistant(settings, media=not args.no_media)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.chat:
        terminal_chat(assistant)
    else:
        from .app import launch

        launch(assistant)
    return 0
