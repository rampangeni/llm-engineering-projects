"""Command line interface: ``webpage-summarizer URL`` or ``--ui`` for Gradio."""

from __future__ import annotations

import argparse
import sys

from .config import BACKENDS, load_settings
from .scraper import ScrapeError
from .summarizer import summarize


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="webpage-summarizer", description="Summarize a web page with an LLM."
    )
    p.add_argument("url", nargs="?", help="URL to summarize (e.g. https://huggingface.co)")
    p.add_argument("--backend", choices=BACKENDS, help="openai (default) or ollama")
    p.add_argument("--model", help="Override the model name for the chosen backend")
    p.add_argument("--max-chars", type=int, default=2000, help="Max page chars sent to the LLM")
    p.add_argument("--ui", action="store_true", help="Launch the Gradio web UI")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.ui:
        from .app import launch

        launch(backend=args.backend, model=args.model)
        return 0
    if not args.url:
        build_parser().error("a URL is required unless --ui is given")
    try:
        settings = load_settings(args.backend, args.model)
        print(summarize(args.url, settings=settings, max_chars=args.max_chars))
    except (ScrapeError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
