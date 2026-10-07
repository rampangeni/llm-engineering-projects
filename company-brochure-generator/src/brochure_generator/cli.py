"""CLI: ``brochure-generator "Hugging Face" https://huggingface.co --model Claude``."""

from __future__ import annotations

import argparse
import sys

from .brochure import stream_brochure
from .prompts import TONES
from .providers import PROVIDERS
from .scraper import ScrapeError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="brochure-generator",
                                description="Generate a company brochure from its website.")
    p.add_argument("company", nargs="?", help="Company name")
    p.add_argument("url", nargs="?", help="Landing page URL")
    p.add_argument("--model", choices=list(PROVIDERS), default="GPT")
    p.add_argument("--tone", choices=list(TONES), default="professional")
    p.add_argument("--ui", action="store_true", help="Launch the Gradio UI")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.ui:
        from .app import launch

        launch()
        return 0
    if not (args.company and args.url):
        parser.error("company and url are required unless --ui is given")
    printed = 0
    try:
        for text in stream_brochure(args.company, args.url, args.model, args.tone):
            sys.stdout.write(text[printed:])
            sys.stdout.flush()
            printed = len(text)
        print()
    except (ScrapeError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
