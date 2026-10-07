"""CLI: run | compare | cache-demo | presets | ui."""

from __future__ import annotations

import argparse
import sys

from .compare import PROMPTS, compare_models, format_rows
from .debate import DebateEngine, format_turn, load_config
from .llm import LLMRouter, ProviderError
from .presets import PRESETS, get_preset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="debate-arena", description="Multi-LLM conversations and comparisons.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run a multi-bot conversation")
    r.add_argument("--preset", default="classic", choices=sorted(PRESETS))
    r.add_argument("--config", help="JSON config file (overrides --preset)")
    r.add_argument("--rounds", type=int, help="Override number of rounds")
    r.add_argument("--topic", help="Moderator opening topic")

    c = sub.add_parser("compare", help="Send one prompt to several models")
    c.add_argument("prompt", help=f"Prompt text or one of: {', '.join(PROMPTS)}")
    c.add_argument("-m", "--model", action="append", required=True, dest="models",
                   help="provider:model (repeatable), e.g. openai:gpt-4.1-mini")
    c.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"])
    c.add_argument("--cost", action="store_true", help="Use litellm to report cost (needs [litellm] extra)")

    d = sub.add_parser("cache-demo", help="Prompt-caching demo with Hamlet (litellm)")
    d.add_argument("--model", default="gemini/gemini-3.1-flash-lite")

    sub.add_parser("presets", help="List built-in presets")
    sub.add_parser("ui", help="Launch the Gradio app")
    return p


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else get_preset(args.preset)
    if args.rounds:
        cfg.rounds = args.rounds
    if args.topic:
        cfg.topic = args.topic
    engine = DebateEngine(cfg, LLMRouter().text)
    for t in engine.seed_turns():
        print(format_turn(t))
    for turn in engine.run():
        print(format_turn(turn), flush=True)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    prompt = PROMPTS.get(args.prompt, args.prompt)
    rows = compare_models(prompt, args.models, reasoning_effort=args.reasoning_effort, with_cost=args.cost)
    print(format_rows(rows))
    return 0 if any(r.error is None for r in rows) else 1


def cmd_cache(args: argparse.Namespace) -> int:
    from .litellm_tools import cache_demo

    for row in cache_demo(args.model):
        cost = "n/a" if row["cost_cents"] is None else f"{row['cost_cents']:.4f}"
        print(f"{row['step']:<40} in={row['input_tokens']} out={row['output_tokens']} "
              f"cached={row['cached_tokens']} cost_cents={cost}")
        print(f"  -> {row['answer'][:160]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "run":
            return cmd_run(args)
        if args.cmd == "compare":
            return cmd_compare(args)
        if args.cmd == "cache-demo":
            return cmd_cache(args)
        if args.cmd == "presets":
            for name in sorted(PRESETS):
                cfg = get_preset(name)
                print(f"{name}: " + ", ".join(f"{b.name}={b.model}" for b in cfg.bots))
            return 0
        if args.cmd == "ui":
            from .app import launch

            launch()
            return 0
    except (ProviderError, ValueError, ImportError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2
