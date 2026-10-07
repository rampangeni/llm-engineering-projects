"""``deal-hunter`` command line interface."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from typing import List

from dotenv import load_dotenv

from .config import Settings
from .log_utils import init_logging

AGENTS = ["scan", "specialist", "frontier", "nn", "ensemble", "message", "plan", "autonomous"]
DEFAULT_TEXT = "Shure MV7+ professional podcaster microphone with usb-c and XLR outputs"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="deal-hunter", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ui", help="launch the Gradio app (deals table, live logs, 3D plot, timer)")
    s.add_argument("--demo", action="store_true", help="offline: canned deals + mock pricer, no API keys")
    s.add_argument("--share", action="store_true")
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=cmd_ui)

    s = sub.add_parser("run-once", help="one planner run, print the resulting opportunities")
    s.add_argument("--demo", action="store_true")
    s.add_argument("--planner", choices=["workflow", "autonomous"])
    s.set_defaults(func=cmd_run_once)

    s = sub.add_parser("build-vectorstore", help="populate Chroma from the HF items dataset")
    s.add_argument("--dataset", help="HF dataset id (default HF_ITEMS_DATASET)")
    s.add_argument("--lite", action="store_true", help="use <user>/items_lite instead of items_full")
    s.add_argument("--split", default="train")
    s.add_argument("--limit", type=int, help="only the first N items (quick test)")
    s.add_argument("--batch-size", type=int, default=1000)
    s.add_argument("--rebuild", action="store_true", help="drop the existing collection first")
    s.set_defaults(func=cmd_build)

    s = sub.add_parser("agent", help="run one agent on its own")
    s.add_argument("name", choices=AGENTS)
    s.add_argument("text", nargs="?", default=None, help="product description (specialist/frontier/nn/ensemble/message)")
    s.add_argument("--demo", action="store_true", help="scan/plan: use canned deals and mocks")
    s.set_defaults(func=cmd_agent)

    s = sub.add_parser("reset-memory", help="truncate memory.json to 2 deals (seed from packaged example if empty)")
    s.set_defaults(func=cmd_reset)

    s = sub.add_parser("check", help="report which prerequisites/credentials are available")
    s.set_defaults(func=cmd_check)
    return p


def _collection(settings: Settings):
    from .vectorstore import open_collection

    return open_collection(settings.chroma_path)


def cmd_ui(args, settings: Settings) -> int:
    from .ui import App

    App(settings, demo=args.demo).run(share=args.share, inbrowser=not args.no_browser)
    return 0


def cmd_run_once(args, settings: Settings) -> int:
    from .framework import DealAgentFramework

    if args.planner:
        settings = dataclasses.replace(settings, planner=args.planner)
    memory = DealAgentFramework(settings, demo=args.demo).run()
    print(json.dumps([o.model_dump() for o in memory], indent=2))
    return 0


def cmd_build(args, settings: Settings) -> int:
    from sentence_transformers import SentenceTransformer

    from .vectorstore import ENCODER_NAME, build_vectorstore, load_items, open_collection, reset_collection

    dataset = args.dataset or settings.hf_items_dataset
    if args.lite and not args.dataset:
        dataset = dataset.replace("items_full", "items_lite")
    collection = reset_collection(settings.chroma_path) if args.rebuild else open_collection(settings.chroma_path)
    if collection.count() and not args.rebuild:
        print(f"Collection already holds {collection.count()} documents; use --rebuild to recreate. Nothing to do.")
        return 0
    print(f"Loading {dataset} [{args.split}] ...")
    rows = load_items(dataset, args.split, token=os.getenv("HF_TOKEN"))
    encoder = SentenceTransformer(ENCODER_NAME)
    n = build_vectorstore(rows, collection, encoder, args.batch_size, args.limit, progress=True)
    print(f"Added {n} documents to {settings.chroma_path!r} (total {collection.count()}).")
    return 0


def cmd_agent(args, settings: Settings) -> int:
    from .framework import build_planner
    from .models import Opportunity

    text = args.text or DEFAULT_TEXT
    name = args.name
    if name == "scan":
        from .agents.scanner import DemoScannerAgent, ScannerAgent

        agent = DemoScannerAgent(settings) if args.demo else ScannerAgent(settings)
        result = agent.scan(memory=[])
        print(result.model_dump_json(indent=2) if result else "No new deals")
    elif name == "specialist":
        from .agents.specialist import SpecialistAgent

        print(SpecialistAgent(settings).price(text))
    elif name == "frontier":
        from .agents.frontier import FrontierAgent

        print(FrontierAgent(_collection(settings), settings).price(text))
    elif name == "nn":
        from .agents.neural_network import NeuralNetworkAgent

        print(NeuralNetworkAgent(settings).price(text))
    elif name == "ensemble":
        from .agents.ensemble import EnsembleAgent

        coll = None if not settings.enable_frontier else _collection(settings)
        print(EnsembleAgent.from_settings(coll, settings).price(text))
    elif name == "message":
        from .agents.messaging import MessagingAgent

        print(MessagingAgent(settings).notify(text, 100.0, 300.0, "https://example.com"))
    else:  # plan | autonomous
        s = dataclasses.replace(settings, planner="autonomous" if name == "autonomous" else "workflow")
        coll = None if args.demo or not s.enable_frontier else _collection(s)
        planner = build_planner(s, coll, demo=args.demo)
        result = planner.plan(memory=[])
        print(result.model_dump_json(indent=2) if isinstance(result, Opportunity) else "No opportunity surfaced")
    return 0


def cmd_reset(args, settings: Settings) -> int:
    from .framework import DealAgentFramework

    DealAgentFramework.reset_memory(settings)
    print(f"Memory at {settings.memory_path!r} reset to 2 deals.")
    return 0


def cmd_check(args, settings: Settings) -> int:
    def have(var):
        return "set" if os.getenv(var) else "MISSING"

    lines: List[str] = [
        f"OPENAI_API_KEY (scanner, frontier, autonomous planner): {have('OPENAI_API_KEY')}",
        f"ANTHROPIC_API_KEY (messaging model {settings.messaging_model}): {have('ANTHROPIC_API_KEY')}",
        f"PUSHOVER_USER/PUSHOVER_TOKEN: {have('PUSHOVER_USER')}/{have('PUSHOVER_TOKEN')} (missing => dry-run)",
        f"HF_TOKEN (build-vectorstore): {have('HF_TOKEN')}",
        f"Specialist backend: {settings.specialist_backend} ({settings.modal_app_name}/{settings.modal_class_name})",
        f"NN weights {settings.nn_weights_path!r}: {'found' if os.path.exists(settings.nn_weights_path) else 'MISSING'}",
        f"Chroma path {settings.chroma_path!r}: {'found' if os.path.isdir(settings.chroma_path) else 'MISSING'}",
        f"Memory {settings.memory_path!r}: {'found' if os.path.exists(settings.memory_path) else 'missing (will start empty)'}",
    ]
    if settings.specialist_backend == "modal" and settings.enable_specialist:
        try:
            import modal

            modal.Cls.from_name(settings.modal_app_name, settings.modal_class_name).hydrate()
            lines.append("Modal pricer lookup: OK")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"Modal pricer lookup: FAILED ({type(exc).__name__}: {exc}) - deploy ../price-model-modal-service or use SPECIALIST_BACKEND=mock")
    print("\n".join(lines))
    return 0


def main(argv: List[str] | None = None) -> int:
    load_dotenv(override=True)
    init_logging(logging.INFO)
    args = build_parser().parse_args(argv)
    return args.func(args, Settings.from_env())


if __name__ == "__main__":
    sys.exit(main())
