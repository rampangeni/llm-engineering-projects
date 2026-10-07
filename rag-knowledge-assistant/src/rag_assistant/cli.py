"""Command line entry point: ``rag <command>``."""

from __future__ import annotations

import argparse
import sys

from .config import PIPELINES, Settings


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag", description="Insurellm RAG knowledge assistant")
    sub = p.add_subparsers(dest="command", required=True)

    def add_pipeline(sp):
        sp.add_argument("-p", "--pipeline", choices=PIPELINES, help="basic or pro (default: $RAG_PIPELINE or basic)")

    sp = sub.add_parser("ingest", help="chunk, embed and store the knowledge base")
    add_pipeline(sp)

    sp = sub.add_parser("ask", help="ask a single question from the terminal")
    add_pipeline(sp)
    sp.add_argument("question")

    sp = sub.add_parser("chat", help="launch the Gradio chat UI")
    add_pipeline(sp)
    sp.add_argument("--no-browser", action="store_true")
    sp.add_argument("--port", type=int, default=None)

    sp = sub.add_parser("eval", help="evaluate one test (--test N) or open the dashboard (--dashboard)")
    add_pipeline(sp)
    sp.add_argument("--test", type=int, help="row number in tests.jsonl to evaluate in the terminal")
    sp.add_argument("--dashboard", action="store_true", help="launch the Gradio evaluation dashboard")
    sp.add_argument("--no-browser", action="store_true")
    sp.add_argument("--port", type=int, default=None)

    sp = sub.add_parser("viz", help="t-SNE plot of the vector store (needs the 'viz' extra)")
    add_pipeline(sp)
    sp.add_argument("--dims", type=int, choices=(2, 3), default=2)
    sp.add_argument("--output", help="write an HTML file instead of opening a window")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    name = getattr(args, "pipeline", None) or settings.pipeline
    launch = {}
    if getattr(args, "no_browser", False):
        launch["inbrowser"] = False
    else:
        launch["inbrowser"] = True
    if getattr(args, "port", None):
        launch["server_port"] = args.port

    if args.command == "ingest":
        from .documents import load_documents
        from .pipelines import get_pipeline

        docs = load_documents(settings.knowledge_base_dir)
        print(f"Loaded {len(docs)} documents; ingesting with the '{name}' pipeline...")
        n = get_pipeline(name, settings).ingest(docs)
        print(f"Stored {n} chunks in {settings.db_dir(name)}")
    elif args.command == "ask":
        from .pipelines import get_pipeline

        answer, ctx = get_pipeline(name, settings).answer(args.question)
        print(answer)
        print("\nSources: " + ", ".join(sorted({c.metadata["source"] for c in ctx})))
    elif args.command == "chat":
        from . import app

        app.main(name, **launch)
    elif args.command == "eval":
        if args.dashboard:
            from .evaluation import dashboard

            dashboard.main(name, **launch)
        elif args.test is not None:
            from .evaluation.evaluator import run_single
            from .evaluation.metrics import load_tests
            from .llm import LiteLLMClient
            from .pipelines import get_pipeline

            run_single(
                get_pipeline(name, settings),
                LiteLLMClient(settings.judge_llm_model),
                load_tests(settings.tests_file),
                args.test,
            )
        else:
            print("Specify --test N or --dashboard", file=sys.stderr)
            return 2
    elif args.command == "viz":
        from .viz import visualize

        visualize(settings, name, args.dims, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
