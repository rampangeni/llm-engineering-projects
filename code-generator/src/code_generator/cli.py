"""CLI: ``code-generator ui`` | ``port FILE`` | ``info``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from code_generator.generator import stream_port
from code_generator.models import default_models
from code_generator.prompts import LANGUAGES
from code_generator.runner import compile_and_run, compile_command_for, run_python, speedup
from code_generator.system_info import retrieve_system_info, rust_toolchain_info


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="code-generator", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    ui = sub.add_parser("ui", help="launch the Gradio UI")
    ui.add_argument("--language", choices=list(LANGUAGES), default="C++")
    ui.add_argument("--no-browser", action="store_true")
    sub.add_parser("info", help="print system/compiler info")
    port = sub.add_parser("port", help="port a Python file and optionally run it")
    port.add_argument("file", type=Path)
    port.add_argument("--language", choices=list(LANGUAGES), default="C++")
    port.add_argument("--model", default=default_models()[0].name)
    port.add_argument("--run", action="store_true", help="compile+run both and compare (EXECUTES CODE)")
    port.add_argument("-o", "--output", type=Path, help="write generated code here")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "info":
        print(json.dumps({"system": retrieve_system_info(), "rust": rust_toolchain_info()}, indent=2))
        return 0
    if args.cmd == "ui":
        from code_generator.app import build_ui

        build_ui(args.language).launch(inbrowser=not args.no_browser)
        return 0
    python = args.file.read_text(encoding="utf-8")
    cmd = compile_command_for(args.language, LANGUAGES[args.language]["filename"], "main")
    code = ""
    for code in stream_port(args.model, python, args.language, cmd, retrieve_system_info()):
        pass
    if args.output:
        args.output.write_text(code, encoding="utf-8")
    else:
        sys.stdout.write(code)
    if args.run:
        py = run_python(python)
        nat = compile_and_run(code, args.language)
        print(f"\n--- python ({py.seconds:.3f}s) ---\n{py.render()}", file=sys.stderr)
        print(f"--- {args.language} ({nat.seconds:.3f}s) ---\n{nat.render()}", file=sys.stderr)
        s = speedup(py.seconds, nat.seconds)
        if s and py.ok and nat.ok:
            print(f"speedup: {s:,.1f}x", file=sys.stderr)
    return 0
