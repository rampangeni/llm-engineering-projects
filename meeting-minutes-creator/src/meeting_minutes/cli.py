"""CLI: ``meeting-minutes run AUDIO|--transcript FILE`` or ``meeting-minutes ui``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from meeting_minutes.pipeline import create_minutes, read_transcript


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="meeting-minutes")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ui", help="launch Gradio UI")
    r = sub.add_parser("run", help="create minutes from audio or a transcript file")
    r.add_argument("audio", nargs="?", help="audio file (mp3/wav/m4a...)")
    r.add_argument("--transcript", help="text file; skips transcription")
    r.add_argument("--backend", choices=["auto", "local", "api"], default=None)
    r.add_argument("--description", default="a Denver City council meeting")
    r.add_argument("-o", "--output", type=Path, help="write minutes markdown here")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "ui":
        from meeting_minutes.app import build_ui

        build_ui().launch()
        return 0
    if not args.audio and not args.transcript:
        print("error: give an audio file or --transcript", file=sys.stderr)
        return 2
    transcript = read_transcript(args.transcript) if args.transcript else None
    minutes = ""
    printed = 0
    for tr, minutes in create_minutes(args.audio, transcript, args.backend, args.description):
        if minutes:
            sys.stdout.write(minutes[printed:])
            sys.stdout.flush()
            printed = len(minutes)
    print()
    if args.output:
        args.output.write_text(minutes, encoding="utf-8")
    return 0
