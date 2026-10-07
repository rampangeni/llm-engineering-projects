"""Command line entry point: ``price-model <command>``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import subprocess
import sys

from dotenv import load_dotenv

from .client import PricerClient
from .config import ServiceSettings
from .preprocessor import Preprocessor

DEPLOY_MODULES = {
    "class": "price_model_service.service",
    "function": "price_model_service.service_function",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="price-model", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("config", help="print effective settings")
    s.set_defaults(func=cmd_config)

    s = sub.add_parser("deploy", help="modal deploy the pricer app")
    s.add_argument("--variant", choices=sorted(DEPLOY_MODULES), default="class")
    s.add_argument("--dry-run", action="store_true", help="print the command only")
    s.set_defaults(func=cmd_deploy)

    s = sub.add_parser("price", help="price a product description via the deployed app")
    s.add_argument("text")
    s.add_argument("--backend", choices=["modal-class", "modal-function", "mock"])
    s.add_argument("--preprocess", action="store_true", help="rewrite text with an LLM first")
    s.add_argument("--preprocessor-model")
    s.set_defaults(func=cmd_price)

    s = sub.add_parser("preprocess", help="only run the rewrite step")
    s.add_argument("text")
    s.add_argument("--model")
    s.set_defaults(func=cmd_preprocess)

    s = sub.add_parser("warm", help="keep the deployed class warm for N minutes")
    s.add_argument("--minutes", type=int, default=20)
    s.set_defaults(func=cmd_warm)

    s = sub.add_parser("cool", help="restore the default scaledown window")
    s.set_defaults(func=cmd_cool)

    s = sub.add_parser("run-ephemeral", help="teaching demos: hello | llama | pricer (ephemeral apps)")
    s.add_argument("demo", choices=["hello", "hello-europe", "llama", "pricer"])
    s.add_argument("text", nargs="?", default=None)
    s.set_defaults(func=cmd_ephemeral)
    return p


def cmd_config(args, settings: ServiceSettings) -> int:
    data = dataclasses.asdict(settings)
    data["finetuned_model"] = settings.finetuned_model or "(none - base model only)"
    print(json.dumps(data, indent=2))
    return 0


def cmd_deploy(args, settings: ServiceSettings) -> int:
    cmd = [sys.executable, "-m", "modal", "deploy", "-m", DEPLOY_MODULES[args.variant]]
    print("+", " ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.call(cmd)


def cmd_price(args, settings: ServiceSettings) -> int:
    if args.backend:
        settings = dataclasses.replace(settings, backend=args.backend)
    pre = Preprocessor(args.preprocessor_model) if args.preprocess else None
    client = PricerClient(settings=settings, preprocessor=pre)
    print(f"${client.price(args.text):.2f}")
    return 0


def cmd_preprocess(args, settings: ServiceSettings) -> int:
    print(Preprocessor(args.model).preprocess(args.text))
    return 0


def cmd_warm(args, settings: ServiceSettings) -> int:
    PricerClient(settings=settings).keep_warm(args.minutes)
    print(f"Container will stay warm for {args.minutes} minutes after the last call.")
    return 0


def cmd_cool(args, settings: ServiceSettings) -> int:
    PricerClient(settings=settings).cool_down()
    print(f"Scaledown window restored to {settings.scaledown_window}s.")
    return 0


def cmd_ephemeral(args, settings: ServiceSettings) -> int:
    import modal

    if args.demo in ("hello", "hello-europe"):
        from .apps.hello import app, hello, hello_europe

        fn, fargs = (hello if args.demo == "hello" else hello_europe), ()
    elif args.demo == "llama":
        from .apps.llama_smoke import app, generate

        fn, fargs = generate, (args.text or "Never gonna give you up, never gonna",)
    else:
        from .apps.pricer_ephemeral import app, price

        fn, fargs = price, (args.text or "Quadcast HyperX condenser mic, connects via usb-c",)
    with modal.enable_output(), app.run():
        print(fn.remote(*fargs))
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args, ServiceSettings.from_env())


if __name__ == "__main__":
    raise SystemExit(main())
