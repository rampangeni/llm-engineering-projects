"""Evaluate a fine-tuned adapter (or the untuned base model) on the test split.

    qlora-evaluate --adapter myuser/price-2025-11-28_18.47.07 --revision <commit> --size 200
    qlora-evaluate --base-only --size 200               # untuned 4-bit base (reference: $110.72)
    qlora-evaluate --adapter artifacts/price-<run>/final --lite --output metrics.json
"""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from price_qlora import config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", help="hub repo id or local folder of the LoRA adapter")
    p.add_argument("--revision", help="hub commit hash / branch of the adapter (pin the checkpoint you picked)")
    p.add_argument("--base-only", action="store_true", help="evaluate the base model without an adapter")
    p.add_argument("--base-model", default=None)
    p.add_argument("--quant", default="4bit", choices=["4bit", "8bit", "bf16", "fp16"])
    p.add_argument("--lite", action="store_true")
    p.add_argument("--dataset", help="prompt dataset (needs a 'test' split with prompt/completion)")
    p.add_argument("--size", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=5)
    p.add_argument("--output", help="write metrics + per-item rows as JSON")
    return p


def main(argv=None):
    load_dotenv(override=True)
    args = build_parser().parse_args(argv)
    if not args.adapter and not args.base_only:
        raise SystemExit("Pass --adapter <repo-or-folder> or --base-only")

    import os

    from datasets import load_dataset
    from huggingface_hub import login

    from price_qlora.evaluator import Tester
    from price_qlora.inference import PricePredictor

    if os.environ.get("HF_TOKEN"):
        login(os.environ["HF_TOKEN"])
    data = list(load_dataset(args.dataset or config.prompts_dataset(args.lite))["test"])
    predictor = PricePredictor(
        base_model=args.base_model, adapter=None if args.base_only else args.adapter,
        revision=args.revision, quant=args.quant, max_new_tokens=args.max_new_tokens,
    )
    title = f"{args.base_model or config.base_model()}" + ("" if args.base_only else f" + {args.adapter}")
    tester = Tester(predictor, data, title=title, size=args.size)
    metrics = tester.run()
    print(metrics.summary_line(title))
    if args.output:
        tester.dump(args.output)


if __name__ == "__main__":
    main()
