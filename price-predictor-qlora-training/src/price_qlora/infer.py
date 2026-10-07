"""Price one product description with a fine-tuned adapter.

    qlora-infer --adapter myuser/price-2025-11-28_18.47.07 --revision <commit> \
        --text "Title: Telpo Headlight Assembly ... Details: ..."
    echo "<summary>" | qlora-infer --adapter ... --stdin
    qlora-infer --adapter ... --test-index 3            # a product from the test split, shows the true price
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from price_qlora import config


def main(argv=None):
    load_dotenv(override=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True)
    p.add_argument("--revision")
    p.add_argument("--base-model", default=None)
    p.add_argument("--quant", default="4bit", choices=["4bit", "8bit", "bf16", "fp16"])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="product summary / description")
    g.add_argument("--stdin", action="store_true")
    g.add_argument("--test-index", type=int)
    p.add_argument("--lite", action="store_true")
    args = p.parse_args(argv)

    import os

    from huggingface_hub import login

    from price_qlora.inference import PricePredictor

    if os.environ.get("HF_TOKEN"):
        login(os.environ["HF_TOKEN"])
    predictor = PricePredictor(args.base_model, args.adapter, args.revision, args.quant)
    if args.test_index is not None:
        from datasets import load_dataset

        row = load_dataset(config.prompts_dataset(args.lite))["test"][args.test_index]
        from price_qlora.parse import extract_price

        print(f"Predicted ${extract_price(predictor(row['prompt'])):.2f}   actual ${float(row['completion']):.2f}")
        return
    text = sys.stdin.read() if args.stdin else args.text
    print(f"${predictor.price(text):.2f}")


if __name__ == "__main__":
    main()
