"""Load the base model at different precisions and compare memory (and optionally error).

    qlora-quant-compare --presets 4bit 8bit bf16
    qlora-quant-compare --presets 4bit --eval-size 50 --lite     # untuned baseline error
"""

from __future__ import annotations

import argparse
import gc

from dotenv import load_dotenv

from price_qlora import config
from price_qlora.quant import PRESETS, footprint_mb, load_base_model


def main(argv=None):
    load_dotenv(override=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default=None)
    p.add_argument("--presets", nargs="+", choices=PRESETS, default=["4bit", "8bit", "bf16"])
    p.add_argument("--eval-size", type=int, default=0, help="also evaluate this many test items (0 = memory only)")
    p.add_argument("--lite", action="store_true")
    p.add_argument("--dataset")
    args = p.parse_args(argv)

    import torch

    base = args.base_model or config.base_model()
    data = None
    if args.eval_size:
        from datasets import load_dataset

        data = load_dataset(args.dataset or config.prompts_dataset(args.lite))["test"]
    for preset in args.presets:
        tokenizer, model = load_base_model(base, preset, "float16")
        print(f"{base} @ {preset}: {footprint_mb(model):,.0f} MB")
        if data is not None:
            from price_qlora.evaluator import Tester
            from price_qlora.inference import PricePredictor

            pred = PricePredictor.from_loaded(tokenizer, model)
            m = Tester(pred, list(data), title=f"{base}@{preset}", size=args.eval_size).run()
            print(m.summary_line(f"{preset}"))
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
