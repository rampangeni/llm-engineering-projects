"""Zero-shot frontier-model comparison.

    eval-frontier --model openai/gpt-4.1-nano --size 200
    eval-frontier --model gemini/gemini-3-pro-preview --size 50 --workers 2
    eval-frontier --all-models           # all predefined models (costs money!)
"""

from __future__ import annotations

from price_dataset.cli.common import base_parser, load_items
from price_dataset.evaluator import Tester
from price_dataset.frontier import FRONTIER_MODELS, make_frontier_pricer


def main(argv=None):
    p = base_parser("Zero-shot frontier models")
    p.add_argument("--model", action="append", help="litellm model id; repeatable")
    p.add_argument("--all-models", action="store_true", help="run every predefined model")
    p.add_argument("--size", type=int, default=200)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--plot-dir")
    args = p.parse_args(argv)

    models = list(FRONTIER_MODELS) if args.all_models else (args.model or ["openai/gpt-4.1-nano"])
    _train, _val, test = load_items(args)
    for model in models:
        t = Tester(make_frontier_pricer(model), test, title=model, size=args.size, workers=args.workers)
        m = t.run()
        print(m.summary_line(model))
        if args.plot_dir:
            import os

            os.makedirs(args.plot_dir, exist_ok=True)
            t.plot(os.path.join(args.plot_dir, model.replace("/", "_")))


if __name__ == "__main__":
    main()
