"""Human baseline.

    human-baseline export --out human_in.csv --n 100     # then fill column 2 with YOUR guesses, save as human_out.csv
    human-baseline eval --csv human_out.csv              # score your guesses on the first N test items
    human-baseline eval --csv data/human_guesses.csv --dataset rampangeni/items_full   # reproduce the reference ~$87.62 run
"""

from __future__ import annotations

from price_dataset.baselines import export_human_csv, make_human_pricer, read_human_csv
from price_dataset.cli.common import base_parser, load_items
from price_dataset.evaluator import evaluate


def main(argv=None):
    p = base_parser("Human price-guessing baseline")
    p.add_argument("command", choices=["export", "eval"])
    p.add_argument("--out", default="human_in.csv")
    p.add_argument("--csv", default="human_out.csv")
    p.add_argument("--n", type=int, default=100)
    args = p.parse_args(argv)

    _train, _val, test = load_items(args)
    if args.command == "export":
        export_human_csv(test, args.out, args.n)
        print(f"Wrote {args.n} summaries to {args.out}. Put your price guess in column 2.")
    else:
        preds = read_human_csv(args.csv)
        evaluate(make_human_pricer(test, preds), test, size=len(preds), workers=1, title="human")


if __name__ == "__main__":
    main()
