"""Run baseline predictors on the test set.

    run-baselines --lite --size 200 --only random constant linear
    run-baselines --only bow rf xgb --output results.json --plot-dir work/plots
"""

from __future__ import annotations

import json

from price_dataset.baselines import (
    make_bow_linear_pricer,
    make_bow_vectorizer,
    make_constant_pricer,
    make_linear_regression_pricer,
    make_random_forest_pricer,
    make_random_pricer,
    make_xgboost_pricer,
)
from price_dataset.cli.common import base_parser, load_items
from price_dataset.evaluator import evaluate

CHOICES = ["random", "constant", "linear", "bow", "rf", "xgb"]


def main(argv=None):
    p = base_parser("Evaluate baseline price predictors")
    p.add_argument("--only", nargs="*", choices=CHOICES, default=CHOICES)
    p.add_argument("--size", type=int, default=200, help="test items to evaluate (default: 200)")
    p.add_argument("--rf-subset", type=int, default=15_000, help="rows used to fit the random forest (default: 15,000)")
    p.add_argument("--xgb-trees", type=int, default=1000)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--output", help="write metrics as JSON")
    p.add_argument("--plot-dir", help="write plotly HTML charts here (needs [plots])")
    args = p.parse_args(argv)

    train, _val, test = load_items(args)
    results = {}

    def run(name, predictor):
        from price_dataset.evaluator import Tester

        t = Tester(predictor, test, title=name, size=args.size, workers=1)
        m = t.run()
        print(m.summary_line(name))
        results[name] = m.to_dict()
        if args.plot_dir:
            import os

            os.makedirs(args.plot_dir, exist_ok=True)
            t.plot(os.path.join(args.plot_dir, name.replace(" ", "_").lower()))

    if "random" in args.only:
        run("random", make_random_pricer())
    if "constant" in args.only:
        pricer = make_constant_pricer(train)
        print(f"training average: {pricer.average:.2f}")
        run("constant", pricer)
    if "linear" in args.only:
        run("linear regression (weight, text length)", make_linear_regression_pricer(train))
    if {"bow", "rf", "xgb"} & set(args.only):
        vectorizer, X = make_bow_vectorizer(train)
        if "bow" in args.only:
            run("bag-of-words linear regression", make_bow_linear_pricer(vectorizer, X, train))
        if "rf" in args.only:
            run("random forest", make_random_forest_pricer(vectorizer, X, train, args.rf_subset, args.n_jobs))
        if "xgb" in args.only:
            run("xgboost", make_xgboost_pricer(vectorizer, X, train, args.xgb_trees, args.n_jobs))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
