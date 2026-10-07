"""Evaluate saved DNN weights.

    eval-dnn --weights artifacts/deep_neural_network.pth --size 200
    eval-dnn --hf-repo myuser/price-dnn            # download weights + meta from the Hub first
"""

from __future__ import annotations

from price_dataset.cli.common import base_parser, load_items


def main(argv=None):
    p = base_parser("Evaluate a saved deep neural network")
    p.add_argument("--weights", default="artifacts/deep_neural_network.pth")
    p.add_argument("--hf-repo", help="download deep_neural_network.pth (+ .meta.json) from this model repo")
    p.add_argument("--size", type=int, default=200)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", help="write metrics JSON")
    args = p.parse_args(argv)

    from price_dataset.dnn.runner import META_NAME, WEIGHTS_NAME, DeepNeuralNetworkPredictor
    from price_dataset.evaluator import evaluate

    weights, meta = args.weights, None
    if args.hf_repo:
        from huggingface_hub import hf_hub_download

        weights = hf_hub_download(args.hf_repo, WEIGHTS_NAME)
        try:
            meta = hf_hub_download(args.hf_repo, META_NAME)
        except Exception:
            print("No meta sidecar in repo - falling back to default y_mean/y_std")
    predictor = DeepNeuralNetworkPredictor(weights, meta, device=args.device)
    _train, _val, test = load_items(args)
    m = evaluate(predictor, test, size=args.size, workers=1, title="deep neural network")
    if args.output:
        import json

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(m.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
