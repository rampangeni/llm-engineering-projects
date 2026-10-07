"""Train the deep neural network and write deep_neural_network.pth (+ .meta.json).

    train-dnn --lite --epochs 2 --hidden-size 512 --out artifacts/dnn_smoke.pth     # quick CPU try-out
    train-dnn --epochs 5 --out artifacts/deep_neural_network.pth                    # default run (GPU/MPS, hours)
    train-dnn ... --push-to-hub myuser/price-dnn                                    # optionally upload the artifacts
"""

from __future__ import annotations

from pathlib import Path

from price_dataset.cli.common import base_parser, load_items


def main(argv=None):
    p = base_parser("Train the deep neural network price model")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--val-size", type=int, default=1000, help="validation items used per epoch (default: val[:1000])")
    p.add_argument("--num-layers", type=int, default=10)
    p.add_argument("--hidden-size", type=int, default=4096)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--t-max", type=int, default=10, help="cosine schedule period in epochs (default: 10)")
    p.add_argument("--train-limit", type=int, help="use only the first N training items (smoke tests)")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--eval-size", type=int, default=200, help="test items evaluated after training (0 = skip)")
    p.add_argument("--out", default="artifacts/deep_neural_network.pth")
    p.add_argument("--push-to-hub", metavar="REPO", help="HF model repo (e.g. user/price-dnn) to upload weights + meta")
    args = p.parse_args(argv)

    from price_dataset.dnn.runner import DeepNeuralNetworkRunner
    from price_dataset.evaluator import evaluate

    train, val, test = load_items(args)
    if args.train_limit:
        train = train[: args.train_limit]
    runner = DeepNeuralNetworkRunner(
        train, val[: args.val_size], num_layers=args.num_layers, hidden_size=args.hidden_size,
        dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, t_max=args.t_max, device=args.device,
    )
    runner.setup()
    runner.train(epochs=args.epochs)

    extra = {"dataset": args.dataset or f"lite={args.lite}", "epochs": args.epochs}
    if args.eval_size:
        m = evaluate(runner.inference, test, size=args.eval_size, workers=1, title="deep neural network")
        extra["test_metrics"] = m.to_dict()
    weights, meta = runner.save(args.out, extra_meta=extra)
    print(f"Saved {weights} and {meta}")

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to_hub, repo_type="model", exist_ok=True)
        for f in (weights, meta):
            api.upload_file(path_or_fileobj=str(f), path_in_repo=Path(f).name, repo_id=args.push_to_hub)
        print(f"Uploaded to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
