"""``openai-ft`` command line.

    openai-ft prepare   --train 100 --val 50            # write jsonl/fine_tune_{train,validation}.jsonl
    openai-ft launch    [--wandb-project gpt-pricer]    # upload + create job (state -> artifacts/ft_state.json)
    openai-ft status    [--wait]                        # monitor the job
    openai-ft evaluate  --model ft | base | <model-id>  # evaluate on test items
    openai-ft compare                                   # base vs fine-tuned side by side
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from price_openai_ft import config, finetune
from price_openai_ft.items import Item
from price_openai_ft.prompts import validate_jsonl, write_jsonl

STATE = "artifacts/ft_state.json"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="openai-ft", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lite", action="store_true", help="use <user>/items_lite (20k) instead of items_full")
    p.add_argument("--dataset", help="explicit hub dataset overriding --lite/HF_USERNAME")
    p.add_argument("--state", default=STATE, help="json file remembering file ids / job id / model name")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("prepare", help="build jsonl files from the hub dataset")
    s.add_argument("--train", type=int, default=config.DEFAULT_TRAIN, help="training examples (default: 100; larger sizes such as 200/2000/20000 also work)")
    s.add_argument("--val", type=int, default=config.DEFAULT_VAL)
    s.add_argument("--out-dir", default="jsonl")

    s = sub.add_parser("launch", help="upload jsonl files and start the fine-tuning job")
    s.add_argument("--out-dir", default="jsonl")
    s.add_argument("--model", default=config.BASE_MODEL)
    s.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    s.add_argument("--batch-size", default=str(config.DEFAULT_BATCH_SIZE), help="int or 'auto'")
    s.add_argument("--lr-multiplier", type=float)
    s.add_argument("--suffix", default=config.SUFFIX)
    s.add_argument("--wandb-project", help="enable W&B integration (needs key linked in OpenAI dashboard)")
    s.add_argument("--no-validation", action="store_true")

    s = sub.add_parser("status", help="show job status / events")
    s.add_argument("--job-id")
    s.add_argument("--wait", action="store_true")
    s.add_argument("--poll-seconds", type=int, default=30)

    for name in ("evaluate", "compare"):
        s = sub.add_parser(name)
        s.add_argument("--size", type=int, default=200)
        s.add_argument("--workers", type=int, default=5)
        s.add_argument("--max-tokens", type=int, default=config.MAX_TOKENS)
        s.add_argument("--output", help="write metrics json")
        if name == "evaluate":
            s.add_argument("--model", default="ft", help="'ft' (from state), 'base' (gpt-4.1-nano) or any model id")
            s.add_argument("--save-predictions", help="write per-item predictions json")
    return p


def _client():
    from openai import OpenAI

    return OpenAI()


def _dataset_name(args) -> str:
    return args.dataset or config.dataset_name(args.lite)


def resolve_model(choice: str, state_path: str) -> str:
    if choice == "base":
        return config.BASE_MODEL_ALIAS
    if choice == "ft":
        try:
            name = finetune.load_state(state_path).get("fine_tuned_model")
        except FileNotFoundError:
            name = None
        if not name:
            raise SystemExit("No fine-tuned model recorded yet: run `openai-ft status --wait` after `launch` first.")
        return name
    return choice


def cmd_prepare(args):
    config.hf_login()
    train, val, _test = Item.from_hub(_dataset_name(args))
    if args.train > len(train) or args.val > len(val):
        raise SystemExit(f"Dataset only has {len(train)} train / {len(val)} validation items")
    if args.train < config.MIN_EXAMPLES:
        raise SystemExit(f"OpenAI needs at least {config.MIN_EXAMPLES} training examples")
    out = Path(args.out_dir)
    tr = write_jsonl(train[: args.train], out / "fine_tune_train.jsonl")
    va = write_jsonl(val[: args.val], out / "fine_tune_validation.jsonl")
    print(f"Wrote {validate_jsonl(tr)} train examples to {tr} and {validate_jsonl(va)} validation examples to {va}")


def cmd_launch(args):
    client = _client()
    out = Path(args.out_dir)
    train_path, val_path = out / "fine_tune_train.jsonl", out / "fine_tune_validation.jsonl"
    n = validate_jsonl(train_path)
    train_id = finetune.upload(client, train_path)
    val_id = None if args.no_validation else finetune.upload(client, val_path)
    batch = args.batch_size if args.batch_size == "auto" else int(args.batch_size)
    job = finetune.launch(
        client, train_file_id=train_id, validation_file_id=val_id, model=args.model, epochs=args.epochs,
        batch_size=batch, learning_rate_multiplier=args.lr_multiplier, suffix=args.suffix,
        wandb_project=args.wandb_project,
    )
    finetune.save_state(args.state, train_file_id=train_id, validation_file_id=val_id, job_id=job.id,
                        base_model=args.model, n_train=n, dataset=_dataset_name(args))
    print(f"Launched job {job.id} ({n} examples). Monitor: openai-ft status --wait   or https://platform.openai.com/finetune")


def cmd_status(args):
    client = _client()
    job_id = args.job_id
    if not job_id:
        try:
            job_id = finetune.load_state(args.state).get("job_id")
        except FileNotFoundError:
            job_id = finetune.latest_job_id(client)
    s = finetune.wait(client, job_id, args.poll_seconds) if args.wait else finetune.status(client, job_id)
    if not args.wait:
        for m in s["events"]:
            print(m)
    print(json.dumps({k: v for k, v in s.items() if k != "events"}, indent=2, default=str))
    if s["fine_tuned_model"]:
        finetune.save_state(args.state, job_id=job_id, fine_tuned_model=s["fine_tuned_model"])
        print(f"Fine-tuned model recorded in {args.state}: {s['fine_tuned_model']}")


def _evaluate(client, model, test, args):
    from price_openai_ft.evaluator import Tester
    from price_openai_ft.predictor import make_predictor

    t = Tester(make_predictor(client, model, args.max_tokens), test, title=model, size=args.size, workers=args.workers)
    m = t.run()
    print(m.summary_line(model))
    return t, m


def cmd_evaluate(args):
    config.hf_login()
    _tr, _va, test = Item.from_hub(_dataset_name(args))
    model = resolve_model(args.model, args.state)
    t, m = _evaluate(_client(), model, test, args)
    if args.output:
        Path(args.output).write_text(json.dumps({model: m.to_dict()}, indent=2), encoding="utf-8")
    if args.save_predictions:
        Path(args.save_predictions).write_text(json.dumps(t.rows(), indent=2), encoding="utf-8")


def cmd_compare(args):
    config.hf_login()
    _tr, _va, test = Item.from_hub(_dataset_name(args))
    client = _client()
    results = {}
    for choice in ("base", "ft"):
        model = resolve_model(choice, args.state)
        _t, m = _evaluate(client, model, test, args)
        results[model] = m.to_dict()
    base, ft = (v["average_error"] for v in results.values())
    print(f"\nFine-tuned vs base: ${ft:,.2f} vs ${base:,.2f} ({'better' if ft < base else 'worse'} by ${abs(base - ft):,.2f})")
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")


COMMANDS = {"prepare": cmd_prepare, "launch": cmd_launch, "status": cmd_status, "evaluate": cmd_evaluate, "compare": cmd_compare}


def main(argv=None):
    load_dotenv(override=True)
    args = build_parser().parse_args(argv)
    COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
