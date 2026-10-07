"""Days 3-4: QLoRA fine-tune with TRL's SFTTrainer + PEFT LoRA; push adapter checkpoints to the Hub.

    qlora-train --lite --epochs 1 --dry-run                 # print the resolved config, no GPU needed
    qlora-train --lite --batch-size 4 --grad-accum 4        # Colab T4 smoke run
    qlora-train                                             # full dataset (800k) on an A100

Produces: HF model repo ``<HF_USERNAME>/price-<RUN_NAME>`` (LoRA adapter, one commit per checkpoint) and
``artifacts/price-<RUN_NAME>/{run.json, final/}``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from price_qlora import config
from price_qlora.config import LoraSettings, TrainSettings


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = TrainSettings()
    p.add_argument("--lite", action="store_true", help="use items_prompts_lite (20k train) instead of full (800k)")
    p.add_argument("--dataset", help="explicit prompt dataset on the hub (splits train/val)")
    p.add_argument("--base-model", default=None, help=f"default $BASE_MODEL or {config.BASE_MODEL}")
    p.add_argument("--hf-user", default=None, help="namespace for the pushed adapter (default $HF_USERNAME)")
    p.add_argument("--run-name", default=None, help="default: timestamp YYYY-MM-DD_HH.MM.SS")
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size, help="per device; 16 fits an A100-40GB, use 4 + --grad-accum 4 on a T4")
    p.add_argument("--grad-accum", type=int, default=d.grad_accum)
    p.add_argument("--lr", type=float, default=d.learning_rate)
    p.add_argument("--warmup-ratio", type=float, default=d.warmup_ratio)
    p.add_argument("--max-length", type=int, default=d.max_length)
    p.add_argument("--eval-size", type=int, default=d.eval_size, help="validation items used for eval loss")
    p.add_argument("--eval-steps", type=int, default=d.eval_steps)
    p.add_argument("--save-steps", type=int, default=d.save_steps, help="each save is pushed to the hub as a revision")
    p.add_argument("--lora-r", type=int, default=d.lora.r)
    p.add_argument("--lora-alpha", type=int, default=d.lora.alpha)
    p.add_argument("--lora-dropout", type=float, default=d.lora.dropout)
    p.add_argument("--target-modules", nargs="+", default=d.lora.target_modules)
    p.add_argument("--compute-dtype", default="auto", choices=["auto", "bfloat16", "float16"])
    p.add_argument("--max-steps", type=int, default=-1, help="stop after N optimiser steps (smoke tests)")
    p.add_argument("--train-limit", type=int, help="use only the first N training rows (smoke tests)")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-push", action="store_true", help="keep checkpoints local only")
    p.add_argument("--public", action="store_true", help="make the hub repo public (default private)")
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--output-dir", default=d.output_dir)
    p.add_argument("--dry-run", action="store_true", help="print resolved settings and exit")
    return p


def settings_from_args(args) -> TrainSettings:
    return TrainSettings(
        dataset=args.dataset or config.prompts_dataset(args.lite),
        base_model=args.base_model or config.base_model(),
        hf_user=args.hf_user or config.hf_user() or "",
        run_name=args.run_name or "",
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        eval_size=args.eval_size,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        compute_dtype=args.compute_dtype,
        wandb=not args.no_wandb,
        push_to_hub=not args.no_push,
        private_repo=not args.public,
        output_dir=args.output_dir,
        lora=LoraSettings(args.lora_r, args.lora_alpha, args.lora_dropout, list(args.target_modules)),
    )


def run_training(s: TrainSettings, args) -> dict:
    """The actual QLoRA run. Needs the ``gpu`` extra and a CUDA GPU."""
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi, login
    from peft import LoraConfig
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer

    from price_qlora.quant import footprint_mb, load_base_model

    if not torch.cuda.is_available():
        raise SystemExit("QLoRA (bitsandbytes) needs a CUDA GPU. See README 'Colab / GPU path'.")
    if os.environ.get("HF_TOKEN"):
        login(os.environ["HF_TOKEN"])
    if s.wandb:
        os.environ.setdefault("WANDB_PROJECT", s.project)
        if not os.environ.get("WANDB_API_KEY"):
            print("WANDB_API_KEY not set -> relying on a previous `wandb login`; use --no-wandb to disable tracking")
    set_seed(s.seed)

    ds = load_dataset(s.dataset)
    train = ds["train"]
    if args.train_limit:
        train = train.select(range(min(args.train_limit, len(train))))
    val = ds["val"].select(range(min(s.eval_size, len(ds["val"]))))
    print(f"train {len(train):,} rows, eval {len(val):,} rows; steps/epoch ~ {s.steps_per_epoch(len(train)):,}")

    if s.adapt_to_dataset(len(train)):
        print(f"Short run: eval/save every {s.eval_steps} steps")
    dtype = config.choose_compute_dtype(s.compute_dtype, torch.cuda.is_bf16_supported())
    tokenizer, model = load_base_model(s.base_model, "4bit", dtype)
    print(f"Loaded {s.base_model} in 4-bit NF4 ({dtype} compute): {footprint_mb(model):,.0f} MB")

    sft_config = SFTConfig(**s.sft_kwargs(bf16=(dtype == "bfloat16")), max_steps=args.max_steps)
    trainer = SFTTrainer(
        model=model,
        train_dataset=train,
        eval_dataset=val,
        peft_config=LoraConfig(**s.lora_kwargs()),
        args=sft_config,
        processing_class=tokenizer,
    )
    trainer.model.print_trainable_parameters()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = Path(s.run_dir) / "final"
    trainer.model.save_pretrained(final_dir)  # adapter only (a few tens of MB)
    record = {"settings": s.to_dict(), "adapter_dir": str(final_dir), "train_rows": len(train)}
    if s.push_to_hub:
        trainer.model.push_to_hub(s.hub_id, private=s.private_repo)
        try:
            commits = HfApi().list_repo_commits(s.hub_id)
            record["latest_revision"] = commits[0].commit_id
            print(f"Adapter on the Hub: https://huggingface.co/{s.hub_id} (latest revision {commits[0].commit_id})")
        except Exception as e:  # network / permissions: not fatal
            print("Could not list hub commits:", e)
    (Path(s.run_dir) / "run.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record


def main(argv=None):
    load_dotenv(override=True)
    args = build_parser().parse_args(argv)
    s = settings_from_args(args)
    s.validate()
    print(json.dumps(s.to_dict(), indent=2))
    if args.dry_run:
        print("Dry run: nothing trained.")
        return
    Path(s.run_dir).mkdir(parents=True, exist_ok=True)
    run_training(s, args)


if __name__ == "__main__":
    main()
