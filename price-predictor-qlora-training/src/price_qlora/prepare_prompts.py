"""Truncate summaries to a token budget, build prompt/completion pairs, push to the Hub.

    qlora-prepare-prompts --lite --dry-run          # only print token statistics
    qlora-prepare-prompts --lite --push             # -> <HF_USERNAME>/items_prompts_lite
    qlora-prepare-prompts --push                    # full: 800k/10k/10k
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from price_qlora import config
from price_qlora.items import Item
from price_qlora.prompts import build_prompt_splits, push_prompts_to_hub, truncation_stats


def main(argv=None):
    load_dotenv(override=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lite", action="store_true")
    p.add_argument("--source", help="summarised items dataset (default <DATASET_USERNAME>/items_{full,lite})")
    p.add_argument("--target", help="where to push (default <HF_USERNAME>/items_prompts_{full,lite})")
    p.add_argument("--base-model", default=None, help="tokenizer to count tokens with (default BASE_MODEL)")
    p.add_argument("--cutoff", type=int, default=config.CUTOFF, help="max summary tokens (default: 110)")
    p.add_argument("--push", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print statistics, build prompts, push nothing")
    args = p.parse_args(argv)

    from huggingface_hub import login
    from transformers import AutoTokenizer

    import os

    if os.environ.get("HF_TOKEN"):
        login(os.environ["HF_TOKEN"])
    source = args.source or config.items_dataset(args.lite)
    train, val, test = Item.from_hub(source)
    print(f"Loaded {len(train):,} train / {len(val):,} val / {len(test):,} test items from {source}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model or config.base_model())
    stats = truncation_stats(train + val + test, tokenizer, args.cutoff)
    print(
        f"Summary tokens: avg {stats['avg_tokens']:.1f}, max {stats['max_tokens']}; "
        f"cutoff {args.cutoff} truncates {stats['truncated']:,} items ({stats['truncated_fraction']:.1%})"
    )
    build_prompt_splits(train, val, test, tokenizer, args.cutoff)
    print("PROMPT:\n" + test[0].prompt + "\nCOMPLETION:\n" + test[0].completion)
    longest = max(it.count_prompt_tokens(tokenizer) for it in train[:5000] + test[:1000])
    print(f"Longest prompt+completion in a sample: {longest} tokens (SFT max_length default {config.MAX_SEQ_LENGTH})")

    if args.push and not args.dry_run:
        target = args.target or config.prompts_dataset(args.lite, user=config.hf_user())
        if not (args.target or config.hf_user()):
            raise SystemExit("Set HF_USERNAME or pass --target")
        push_prompts_to_hub(target, train, val, test)
        print("Pushed", target)


if __name__ == "__main__":
    main()
