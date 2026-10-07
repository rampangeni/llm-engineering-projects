"""Rewrite every item's text into a short standard summary and push the final dataset.

    summarize-items submit --lite        # upload + submit Groq batches (state in work/lite/)
    summarize-items fetch  --lite        # poll / collect finished batches (re-run until done)
    summarize-items push   --lite        # drop full/id and push <HF_USERNAME>/items_lite
    summarize-items direct --lite --limit 20   # no batch API: synchronous litellm calls (any provider, e.g. ollama)
"""

from __future__ import annotations

import time
from pathlib import Path

from price_dataset import config
from price_dataset.batch import BatchRunner, missing_summaries
from price_dataset.cli.common import base_parser, load_items
from price_dataset.items import Item


def assign_ids(items):
    for index, item in enumerate(items):
        item.id = index


def split_for_push(items, lite: bool):
    """Re-split the flat list [train + val + test] exactly as the summarisation step does."""
    if lite:
        return items[:20_000], items[20_000:21_000], items[21_000:]
    return items[:800_000], items[800_000:810_000], items[810_000:]


def finalize(items, lite: bool, dataset_name: str):
    for item in items:
        item.full = None
        item.id = None
    train, val, test = split_for_push(items, lite)
    Item.push_to_hub(dataset_name, train, val, test)
    if not lite:
        Item.push_to_hub(config.dataset_name("items", True), train[:20_000], val[:1_000], test[:1_000])


def main(argv=None):
    p = base_parser("Summarise items with an LLM")
    p.add_argument("command", choices=["submit", "fetch", "push", "direct"])
    p.add_argument("--work-dir", default="work", help="where batch files / state live")
    p.add_argument("--wait", action="store_true", help="fetch: keep polling until all batches complete")
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--limit", type=int, help="direct: only summarise this many items")
    p.add_argument("--model", help="direct: litellm model (default PREPROCESSOR_MODEL)")
    args = p.parse_args(argv)

    lite = args.lite
    train, val, test = load_items(args, kind="items_raw")
    items = train + val + test
    assign_ids(items)
    folder = Path(args.work_dir) / ("lite" if lite else "full")
    runner = BatchRunner(items, folder)

    if args.command == "submit":
        runner.create()
        runner.run()
    elif args.command == "fetch":
        runner.load()
        runner.apply_finished_outputs()
        while True:
            finished, total = runner.fetch()
            if finished == total or not args.wait:
                break
            time.sleep(args.poll_seconds)
        print(f"{len(missing_summaries(items)):,} items still lack a summary")
    elif args.command == "direct":
        from price_dataset.preprocessor import Preprocessor

        pre = Preprocessor(args.model) if args.model else Preprocessor()
        for item in items[: args.limit]:
            item.summary = pre.preprocess(item.full)
        print(f"tokens in/out: {pre.total_input_tokens}/{pre.total_output_tokens}  cost ${pre.total_cost:.4f}")
    elif args.command == "push":
        runner.load()
        runner.apply_finished_outputs()
        missing = missing_summaries(items)
        if missing:
            raise SystemExit(f"{len(missing):,} items have no summary yet (first: {missing[:5]}); run `fetch` first")
        finalize(items, lite, config.dataset_name("items", lite))
        print("Pushed", config.dataset_name("items", lite))


if __name__ == "__main__":
    main()
