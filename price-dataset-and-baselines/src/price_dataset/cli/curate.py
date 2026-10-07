"""Build the raw items dataset and push it to the Hub.

    curate-items --categories Appliances --sample-size 2000   # quick local try-out, no push
    curate-items --push                                         # the full 820k run (hours, ~tens of GB download)
"""

from __future__ import annotations

from price_dataset import config
from price_dataset.cli.common import base_parser
from price_dataset.curate import (
    category_counts,
    default_split_sizes,
    lite_subset,
    shuffle_and_dedupe,
    split_items,
    weighted_sample,
)
from price_dataset.items import Item


def main(argv=None):
    p = base_parser("Curate the Amazon items dataset")
    p.add_argument("--categories", nargs="*", default=config.CATEGORIES, help="Amazon-Reviews-2023 meta categories")
    p.add_argument("--sample-size", type=int, default=config.FULL_SAMPLE_SIZE, help="price-weighted sample size (default: 820,000)")
    p.add_argument("--workers", type=int, default=None, help="parser processes (default cpu-1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--push", action="store_true", help="push to <HF_USERNAME>/items_raw_{full,lite}")
    p.add_argument("--no-lite", action="store_true", help="when pushing the full set, skip the lite subset")
    args = p.parse_args(argv)

    from price_dataset.loaders import WORKERS, ItemLoader

    items: list[Item] = []
    for cat in args.categories:
        items.extend(ItemLoader(cat).load(args.workers or WORKERS))
    print(f"A grand total of {len(items):,} items")
    items = shuffle_and_dedupe(items, args.seed)
    print(f"After deduplication, we have {len(items):,} items")

    size = min(args.sample_size, len(items))
    if size < args.sample_size:
        print(f"WARNING: only {len(items):,} items available; sampling {size:,} instead of {args.sample_size:,}")
    sample = weighted_sample(items, size, args.seed)
    print("Category counts:", dict(category_counts(sample)))

    n_train, n_val = default_split_sizes(size)
    train, val, test = split_items(sample, n_train, n_val)
    print(f"Split: {len(train):,} train / {len(val):,} val / {len(test):,} test")

    if not args.push:
        print("Dry run (no --push): nothing uploaded.")
        return
    full_name = config.dataset_name("items_raw", lite=False)
    config.hf_login()
    Item.push_to_hub(full_name, train, val, test)
    if not args.no_lite:
        Item.push_to_hub(config.dataset_name("items_raw", lite=True), *lite_subset(train, val, test))
    print("Pushed", full_name)


if __name__ == "__main__":
    main()
