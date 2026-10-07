from __future__ import annotations

import argparse

from dotenv import load_dotenv

from price_dataset import config
from price_dataset.items import Item


def base_parser(description: str) -> argparse.ArgumentParser:
    load_dotenv(override=True)
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--lite", action="store_true", help="use the lite dataset (20k train) instead of full (800k)")
    p.add_argument("--dataset", help="explicit hub dataset (overrides --lite and HF_USERNAME), e.g. rampangeni/items_full")
    return p


def resolve_dataset(args, kind: str = "items") -> str:
    return args.dataset or config.dataset_name(kind, args.lite)


def load_items(args, kind: str = "items"):
    name = resolve_dataset(args, kind)
    config.hf_login()
    train, val, test = Item.from_hub(name)
    print(f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items from {name}")
    return train, val, test
