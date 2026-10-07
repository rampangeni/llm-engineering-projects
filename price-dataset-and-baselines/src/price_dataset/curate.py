"""Pure-logic curation steps: dedupe, price-weighted sampling, splitting."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from price_dataset.config import (
    FULL_SAMPLE_SIZE,
    FULL_TRAIN,
    FULL_VAL,
    LITE_TEST,
    LITE_TRAIN,
    LITE_VAL,
)
from price_dataset.items import Item

# Down-weight the categories that otherwise dominate the sample
CATEGORY_WEIGHTS = {"Tools_and_Home_Improvement": 0.5, "Automotive": 0.05}


def shuffle_and_dedupe(items: list[Item], seed: int = 42) -> list[Item]:
    """Shuffle (seeded), then drop duplicate titles and duplicate full texts."""
    items = list(items)
    random.seed(seed)
    random.shuffle(items)
    seen: set = set()
    items = [x for x in items if not (x.title in seen or seen.add(x.title))]
    seen = set()
    items = [x for x in items if not (x.full in seen or seen.add(x.full))]
    return items


def sampling_weights(prices, categories, category_weights=None) -> np.ndarray:
    """Probability per item ~ normalised_price**2, scaled per category (favours pricey items)."""
    category_weights = CATEGORY_WEIGHTS if category_weights is None else category_weights
    prices = np.asarray(prices, dtype=float)
    categories = np.asarray(categories)
    p = (prices - prices.min()) / (prices.max() - prices.min() + 1e-9)
    w = p**2
    for cat, factor in category_weights.items():
        w[categories == cat] *= factor
    total = w.sum()
    if total <= 0:
        raise ValueError("All sampling weights are zero; cannot sample")
    return w / total


def weighted_sample(items: list[Item], size: int, seed: int = 42) -> list[Item]:
    """Sample ``size`` items without replacement using :func:`sampling_weights`, then re-shuffle."""
    if size > len(items):
        raise ValueError(f"Requested sample of {size:,} but only {len(items):,} items available")
    np.random.seed(seed)
    w = sampling_weights([i.price for i in items], [i.category for i in items])
    nonzero = int((w > 0).sum())
    if size > nonzero:
        raise ValueError(f"Only {nonzero:,} items have non-zero weight; cannot sample {size:,}")
    idx = np.random.choice(len(items), size=size, replace=False, p=w)
    sample = [items[i] for i in idx]
    random.seed(seed)
    random.shuffle(sample)
    return sample


def split_items(items: list[Item], n_train: int, n_val: int):
    """train = first n_train, val = next n_val, test = the rest."""
    return items[:n_train], items[n_train : n_train + n_val], items[n_train + n_val :]


def lite_subset(train, val, test):
    """The 'lite' dataset: first 20k/1k/1k of the full train/val/test."""
    return train[:LITE_TRAIN], val[:LITE_VAL], test[:LITE_TEST]


def category_counts(items: list[Item]) -> Counter:
    return Counter(i.category for i in items)


def default_split_sizes(sample_size: int) -> tuple[int, int]:
    """Default split for the full sample, proportional otherwise (val = 1.25%, test = 1.25%)."""
    if sample_size == FULL_SAMPLE_SIZE:
        return FULL_TRAIN, FULL_VAL
    n_val = max(int(sample_size * 0.0125), 1)
    n_train = sample_size - 2 * n_val
    return n_train, n_val
