import numpy as np
import pytest

from conftest import make_item
from price_dataset import config
from price_dataset.curate import (
    default_split_sizes,
    lite_subset,
    sampling_weights,
    shuffle_and_dedupe,
    split_items,
    weighted_sample,
)


def test_dedupe_titles_and_full():
    a = make_item(1, full="same text")
    b = make_item(2, full="same text")  # different title, same full
    c = make_item(1, full="other")  # same title as a
    d = make_item(3, full="unique")
    out = shuffle_and_dedupe([a, b, c, d], seed=1)
    assert len(out) == 2 or len(out) == 3  # depends on shuffle order, never all 4
    assert len({x.title for x in out}) == len(out)
    assert len({x.full for x in out}) == len(out)


def test_dedupe_is_deterministic():
    items = [make_item(i, full=f"f{i}") for i in range(20)]
    assert [x.title for x in shuffle_and_dedupe(items, 7)] == [x.title for x in shuffle_and_dedupe(items, 7)]


def test_weights_favour_expensive_and_downweight_categories():
    w = sampling_weights([1, 100, 1000, 1000], ["A", "A", "Automotive", "B"])
    assert w.sum() == pytest.approx(1)
    assert w[3] > w[1] > w[0]
    assert w[2] == pytest.approx(w[3] * 0.05)


def test_weighted_sample_properties():
    items = [make_item(i, price=1 + i) for i in range(200)]
    s1 = weighted_sample(items, 50, seed=3)
    s2 = weighted_sample(items, 50, seed=3)
    assert [i.title for i in s1] == [i.title for i in s2]
    assert len({i.title for i in s1}) == 50


def test_weighted_sample_too_big():
    with pytest.raises(ValueError):
        weighted_sample([make_item(i) for i in range(5)], 10)


def test_split_and_lite():
    items = [make_item(i) for i in range(100)]
    tr, va, te = split_items(items, 80, 10)
    assert (len(tr), len(va), len(te)) == (80, 10, 10)
    big = list(range(50_000))
    ltr, lva, lte = lite_subset(big, big[:5000], big[:5000])
    assert (len(ltr), len(lva), len(lte)) == (20_000, 1_000, 1_000)


def test_default_split_sizes():
    assert default_split_sizes(config.FULL_SAMPLE_SIZE) == (800_000, 10_000)
    n_train, n_val = default_split_sizes(1000)
    assert n_train + 2 * n_val == 1000


def test_dataset_names(monkeypatch):
    monkeypatch.setenv("HF_USERNAME", "me")
    assert config.dataset_name("items", True) == "me/items_lite"
    assert config.dataset_name("items_raw", False) == "me/items_raw_full"
    with pytest.raises(ValueError):
        config.dataset_name("nope", True)
