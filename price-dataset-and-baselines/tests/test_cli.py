import pytest

from price_dataset.cli import common


def test_base_parser_and_resolve(monkeypatch):
    monkeypatch.setenv("HF_USERNAME", "me")
    p = common.base_parser("x")
    assert common.resolve_dataset(p.parse_args([])) == "me/items_full"
    assert common.resolve_dataset(p.parse_args(["--lite"])) == "me/items_lite"
    assert common.resolve_dataset(p.parse_args(["--lite"]), "items_raw") == "me/items_raw_lite"
    assert common.resolve_dataset(p.parse_args(["--dataset", "a/b"])) == "a/b"


def test_summarize_split_for_push():
    from price_dataset.cli.summarize import split_for_push

    items = list(range(22_000))
    tr, va, te = split_for_push(items, lite=True)
    assert (len(tr), len(va), len(te)) == (20_000, 1_000, 1_000)
