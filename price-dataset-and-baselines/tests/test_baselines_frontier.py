import csv
from types import SimpleNamespace

import pytest

from conftest import make_item
from price_dataset.baselines import (
    export_human_csv,
    get_features,
    make_constant_pricer,
    make_human_pricer,
    make_random_pricer,
    read_human_csv,
)
from price_dataset.frontier import FRONTIER_MODELS, make_frontier_pricer, messages_for


def test_constant_pricer_is_training_mean(items):
    p = make_constant_pricer(items)
    assert p.average == pytest.approx(sum(i.price for i in items) / len(items))
    assert p(items[0]) == p.average


def test_random_pricer_seeded_and_in_range(items):
    a, b = make_random_pricer(1), make_random_pricer(1)
    seq_a = [a(items[0]) for _ in range(20)]
    assert seq_a == [b(items[0]) for _ in range(20)]
    assert all(1 <= v < 1000 for v in seq_a)


def test_get_features():
    f = get_features(make_item(summary="abcd", weight=0))
    assert f == {"weight": 0, "weight_unknown": 1, "text_length": 4}
    assert get_features(make_item(summary="ab", weight=2.5)) == {"weight": 2.5, "weight_unknown": 0, "text_length": 2}


def test_human_csv_roundtrip(tmp_path, items):
    path = tmp_path / "in.csv"
    export_human_csv(items, path, n=5)
    rows = list(csv.reader(open(path, encoding="utf-8")))
    assert len(rows) == 5 and rows[0][1] == "0"
    # multi-line summaries survive csv quoting
    assert "\n" in rows[0][0]
    out = tmp_path / "out.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for i, r in enumerate(rows):
            w.writerow([r[0], 10 * (i + 1)])
    preds = read_human_csv(out)
    pricer = make_human_pricer(items, preds)
    assert [pricer(items[i]) for i in range(5)] == [10, 20, 30, 40, 50]


def test_bundled_human_file_is_parseable():
    from pathlib import Path

    preds = read_human_csv(Path(__file__).parent.parent / "data" / "human_guesses.csv")
    assert len(preds) == 100 and all(p >= 0 for p in preds)


def test_frontier_messages_and_pricer():
    item = make_item(summary="SUMMARY TEXT")
    msgs = messages_for(item)
    assert msgs[0]["role"] == "user" and msgs[0]["content"].endswith("SUMMARY TEXT")
    assert "no explanation" in msgs[0]["content"]

    calls = []

    def fake_completion(**kw):
        calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="$42"))])

    pricer = make_frontier_pricer("gpt-5.1", completion_fn=fake_completion)
    assert pricer(item) == "$42"
    assert calls[0]["model"] == "gpt-5.1" and calls[0]["reasoning_effort"] == "high" and calls[0]["seed"] == 42
    assert "gpt-5.1" in FRONTIER_MODELS


def test_sklearn_baselines_smoke():
    pytest.importorskip("sklearn")
    from price_dataset.baselines import (
        make_bow_linear_pricer,
        make_bow_vectorizer,
        make_linear_regression_pricer,
        make_random_forest_pricer,
    )

    train = [make_item(i, price=10.0 + 3 * i, summary=f"widget{i % 7} gadget thing{i % 5}", weight=float(i % 4)) for i in range(60)]
    assert isinstance(make_linear_regression_pricer(train)(train[0]), float)
    vec, X = make_bow_vectorizer(train, max_features=50)
    assert make_bow_linear_pricer(vec, X, train)(train[3]) >= 0
    assert make_random_forest_pricer(vec, X, train, subset=40, n_jobs=1)(train[3]) >= 0
