import math

import pytest

from conftest import make_item
from price_dataset.evaluator import (
    Tester,
    color_for,
    compute_metrics,
    evaluate,
    post_process,
    r2_score,
    rmsle,
    running_mean_ci,
)


@pytest.mark.parametrize(
    "raw,expected",
    [("$1,299.00", 1299.0), ("Price is $45", 45.0), ("about 12.5 dollars", 12.5), ("no idea", 0), (7, 7)],
)
def test_post_process(raw, expected):
    assert post_process(raw) == expected


@pytest.mark.parametrize(
    "error,truth,color",
    [
        (39, 400, "green"),  # < $40
        (10, 30, "green"),
        (50, 200, "orange"),  # 25% relative, < $80
        (90, 400, "orange"),  # 22.5% relative, >= $80 -> still < 40%
        (90, 100, "red"),
    ],
)
def test_color_thresholds(error, truth, color):
    assert color_for(error, truth) == color


def test_color_relative_green():
    assert color_for(100, 1000) == "green"  # 10% relative


def test_rmsle_and_r2():
    assert rmsle([10, 20], [10, 20]) == 0
    assert rmsle([0], [e - 1 for e in [math.e]]) == pytest.approx(1.0)
    assert r2_score([1, 2, 3], [1, 2, 3]) == 1
    assert r2_score([1, 2, 3], [2, 2, 2]) == 0  # predicting the mean


def test_compute_metrics():
    truths = [100, 100, 100, 100]
    guesses = [100, 130, 170, 500]  # errors 0,30,70,400 -> green,green,orange(70<80),red
    m = compute_metrics(truths, guesses)
    assert m.n == 4 and (m.green, m.orange, m.red) == (2, 1, 1)
    assert m.average_error == pytest.approx(125)
    assert m.hit_rate == 0.5
    assert m.mse == pytest.approx((0 + 900 + 4900 + 160000) / 4)
    assert "hit rate 50.0%" in m.summary_line("x")


def test_running_mean_ci():
    means, ci = running_mean_ci([10, 20, 30])
    assert means == [10, 15, 20]
    assert ci[0] == 0 and ci[2] > 0


def test_tester_runs_with_strings_and_threads():
    data = [make_item(i, price=100.0) for i in range(10)]
    t = Tester(lambda item: "$110.00", data, size=5, workers=2, quiet=True, title="const")
    m = t.run()
    assert m.n == 5 and m.average_error == pytest.approx(10)
    assert t.titles[0].startswith("Thing")
    assert t.colors == ["green"] * 5


def test_tester_size_clamped_and_custom_accessors():
    data = [{"prompt": "Title: Foo\n", "completion": "20"}]
    m = evaluate(
        lambda d: "21", data, size=50, quiet=True,
        truth_fn=lambda d: float(d["completion"]), title_fn=lambda d: "Foo",
    )
    assert m.n == 1 and m.average_error == 1
