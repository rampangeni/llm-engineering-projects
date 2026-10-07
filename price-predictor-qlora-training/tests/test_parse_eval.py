import pytest

from price_qlora.evaluator import Tester, color_for, compute_metrics, title_of
from price_qlora.parse import extract_price


@pytest.mark.parametrize(
    "text,expected",
    [
        ("45.00", 45.0),
        (" 1,234.00<|end_of_text|>", 1234.0),
        ("Price is $67.50\n", 67.5),
        ("blah Price is $12 more text Price is $99.00", 99.0),  # last prefix wins (continuation)
        ("no digits", 0.0),
        ("", 0.0),
    ],
)
def test_extract_price(text, expected):
    assert extract_price(text) == expected


def test_color_and_metrics():
    assert color_for(10, 500) == "green" and color_for(60, 200) == "orange" and color_for(300, 400) == "red"
    m = compute_metrics([100.0, 100.0, 100.0], [100.0, 150.0, 500.0])
    assert (m.green, m.orange, m.red) == (1, 1, 1)
    assert m.average_error == pytest.approx((0 + 50 + 400) / 3)
    assert m.hit_rate == pytest.approx(1 / 3)
    assert m.rmsle > 0


def test_perfect_prediction_has_zero_error():
    m = compute_metrics([5.0, 10.0], [5.0, 10.0])
    assert m.average_error == 0 and m.mse == 0 and m.rmsle == 0 and m.r2_percent == 100


def test_title_of():
    assert title_of("Q\n\nTitle: Widget Pro\nCategory: x") == "Widget Pro"
    assert title_of("no title here").startswith("no title")
    assert title_of("Title: " + "x" * 60).endswith("...")


def test_tester_parses_generated_text(tmp_path):
    data = [{"prompt": f"Title: T{i}\n\nPrice is $", "completion": str(float(10 * (i + 1)))} for i in range(4)]
    t = Tester(lambda prompt: "20.00", data, size=3, quiet=True)
    m = t.run()
    assert m.n == 3 and t.guesses == [20.0, 20.0, 20.0] and t.truths == [10.0, 20.0, 30.0]
    t.dump(str(tmp_path / "o.json"))
    assert (tmp_path / "o.json").exists()
    assert t.rows()[0]["title"] == "T0"
