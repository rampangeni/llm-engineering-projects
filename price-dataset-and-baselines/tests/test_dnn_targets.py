import math

import pytest

from price_dataset.dnn.targets import DEFAULT_Y_MEAN, DEFAULT_Y_STD, TargetScaler


def test_roundtrip():
    prices = [1.0, 10.0, 25.0, 99.0, 450.0]
    s = TargetScaler.fit(prices)
    for p in prices:
        assert s.decode(s.encode(p)) == pytest.approx(p)


def test_fit_is_standardised_log_space():
    prices = [5.0, 20.0, 80.0, 300.0]
    s = TargetScaler.fit(prices)
    enc = [s.encode(p) for p in prices]
    assert sum(enc) / len(enc) == pytest.approx(0, abs=1e-9)
    assert math.sqrt(sum(e * e for e in enc) / (len(enc) - 1)) == pytest.approx(1)  # unbiased std like torch


def test_decode_clamps_to_zero():
    assert TargetScaler(DEFAULT_Y_MEAN, DEFAULT_Y_STD).decode(-50) == 0


def test_default_constants_give_plausible_prices():
    s = TargetScaler(DEFAULT_Y_MEAN, DEFAULT_Y_STD)
    assert 70 < s.decode(0) < 90  # exp(4.43)-1 ≈ 83
