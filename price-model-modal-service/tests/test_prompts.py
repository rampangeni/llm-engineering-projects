import pytest

from price_model_service.prompts import PREFIX, build_prompt, extract_price


def test_build_prompt_matches_training_format():
    p = build_prompt("a mic")
    assert p == "What does this cost to the nearest dollar?\n\na mic\n\nPrice is $"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("99.00", 99.0),
        ("1,299.50 and more", 1299.5),
        (" 45", 45.0),
        ("garbage", 0.0),
        ("", 0.0),
    ],
)
def test_extract_price_continuation(text, expected):
    assert extract_price(text) == expected


def test_extract_price_full_decode_uses_last_prefix():
    decoded = f"Q\n\nthe item {PREFIX}5 off\n\n{PREFIX}120.00<|eot|>"
    assert extract_price(decoded) == 120.0
