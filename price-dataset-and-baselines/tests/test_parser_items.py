import json

import pytest

from price_dataset.items import PREFIX, QUESTION, Item
from price_dataset.parser import MAX_TEXT_TOTAL, get_weight, parse, scrub, simplify


def datapoint(price="12.5", details=None, desc=None):
    return {
        "price": price,
        "title": "Acme Widget",
        "description": desc if desc is not None else ["A long description. " * 40],
        "features": ["feature one", "feature two"],
        "details": json.dumps(details if details is not None else {"Item Weight": "8 ounces", "Part Number": "X1"}),
    }


def test_get_weight_units():
    assert get_weight({"Item Weight": "2 pounds"}) == 2
    assert get_weight({"Item Weight": "8 ounces"}) == 0.5
    assert get_weight({"Item Weight": "453.592 grams"}) == pytest.approx(1.0)
    assert get_weight({"Item Weight": "50 hundredths pounds"}) == 0.5
    assert get_weight({}) == 0
    assert get_weight({"Item Weight": "bogus"}) == 0
    assert get_weight({"Item Weight": "3 furlongs"}) == 0


def test_simplify_limits_and_whitespace():
    out = simplify("a\nb\t c\r" + "x" * 5000)
    assert "\n" not in out and len(out) <= 3000


def test_scrub_removes_noise_and_part_numbers():
    details = {"Part Number": "ABC1234", "Brand": "Acme", "Best Sellers Rank": "1"}
    out = scrub("Title", ["desc ABCD1234 here"], ["feat"], details)
    assert "Part Number" not in out and "Best Sellers Rank" not in out
    assert "ABCD1234" not in out  # mixed upper+digit 8 chars removed
    assert "Brand" in out and len(out) <= MAX_TEXT_TOTAL


def test_parse_ok_and_weight():
    item = parse(datapoint(), "Appliances")
    assert item is not None
    assert item.price == 12.5 and item.category == "Appliances"
    assert item.weight == 0.5
    assert len(item.full) >= 600


@pytest.mark.parametrize("price", ["None", "0.10", "1000", "abc"])
def test_parse_rejects_bad_prices(price):
    assert parse(datapoint(price=price), "X") is None


def test_parse_rejects_short_text():
    assert parse(datapoint(desc=["tiny"]), "X") is None


def test_make_prompt_and_test_prompt():
    item = Item(title="t", category="c", price=12.6)
    item.make_prompt("SUMMARY")
    assert item.prompt == f"{QUESTION}\n\nSUMMARY\n\n{PREFIX}13.00"
    assert item.test_prompt().endswith(PREFIX)
    assert "13.00" not in item.test_prompt()


def test_item_ignores_extra_hub_columns():
    row = {"title": "t", "category": "c", "price": 1.0, "unexpected": 5}
    assert Item.model_validate(row).title == "t"
