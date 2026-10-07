"""Turn a raw Amazon-Reviews-2023 metadata row into a clean Item."""

from __future__ import annotations

import json
import re

from price_dataset.items import Item

MIN_CHARS = 600
MIN_PRICE = 0.5
MAX_PRICE = 999.49
MAX_TEXT_EACH = 3000
MAX_TEXT_TOTAL = 4000

REMOVALS = [
    "Part Number",
    "Best Sellers Rank",
    "Batteries Included?",
    "Batteries Required?",
    "Item model number",
]

# 7+ char tokens that mix capitals and digits = part / model numbers
_PART_NUMBER = re.compile(r"\b(?=[A-Z0-9]{7,}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+\b")


def simplify(text_list) -> str:
    """Return a string without excess whitespace, limited to MAX_TEXT_EACH characters."""
    return (
        str(text_list)
        .replace("\n", " ")
        .replace("\r", "")
        .replace("\t", "")
        .replace("  ", " ")
        .strip()[:MAX_TEXT_EACH]
    )


def scrub(title, description, features, details) -> str:
    """Return a cleansed full text with product numbers and unimportant details removed."""
    for remove in REMOVALS:
        details.pop(remove, None)
    result = title + "\n"
    if description:
        result += simplify(description) + "\n"
    if features:
        result += simplify(features) + "\n"
    if details:
        result += json.dumps(details) + "\n"
    return _PART_NUMBER.sub("", result).strip()[:MAX_TEXT_TOTAL]


def get_weight(details) -> float:
    """Return the item weight in pounds, or 0 if absent / unparseable."""
    weight_str = details.get("Item Weight")
    if weight_str:
        parts = weight_str.split(" ")
        try:
            amount = float(parts[0])
            unit = parts[1].lower()
            if unit == "pounds":
                return amount
            elif unit == "ounces":
                return amount / 16
            elif unit == "grams":
                return amount / 453.592
            elif unit == "milligrams":
                return amount / 453592
            elif unit == "kilograms":
                return amount / 0.453592
            elif unit == "hundredths" and parts[2].lower() == "pounds":
                return amount / 100
        except (ValueError, IndexError):
            return 0
    return 0


def parse(datapoint, category) -> Item | None:
    """Return an Item, or None if the price is missing/out of range or text is too short."""
    try:
        price = float(datapoint["price"])
    except (ValueError, TypeError):
        return None
    if MIN_PRICE <= price <= MAX_PRICE:
        title = datapoint["title"]
        description = datapoint["description"]
        features = datapoint["features"]
        details = json.loads(datapoint["details"])
        weight = get_weight(details)
        full = scrub(title, description, features, details)
        if len(full) >= MIN_CHARS:
            return Item(title=title, category=category, price=price, full=full, weight=weight)
    return None
