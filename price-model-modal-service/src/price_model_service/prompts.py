"""Prompt format and answer parsing. Must match the format used at training time
(see ../price-predictor-qlora-training): the model completes ``Price is $<number>``."""

from __future__ import annotations

import re

PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"

_NUMBER = re.compile(r"[-+]?\d*\.\d+|\d+")


def build_prompt(description: str) -> str:
    return f"{QUESTION}\n\n{description}\n\n{PREFIX}"


def extract_price(text: str) -> float:
    """Pull the first number out of the model's continuation. Returns 0.0 if none found.

    Accepts either just the generated continuation or a full decoded sequence that still
    contains the prompt (in which case the text after the last ``Price is $`` is used).
    """
    if PREFIX in text:
        text = text.rsplit(PREFIX, 1)[1]
    text = text.replace(",", "").replace("$", "")
    match = _NUMBER.search(text)
    return float(match.group()) if match else 0.0
