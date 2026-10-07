"""Turn model output into a number."""

from __future__ import annotations

import re

from price_qlora.config import PREFIX

_NUMBER = re.compile(r"[-+]?\d*\.\d+|\d+")


def extract_price(text: str) -> float:
    """'... Price is $1,234.00<eos>' -> 1234.0 ; returns 0 if no number is found.

    Works on the generated continuation alone (``'45.00'``) or on prompt+continuation
    (anything before the last 'Price is $' is ignored).
    """
    if PREFIX in text:
        text = text.rsplit(PREFIX, 1)[1]
    text = text.replace(",", "")
    match = _NUMBER.search(text)
    return float(match.group()) if match else 0.0
