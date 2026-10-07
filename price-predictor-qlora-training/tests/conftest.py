import pytest

from price_qlora.items import Item


class FakeTokenizer:
    """Whitespace tokenizer: 1 word = 1 token. encode/decode round-trip on single-space text."""

    def encode(self, text, add_special_tokens=False):
        return text.split(" ")

    def decode(self, tokens):
        return " ".join(tokens)


@pytest.fixture
def tok():
    return FakeTokenizer()


def make_item(i=0, price=49.6, words=5):
    return Item(title=f"t{i}", category="c", price=price, summary=" ".join(f"w{j}" for j in range(words)))
