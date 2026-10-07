import pytest

from price_dataset.items import Item


def make_item(i=0, price=50.0, category="Electronics", summary=None, weight=1.0, full=None):
    return Item(
        title=f"Thing {i}",
        category=category,
        price=price,
        weight=weight,
        summary=summary if summary is not None else f"Title: Thing {i}\nCategory: {category}\nDescription: a widget number {i}",
        full=full if full is not None else f"Thing {i} full text {i}",
        id=i,
    )


@pytest.fixture
def items():
    return [make_item(i, price=10.0 + i * 7, category=["Electronics", "Automotive", "Appliances"][i % 3]) for i in range(30)]
