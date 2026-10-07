"""Pydantic data models shared by all agents."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"


class Deal(BaseModel):
    """A deal summarised by the scanner."""

    product_description: str = Field(
        description="Your clearly expressed summary of the product in 3-4 sentences. Details of the item are much more important than why it's a good deal. Avoid mentioning discounts and coupons; focus on the item itself. There should be a short paragraph of text for each item you choose."
    )
    price: float = Field(
        description="The actual price of this product, as advertised in the deal. Be sure to give the actual price; for example, if a deal is described as $100 off the usual $300 price, you should respond with $200"
    )
    url: str = Field(description="The URL of the deal, as provided in the input")


class DealSelection(BaseModel):
    """The structured output the scanner LLM must return."""

    deals: List[Deal] = Field(
        description="Your selection of the 5 deals that have the most detailed, high quality description and the most clear price. You should be confident that the price reflects the deal, that it is a good deal, with a clear description"
    )


class Opportunity(BaseModel):
    """A deal whose estimated true value exceeds its asking price."""

    deal: Deal
    estimate: float
    discount: float


class Item(BaseModel):
    """A product with a known price (row of the HF items dataset built by price-dataset-and-baselines)."""

    title: str = ""
    category: str = ""
    price: float
    full: Optional[str] = None
    weight: Optional[float] = None
    summary: Optional[str] = None
    prompt: Optional[str] = None
    id: Optional[int] = None
