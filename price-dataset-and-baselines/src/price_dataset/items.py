"""The Item data model and HuggingFace Hub (de)serialisation.

Hub schema (every split: ``train`` / ``validation`` / ``test``), one row per Item:
title:str, category:str, price:float, full:str|None, weight:float|None,
summary:str|None, prompt:str|None, id:int|None
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"


class Item(BaseModel):
    """An Item is a data-point of a Product with a Price."""

    title: str
    category: str
    price: float
    full: Optional[str] = None  # scrubbed raw text (dropped before the final push)
    weight: Optional[float] = None  # pounds; 0 when unknown
    summary: Optional[str] = None  # LLM rewrite produced by the summarisation step
    prompt: Optional[str] = None
    id: Optional[int] = None  # temporary index used while batching summaries

    def make_prompt(self, text: str) -> None:
        self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}{round(self.price)}.00"

    def test_prompt(self) -> str:
        """The prompt without the answer, ending in 'Price is $'."""
        return self.prompt.split(PREFIX)[0] + PREFIX

    def __repr__(self) -> str:
        return f"<{self.title} = ${self.price}>"

    @staticmethod
    def push_to_hub(dataset_name: str, train: list["Item"], val: list["Item"], test: list["Item"]):
        """Push Item lists to the HuggingFace Hub as a DatasetDict (train/validation/test)."""
        from datasets import Dataset, DatasetDict

        DatasetDict(
            {
                "train": Dataset.from_list([item.model_dump() for item in train]),
                "validation": Dataset.from_list([item.model_dump() for item in val]),
                "test": Dataset.from_list([item.model_dump() for item in test]),
            }
        ).push_to_hub(dataset_name)

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list["Item"], list["Item"], list["Item"]]:
        """Load (train, validation, test) Items from the Hub."""
        from datasets import load_dataset

        ds = load_dataset(dataset_name)
        return (
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["validation"]],
            [cls.model_validate(row) for row in ds["test"]],
        )
