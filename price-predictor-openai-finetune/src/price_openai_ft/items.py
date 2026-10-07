"""Minimal Item model matching the hub schema of ``<user>/items_{full,lite}``.

Columns: title, category, price, full, weight, summary, prompt, id (full/id/prompt may be null).
Only ``summary`` and ``price`` are needed for fine-tuning.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Item(BaseModel):
    title: str
    category: str
    price: float
    full: Optional[str] = None
    weight: Optional[float] = None
    summary: Optional[str] = None
    prompt: Optional[str] = None
    id: Optional[int] = None

    def __repr__(self) -> str:
        return f"<{self.title} = ${self.price}>"

    @classmethod
    def from_hub(cls, dataset_name: str):
        """-> (train, validation, test) lists of Item."""
        from datasets import load_dataset

        ds = load_dataset(dataset_name)
        return tuple([cls.model_validate(r) for r in ds[split]] for split in ("train", "validation", "test"))
