"""Item with the prompt/completion helpers."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from price_qlora.config import PREFIX, QUESTION


class Item(BaseModel):
    """A product with a price. Hub schema of ``<user>/items_{full,lite}``; extra columns are ignored."""

    title: str
    category: str
    price: float
    full: Optional[str] = None
    weight: Optional[float] = None
    summary: Optional[str] = None
    prompt: Optional[str] = None
    completion: Optional[str] = None
    id: Optional[int] = None

    def __repr__(self) -> str:
        return f"<{self.title} = ${self.price}>"

    @classmethod
    def from_hub(cls, dataset_name: str):
        """-> (train, validation, test) lists of Item."""
        from datasets import load_dataset

        ds = load_dataset(dataset_name)
        return tuple([cls.model_validate(r) for r in ds[s]] for s in ("train", "validation", "test"))

    # ---- tokens / prompts -------------------------------------------------
    def truncated_summary(self, tokenizer, max_tokens: int) -> str:
        tokens = tokenizer.encode(self.summary, add_special_tokens=False)
        if len(tokens) > max_tokens:
            return tokenizer.decode(tokens[:max_tokens]).rstrip()
        return self.summary

    def count_tokens(self, tokenizer) -> int:
        """Tokens in the summary."""
        return len(tokenizer.encode(self.summary, add_special_tokens=False))

    def make_prompts(self, tokenizer, max_tokens: int, do_round: bool) -> None:
        """prompt = QUESTION + (summary truncated to max_tokens) + 'Price is $'.

        completion = '<rounded>.00' (train/val, ``do_round=True``) or the exact price string (test).
        """
        summary = self.truncated_summary(tokenizer, max_tokens)
        self.prompt = f"{QUESTION}\n\n{summary}\n\n{PREFIX}"
        self.completion = f"{round(self.price)}.00" if do_round else str(self.price)

    def count_prompt_tokens(self, tokenizer) -> int:
        return len(tokenizer.encode(self.prompt + self.completion, add_special_tokens=False))

    def to_datapoint(self) -> dict:
        return {"prompt": self.prompt, "completion": self.completion}
