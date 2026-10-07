"""Build the prompt/completion datasets and prompts for inference."""

from __future__ import annotations

from price_qlora.config import CUTOFF, PREFIX, QUESTION
from price_qlora.items import Item


def make_inference_prompt(summary: str, tokenizer=None, cutoff: int = CUTOFF) -> str:
    """Prompt for one description; truncates to ``cutoff`` tokens if a tokenizer is given."""
    if tokenizer is not None:
        tokens = tokenizer.encode(summary, add_special_tokens=False)
        if len(tokens) > cutoff:
            summary = tokenizer.decode(tokens[:cutoff]).rstrip()
    return f"{QUESTION}\n\n{summary}\n\n{PREFIX}"


def build_prompt_splits(train, val, test, tokenizer, cutoff: int = CUTOFF):
    """Mutates items in place: train/val get rounded completions, test keeps the exact price."""
    for item in train + val:
        item.make_prompts(tokenizer, cutoff, True)
    for item in test:
        item.make_prompts(tokenizer, cutoff, False)
    return train, val, test


def truncation_stats(items, tokenizer, cutoff: int = CUTOFF) -> dict:
    counts = [it.count_tokens(tokenizer) for it in items]
    cut = sum(1 for c in counts if c > cutoff)
    return {
        "n": len(counts),
        "avg_tokens": sum(counts) / len(counts),
        "max_tokens": max(counts),
        "truncated": cut,
        "truncated_fraction": cut / len(counts),
    }


def push_prompts_to_hub(dataset_name: str, train, val, test) -> None:
    """DatasetDict with splits ``train`` / ``val`` / ``test`` and columns ``prompt``, ``completion``."""
    from datasets import Dataset, DatasetDict

    DatasetDict(
        {
            "train": Dataset.from_list([i.to_datapoint() for i in train]),
            "val": Dataset.from_list([i.to_datapoint() for i in val]),
            "test": Dataset.from_list([i.to_datapoint() for i in test]),
        }
    ).push_to_hub(dataset_name)
