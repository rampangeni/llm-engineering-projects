from __future__ import annotations

import os

DEFAULT_USERNAME = "rampangeni"  # private pre-built datasets: needs `hf auth login` or HF_TOKEN
BASE_MODEL = "gpt-4.1-nano-2025-04-14"  # snapshot that supports fine-tuning
BASE_MODEL_ALIAS = "gpt-4.1-nano"  # zero-shot baseline model
SUFFIX = "pricer"
SEED = 42
DEFAULT_TRAIN = 100  # 100 train examples ...
DEFAULT_VAL = 50  # ... 50 validation examples
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 1
MAX_TOKENS = 7  # "$123.45" fits; evaluation caps the completion at 7 tokens
MIN_EXAMPLES = 10  # OpenAI's minimum for a fine-tuning job


def dataset_name(lite: bool = False, user: str | None = None) -> str:
    """The summarised dataset produced by price-dataset-and-baselines."""
    user = user or os.environ.get("HF_USERNAME", DEFAULT_USERNAME)
    return f"{user}/items_{'lite' if lite else 'full'}"


def hf_login() -> None:
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login

        login(token)
