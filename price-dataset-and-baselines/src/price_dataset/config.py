"""Central configuration: dataset repo names, defaults and HF login helper.

Everything that names a repo on the HuggingFace Hub is derived from ``HF_USERNAME``
so downstream projects (OpenAI fine-tune, QLoRA training, week-8 services) can point
at the same artifacts by setting one environment variable.
"""

from __future__ import annotations

import os

DEFAULT_USERNAME = "rampangeni"  # private pre-built datasets: needs `hf auth login` or HF_TOKEN

# The eight Amazon-Reviews-2023 meta categories used here.
CATEGORIES = [
    "Automotive",
    "Electronics",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Cell_Phones_and_Accessories",
    "Toys_and_Games",
    "Appliances",
    "Musical_Instruments",
]

# Default dataset sizes
FULL_SAMPLE_SIZE = 820_000
FULL_TRAIN, FULL_VAL = 800_000, 10_000  # remaining 10,000 are the test set
LITE_TRAIN, LITE_VAL, LITE_TEST = 20_000, 1_000, 1_000

KINDS = ("items_raw", "items")  # raw = no summaries; items = with LLM summaries


def username() -> str:
    return os.environ.get("HF_USERNAME", DEFAULT_USERNAME)


def dataset_name(kind: str, lite: bool, user: str | None = None) -> str:
    """e.g. dataset_name('items', True) -> 'rampangeni/items_lite'.

    kind='items_raw'  -> <user>/items_raw_{full|lite}  (no summary)
    kind='items'      -> <user>/items_{full|lite}      (with summary)
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return f"{user or username()}/{kind}_{'lite' if lite else 'full'}"


def hf_login() -> None:
    """Log in to the Hub if HF_TOKEN is set (no-op otherwise)."""
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login

        login(token)
