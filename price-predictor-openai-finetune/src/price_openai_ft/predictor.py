"""Inference functions (``predictor(item) -> str``) for a base or fine-tuned chat model."""

from __future__ import annotations

from price_openai_ft import config
from price_openai_ft.prompts import test_messages_for


def make_predictor(client, model: str, max_tokens: int = config.MAX_TOKENS):
    def predictor(item) -> str:
        response = client.chat.completions.create(
            model=model, messages=test_messages_for(item), max_tokens=max_tokens
        )
        return response.choices[0].message.content

    predictor.__name__ = model.replace(":", "_").replace("-", "_")
    return predictor
