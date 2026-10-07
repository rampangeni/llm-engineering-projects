"""Rewrite a messy product blurb into a short, consistent description before pricing.

The fine-tuned model was trained on LLM-summarised descriptions (Title/Category/Brand/
Description/Details), so feeding it the same shape at inference time improves accuracy.
Backed by litellm, so any provider works (local Ollama by default).
"""

from __future__ import annotations

import os

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


def default_model_name() -> str:
    return os.getenv("PRICER_PREPROCESSOR_MODEL", "ollama/llama3.2")


class Preprocessor:
    def __init__(self, model_name: str | None = None, reasoning_effort: str | None = None,
                 base_url: str | None = None):
        self.model_name = model_name or default_model_name()
        self.reasoning_effort = (
            reasoning_effort
            if reasoning_effort is not None
            else ("low" if "gpt-oss" in self.model_name else None)
        )
        self.base_url = base_url
        if "ollama" in self.model_name and not base_url:
            self.base_url = "http://localhost:11434"
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def messages_for(self, text: str) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

    def preprocess(self, text: str) -> str:
        from litellm import completion  # lazy: litellm import is slow

        response = completion(
            messages=self.messages_for(text),
            model=self.model_name,
            reasoning_effort=self.reasoning_effort,
            api_base=self.base_url,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.total_input_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.total_output_tokens += getattr(usage, "completion_tokens", 0) or 0
        hidden = getattr(response, "_hidden_params", None) or {}
        self.total_cost += hidden.get("response_cost") or 0.0
        return response.choices[0].message.content
