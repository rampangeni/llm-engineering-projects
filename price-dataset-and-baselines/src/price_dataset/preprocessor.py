"""Synchronous LLM rewrite of an item's full text into the standard summary format.

Uses litellm so any provider works (Groq, OpenAI, Ollama ...). For the whole dataset use the
Groq Batch API instead (see ``batch.py``) - it is ~50% cheaper and runs unattended.
"""

from __future__ import annotations

import os

DEFAULT_MODEL_NAME = os.environ.get("PREPROCESSOR_MODEL", "groq/openai/gpt-oss-20b")
DEFAULT_REASONING_EFFORT = "low"

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


class Preprocessor:
    def __init__(self, model_name=DEFAULT_MODEL_NAME, reasoning_effort=DEFAULT_REASONING_EFFORT, completion_fn=None):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.model_name = model_name
        # only gpt-oss style models accept reasoning_effort
        self.reasoning_effort = reasoning_effort if "gpt-oss" in model_name else None
        self._completion = completion_fn

    def messages_for(self, text: str) -> list[dict]:
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]

    def preprocess(self, text: str) -> str:
        completion = self._completion
        if completion is None:
            from litellm import completion  # lazy: optional dependency
        kwargs = {"messages": self.messages_for(text), "model": self.model_name}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        response = completion(**kwargs)
        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens
        try:
            self.total_cost += response._hidden_params["response_cost"] or 0
        except (AttributeError, KeyError, TypeError):
            pass  # cost is not known for every provider (e.g. local Ollama)
        return response.choices[0].message.content
