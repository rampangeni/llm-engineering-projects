"""Thin LLM client around litellm so every call site is mockable."""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    def chat(self, messages: list[dict], response_model: type[T] | None = None) -> str | T: ...


class LiteLLMClient:
    """Chat completions via litellm. With ``response_model`` returns a parsed pydantic object."""

    def __init__(self, model: str, temperature: float = 0.0, max_attempts: int = 5):
        self.model = model
        self.temperature = temperature
        self.max_attempts = max_attempts

    def chat(self, messages: list[dict], response_model: type[T] | None = None):
        @retry(
            wait=wait_exponential(multiplier=1, min=2, max=60),
            stop=stop_after_attempt(self.max_attempts),
            reraise=True,
        )
        def _call():
            from litellm import completion  # lazy: heavy import

            kwargs = {"model": self.model, "messages": messages}
            if response_model is not None:
                kwargs["response_format"] = response_model
            resp = completion(**kwargs)
            content = resp.choices[0].message.content
            if response_model is not None:
                return response_model.model_validate_json(content)
            return content

        return _call()
