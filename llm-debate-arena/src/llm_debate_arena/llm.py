"""Provider registry and a thin router over OpenAI-compatible Chat Completions endpoints.

Model specs look like ``provider:model`` (``anthropic:claude-haiku-4-5``,
``ollama:llama3.2:1b``). A spec without a known provider prefix is an OpenAI model.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    base_url: str | None  # None -> api.openai.com
    key_env: str | None   # None -> no key needed (Ollama)


PROVIDERS: dict[str, ProviderInfo] = {
    p.name: p
    for p in (
        ProviderInfo("openai", None, "OPENAI_API_KEY"),
        ProviderInfo("anthropic", "https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY"),
        ProviderInfo("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_API_KEY"),
        ProviderInfo("deepseek", "https://api.deepseek.com", "DEEPSEEK_API_KEY"),
        ProviderInfo("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
        ProviderInfo("grok", "https://api.x.ai/v1", "GROK_API_KEY"),
        ProviderInfo("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
        ProviderInfo("ollama", "http://localhost:11434/v1", None),
    )
}


class ProviderError(RuntimeError):
    """Missing key, unknown provider or failed call."""


def parse_model(spec: str) -> tuple[str, str]:
    """Split ``provider:model``; default provider is openai. Only the first colon splits."""
    prefix, sep, rest = spec.partition(":")
    if sep and prefix.lower() in PROVIDERS:
        return prefix.lower(), rest
    return "openai", spec


@dataclass
class Completion:
    spec: str
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    seconds: float = 0.0


class LLMRouter:
    """Lazily builds and caches one client per provider."""

    def __init__(self) -> None:
        load_dotenv(override=True)
        self._clients: dict[str, OpenAI] = {}

    def client(self, provider: str) -> OpenAI:
        if provider not in PROVIDERS:
            raise ProviderError(f"Unknown provider {provider!r}; known: {sorted(PROVIDERS)}")
        if provider not in self._clients:
            info = PROVIDERS[provider]
            key = os.getenv(info.key_env) if info.key_env else "ollama"
            if not key:
                raise ProviderError(f"{info.key_env} is not set (needed for provider {provider!r})")
            base_url = info.base_url
            if provider == "ollama":
                base_url = os.getenv("OLLAMA_BASE_URL", base_url)
            self._clients[provider] = OpenAI(api_key=key, base_url=base_url)
        return self._clients[provider]

    def chat(self, spec: str, messages: list[dict[str, str]], **kwargs: Any) -> Completion:
        """One non-streaming chat completion. Extra kwargs (e.g. reasoning_effort) pass through."""
        provider, model = parse_model(spec)
        client = self.client(provider)
        start = time.perf_counter()
        try:
            resp = client.chat.completions.create(model=model, messages=messages, **kwargs)
        except Exception as exc:
            raise ProviderError(f"{spec}: {exc}") from exc
        usage = getattr(resp, "usage", None)
        return Completion(
            spec=spec,
            text=resp.choices[0].message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            seconds=time.perf_counter() - start,
        )

    def text(self, spec: str, messages: list[dict[str, str]]) -> str:
        """Convenience ChatFn: returns only the reply text."""
        return self.chat(spec, messages).text
