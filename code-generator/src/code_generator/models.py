"""Model registry and OpenAI-compatible client factory.

Every provider is reached through the ``openai`` SDK by changing
``base_url``: Anthropic, Gemini, Grok, Groq, OpenRouter and Ollama all expose an
OpenAI-compatible chat completions endpoint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

BASE_URLS = {
    "openai": None,
    "anthropic": "https://api.anthropic.com/v1/",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "grok": "https://api.x.ai/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
}

KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "grok": "GROK_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "ollama": None,  # no key needed
}


@dataclass(frozen=True)
class ModelSpec:
    name: str  # model id sent to the API (also the dropdown label)
    provider: str  # key into BASE_URLS / KEY_ENV
    reasoning_effort: str | None = None  # only sent when set (GPT-5 family)


def default_models() -> list[ModelSpec]:
    """Model list. Frontier names overridable via env."""
    return [
        ModelSpec(os.getenv("OPENAI_MODEL", "gpt-5"), "openai", "high"),
        ModelSpec(os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929"), "anthropic"),
        ModelSpec(os.getenv("GROK_MODEL", "grok-4"), "grok"),
        ModelSpec(os.getenv("GEMINI_MODEL", "gemini-2.5-pro"), "gemini"),
        ModelSpec("qwen2.5-coder", "ollama"),
        ModelSpec("deepseek-coder-v2", "ollama"),
        ModelSpec("gpt-oss:20b", "ollama"),
        ModelSpec("qwen/qwen3-coder-30b-a3b-instruct", "openrouter"),
        ModelSpec("openai/gpt-oss-120b", "groq"),
    ]


def find_model(name: str, models: list[ModelSpec] | None = None) -> ModelSpec:
    for m in models or default_models():
        if m.name == name:
            return m
    raise KeyError(f"Unknown model: {name}")


def make_client(provider: str) -> OpenAI:
    """Build an OpenAI-compatible client; raise a clear error if the key is missing."""
    env = KEY_ENV[provider]
    key = os.getenv(env) if env else "ollama"
    if not key:
        raise RuntimeError(f"{env} is not set; add it to .env to use {provider} models.")
    return OpenAI(api_key=key, base_url=BASE_URLS[provider])
