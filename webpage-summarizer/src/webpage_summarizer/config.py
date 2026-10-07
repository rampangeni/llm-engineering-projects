"""Environment-driven configuration and OpenAI-compatible client factory."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"
BACKENDS = ("openai", "ollama")


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings for one backend."""

    backend: str
    model: str
    base_url: str | None
    api_key: str | None


def load_settings(backend: str | None = None, model: str | None = None) -> Settings:
    """Build Settings from CLI overrides, then env vars (.env is loaded), then defaults."""
    load_dotenv(override=True)
    backend = (backend or os.getenv("LLM_BACKEND") or "openai").lower()
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; expected one of {BACKENDS}")
    if backend == "ollama":
        return Settings(
            backend="ollama",
            model=model or os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL,
            base_url=os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_URL,
            api_key="ollama",  # Ollama ignores the key but the client requires one
        )
    return Settings(
        backend="openai",
        model=model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def make_client(settings: Settings) -> OpenAI:
    """Create an OpenAI-SDK client for the chosen backend."""
    if settings.backend == "openai" and not settings.api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env, or use --backend ollama."
        )
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url)
