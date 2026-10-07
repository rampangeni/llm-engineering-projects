"""Environment-driven settings and client factory."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI


def _flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    backend: str
    model: str
    image_model: str
    tts_model: str
    tts_voice: str
    base_url: str | None
    api_key: str | None
    admin_tools: bool

    @property
    def media_supported(self) -> bool:
        """Image generation and TTS need the OpenAI API (Ollama has neither)."""
        return self.backend == "openai"


def load_settings(backend: str | None = None, model: str | None = None) -> Settings:
    load_dotenv(override=True)
    backend = (backend or os.getenv("ASSISTANT_BACKEND") or "openai").lower()
    if backend not in ("openai", "ollama"):
        raise ValueError(f"Unknown backend {backend!r}; use 'openai' or 'ollama'")
    common = dict(
        image_model=os.getenv("IMAGE_MODEL", "gpt-image-1-mini"),
        tts_model=os.getenv("TTS_MODEL", "gpt-4o-mini-tts"),
        tts_voice=os.getenv("TTS_VOICE", "onyx"),
        admin_tools=_flag("ENABLE_ADMIN_TOOLS"),
    )
    if backend == "ollama":
        return Settings("ollama", model or os.getenv("OLLAMA_MODEL", "llama3.2"),
                        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                        api_key="ollama", **common)
    return Settings("openai", model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                    base_url=os.getenv("OPENAI_BASE_URL") or None,
                    api_key=os.getenv("OPENAI_API_KEY"), **common)


def make_client(settings: Settings) -> OpenAI:
    if not settings.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or use --backend ollama.")
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url)
