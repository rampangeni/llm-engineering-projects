"""Model providers, all reached through OpenAI-compatible Chat Completions endpoints."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str | None          # None = api.openai.com
    key_env: str | None           # None = no key needed
    model_env: str
    default_model: str            # used to write the brochure
    link_model_env: str
    default_link_model: str       # cheap/fast model used to pick links
    json_mode: bool               # supports response_format={"type": "json_object"}


PROVIDERS: dict[str, Provider] = {
    "GPT": Provider("GPT", None, "OPENAI_API_KEY", "GPT_MODEL", "gpt-4.1-mini",
                    "GPT_LINK_MODEL", "gpt-5-nano", True),
    "Claude": Provider("Claude", "https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY",
                       "CLAUDE_MODEL", "claude-sonnet-4-5-20250929",
                       "CLAUDE_LINK_MODEL", "claude-haiku-4-5", False),
    "Gemini": Provider("Gemini", "https://generativelanguage.googleapis.com/v1beta/openai/",
                       "GOOGLE_API_KEY", "GEMINI_MODEL", "gemini-2.5-flash",
                       "GEMINI_LINK_MODEL", "gemini-2.5-flash", True),
    "Ollama": Provider("Ollama", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"), None,
                       "OLLAMA_MODEL", "llama3.2", "OLLAMA_MODEL", "llama3.2", True),
}


def get_provider(name: str) -> Provider:
    try:
        return PROVIDERS[name]
    except KeyError:
        raise ValueError(f"Unknown model {name!r}; choose from {list(PROVIDERS)}") from None


def brochure_model(p: Provider) -> str:
    return os.getenv(p.model_env) or p.default_model


def link_model(p: Provider) -> str:
    return os.getenv(p.link_model_env) or p.default_link_model


def make_client(p: Provider) -> OpenAI:
    """Build an OpenAI-SDK client for the provider; raises if its key is missing."""
    load_dotenv(override=True)
    base_url = p.base_url
    if p.name == "Ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", base_url)
    key = os.getenv(p.key_env) if p.key_env else "ollama"
    if not key:
        raise RuntimeError(f"{p.key_env} is not set (needed for {p.name}). Add it to .env.")
    return OpenAI(api_key=key, base_url=base_url)
