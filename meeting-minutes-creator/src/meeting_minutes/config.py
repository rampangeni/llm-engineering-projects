"""Environment-driven configuration. Default model names are set below."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class Settings:
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "openai/whisper-medium"))
    llama_model: str = field(
        default_factory=lambda: os.getenv("LLAMA_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    )
    max_new_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_NEW_TOKENS", "2000")))
    openai_transcribe_model: str = field(default_factory=lambda: os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1"))
    openai_chat_model: str = field(default_factory=lambda: os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"))
    backend: str = field(default_factory=lambda: os.getenv("BACKEND", "auto"))  # auto | local | api


def cuda_available() -> bool:
    """True only if torch is installed *and* sees a CUDA GPU (never imports torch eagerly elsewhere)."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def resolve_backend(requested: str, cuda: bool | None = None) -> str:
    """Map auto|local|api to local|api."""
    if requested not in {"auto", "local", "api"}:
        raise ValueError(f"backend must be auto, local or api, got {requested!r}")
    if requested != "auto":
        return requested
    return "local" if (cuda_available() if cuda is None else cuda) else "api"
