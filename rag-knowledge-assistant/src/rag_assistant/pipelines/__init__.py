from __future__ import annotations

from ..config import Settings
from .base import Pipeline


def get_pipeline(name: str, settings: Settings | None = None, **kwargs) -> Pipeline:
    """Factory: ``name`` is 'basic' or 'pro'. kwargs (llm/embedder/store) are for injection in tests."""
    settings = settings or Settings.from_env()
    if name == "basic":
        from .basic import BasicPipeline

        return BasicPipeline(settings, **kwargs)
    if name == "pro":
        from .pro import ProPipeline

        return ProPipeline(settings, **kwargs)
    raise ValueError(f"Unknown pipeline {name!r}; choose 'basic' or 'pro'")


__all__ = ["Pipeline", "get_pipeline"]
