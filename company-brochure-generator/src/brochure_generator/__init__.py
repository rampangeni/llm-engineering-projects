"""Company brochure generator (two-step LLM pipeline with streaming)."""

from .brochure import create_brochure, select_relevant_links, stream_brochure

__all__ = ["create_brochure", "select_relevant_links", "stream_brochure"]
__version__ = "0.1.0"
