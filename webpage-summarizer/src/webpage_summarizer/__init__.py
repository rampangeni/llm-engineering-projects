"""Summarize web pages with an LLM (OpenAI or Ollama)."""

from .config import Settings, load_settings
from .scraper import ScrapeError, fetch_website_contents
from .summarizer import messages_for, summarize

__all__ = [
    "Settings",
    "load_settings",
    "ScrapeError",
    "fetch_website_contents",
    "messages_for",
    "summarize",
]
__version__ = "0.1.0"
