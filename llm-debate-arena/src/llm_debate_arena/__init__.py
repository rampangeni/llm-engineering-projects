"""LLM Debate Arena: multi-provider chat, comparison and multi-bot conversations."""

from .debate import Bot, DebateConfig, DebateEngine, Turn
from .llm import Completion, LLMRouter, parse_model

__all__ = ["Bot", "DebateConfig", "DebateEngine", "Turn", "Completion", "LLMRouter", "parse_model"]
__version__ = "0.1.0"
