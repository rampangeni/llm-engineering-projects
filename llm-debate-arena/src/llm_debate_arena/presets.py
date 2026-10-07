"""Built-in conversations. ``classic`` is the classic GPT-vs-Claude exchange."""

from __future__ import annotations

from .debate import Bot, DebateConfig, Turn

ARGUMENTATIVE = (
    "You are a chatbot who is very argumentative; "
    "you disagree with anything in the conversation and you challenge everything, in a snarky way."
)
POLITE = (
    "You are a very polite, courteous chatbot. You try to agree with "
    "everything the other person says, or find common ground. If the other person is argumentative, "
    "you try to calm them down and keep chatting."
)
SKEPTIC = "You are a skeptical scientist who asks for evidence and keeps replies under 3 sentences."
OPTIMIST = "You are an upbeat futurist who sees opportunity everywhere and keeps replies under 3 sentences."


def classic() -> DebateConfig:
    return DebateConfig(
        bots=[
            Bot("GPT", "openai:gpt-4.1-mini", ARGUMENTATIVE),
            Bot("Claude", "anthropic:claude-haiku-4-5", POLITE),
        ],
        rounds=5,
        seed=[Turn("GPT", "Hi there"), Turn("Claude", "Hi")],
    )


def local() -> DebateConfig:
    """Same personalities, both on Ollama (free, no API keys)."""
    return DebateConfig(
        bots=[Bot("Argumentative", "ollama:llama3.2", ARGUMENTATIVE), Bot("Polite", "ollama:llama3.2", POLITE)],
        rounds=3,
        seed=[Turn("Argumentative", "Hi there"), Turn("Polite", "Hi")],
    )


def roundtable() -> DebateConfig:
    """Three providers, three personas, one topic."""
    return DebateConfig(
        bots=[
            Bot("GPT", "openai:gpt-4.1-mini", ARGUMENTATIVE),
            Bot("Claude", "anthropic:claude-haiku-4-5", OPTIMIST),
            Bot("Gemini", "gemini:gemini-2.5-flash", SKEPTIC),
        ],
        rounds=3,
        topic="Will AI agents replace most software engineering jobs?",
    )


PRESETS = {"classic": classic, "local": local, "roundtable": roundtable}


def get_preset(name: str) -> DebateConfig:
    try:
        return PRESETS[name]()
    except KeyError:
        raise ValueError(f"Unknown preset {name!r}; choose from {sorted(PRESETS)}") from None
