"""Configurable multi-bot conversation engine (generalises the GPT-vs-Claude loop)."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ChatFn = Callable[[str, list[dict[str, str]]], str]  # (model spec, messages) -> reply text
MODERATOR = "Moderator"


@dataclass
class Bot:
    name: str
    model: str            # "provider:model"
    system_prompt: str


@dataclass
class Turn:
    speaker: str
    content: str


@dataclass
class DebateConfig:
    bots: list[Bot]
    rounds: int = 5
    seed: list[Turn] = field(default_factory=list)   # pre-written opening turns
    topic: str | None = None                          # optional moderator opener

    def validate(self) -> None:
        if len(self.bots) < 2:
            raise ValueError("A conversation needs at least 2 bots")
        names = [b.name for b in self.bots]
        if len(set(names)) != len(names) or MODERATOR in names:
            raise ValueError(f"Bot names must be unique and not {MODERATOR!r}: {names}")
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")


def config_from_dict(data: dict[str, Any]) -> DebateConfig:
    cfg = DebateConfig(
        bots=[Bot(**b) for b in data["bots"]],
        rounds=int(data.get("rounds", 5)),
        seed=[Turn(**t) for t in data.get("seed", [])],
        topic=data.get("topic"),
    )
    cfg.validate()
    return cfg


def config_to_dict(cfg: DebateConfig) -> dict[str, Any]:
    return {
        "bots": [vars(b) for b in cfg.bots],
        "rounds": cfg.rounds,
        "seed": [vars(t) for t in cfg.seed],
        "topic": cfg.topic,
    }


def load_config(path: str | Path) -> DebateConfig:
    return config_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_messages(bot: Bot, transcript: list[Turn], multi_party: bool) -> list[dict[str, str]]:
    """Render the shared transcript from ``bot``'s point of view.

    Own turns -> assistant; everyone else -> user (prefixed with the speaker name when more
    than two bots take part). Consecutive same-role messages are merged and the list is
    forced to start with a user message, which Anthropic requires.
    """
    msgs: list[dict[str, str]] = []
    for turn in transcript:
        own = turn.speaker == bot.name
        role = "assistant" if own else "user"
        content = turn.content
        if not own and multi_party and turn.speaker != MODERATOR:
            content = f"{turn.speaker}: {content}"
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] += "\n\n" + content
        else:
            msgs.append({"role": role, "content": content})
    if not msgs or msgs[0]["role"] == "assistant":
        msgs.insert(0, {"role": "user", "content": "Hello"})
    return [{"role": "system", "content": bot.system_prompt}, *msgs]


class DebateEngine:
    """Runs round-robin turns. ``chat`` is injected so tests/UIs can supply any backend."""

    def __init__(self, config: DebateConfig, chat: ChatFn) -> None:
        config.validate()
        self.config = config
        self.chat = chat
        self.transcript: list[Turn] = []
        if config.topic:
            self.transcript.append(Turn(MODERATOR, f"The topic of discussion is: {config.topic}"))
        self.transcript.extend(config.seed)

    def run(self) -> Iterator[Turn]:
        """Yield each new turn as soon as it is produced (seed turns are not re-yielded)."""
        multi = len(self.config.bots) > 2
        for _ in range(self.config.rounds):
            for bot in self.config.bots:
                reply = self.chat(bot.model, build_messages(bot, self.transcript, multi))
                turn = Turn(bot.name, reply)
                self.transcript.append(turn)
                yield turn

    def seed_turns(self) -> list[Turn]:
        return list(self.transcript)


def format_turn(turn: Turn) -> str:
    return f"### {turn.speaker}:\n{turn.content}\n"
