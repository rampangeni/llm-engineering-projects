"""Chat orchestration: system prompt, tool-calling loop, optional city image + speech."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .media import Media
from .tools import ToolRegistry

SYSTEM_PROMPT = """
You are a helpful assistant for an Airline called FlightAI.
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so.
Use your tools to look up ticket prices and to book tickets; never guess a price.
Before booking, make sure you know the destination city and the passenger's name.
"""

MAX_TOOL_ROUNDS = 5


@dataclass
class AssistantReply:
    text: str
    cities: list[str] = field(default_factory=list)   # cities touched by tools this turn
    image: Any = None                                  # PIL image or None
    audio: bytes | None = None                         # mp3 bytes or None


def _assistant_dict(message: Any) -> dict[str, Any]:
    """Convert an SDK assistant message with tool calls to a plain dict."""
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in message.tool_calls
        ],
    }


class Assistant:
    def __init__(self, client: Any, settings: Settings, registry: ToolRegistry,
                 media: Media | None = None, system_prompt: str = SYSTEM_PROMPT) -> None:
        self.client, self.settings, self.registry = client, settings, registry
        self.media, self.system_prompt = media, system_prompt

    def complete(self, history: list[dict[str, str]]) -> tuple[str, list[str]]:
        """Run the chat + tool loop. ``history`` is a list of {"role","content"} dicts."""
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        messages += [{"role": h["role"], "content": h["content"]} for h in history]
        tools = self.registry.schemas()
        cities: list[str] = []
        for _ in range(MAX_TOOL_ROUNDS + 1):
            response = self.client.chat.completions.create(
                model=self.settings.model, messages=messages, tools=tools)
            choice = response.choices[0]
            if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
                return choice.message.content or "", cities
            messages.append(_assistant_dict(choice.message))
            for call in choice.message.tool_calls:
                result = self.registry.dispatch(call.function.name, call.function.arguments)
                if result.meta.get("city"):
                    cities.append(result.meta["city"])
                messages.append({"role": "tool", "content": result.content, "tool_call_id": call.id})
        return "Sorry, I could not complete that request.", cities

    def respond(self, history: list[dict[str, str]]) -> AssistantReply:
        """Full multimodal turn: text, plus speech and a city image when enabled."""
        text, cities = self.complete(history)
        reply = AssistantReply(text=text, cities=cities)
        if self.media:
            reply.audio = self.media.speak(text)
            if cities:
                reply.image = self.media.draw_city(cities[0])
        return reply
