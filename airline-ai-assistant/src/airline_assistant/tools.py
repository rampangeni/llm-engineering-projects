"""Tool registry: schema + handler in one place, JSON-argument dispatch with safe errors."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .tickets import TicketError, TicketStore, money


@dataclass
class ToolResult:
    """What a handler returns: text for the LLM plus optional metadata for the UI."""

    content: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]          # JSON schema for the arguments
    handler: Callable[..., ToolResult | str]
    admin_only: bool = False

    def schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


def object_schema(properties: dict[str, dict[str, str]], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False}


class ToolRegistry:
    def __init__(self, include_admin: bool = False) -> None:
        self._tools: dict[str, Tool] = {}
        self.include_admin = include_admin

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} already registered")
        self._tools[tool.name] = tool
        return tool

    def _enabled(self) -> list[Tool]:
        return [t for t in self._tools.values() if self.include_admin or not t.admin_only]

    def names(self) -> list[str]:
        return [t.name for t in self._enabled()]

    def schemas(self) -> list[dict[str, Any]]:
        """The ``tools=`` argument for the chat completions API."""
        return [t.schema() for t in self._enabled()]

    def dispatch(self, name: str, arguments_json: str) -> ToolResult:
        """Run a tool call. Never raises: problems are returned as text the model can react to."""
        tool = next((t for t in self._enabled() if t.name == name), None)
        if tool is None:
            return ToolResult(f"Error: unknown tool {name!r}")
        try:
            args = json.loads(arguments_json or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
            out = tool.handler(**args)
        except TicketError as exc:
            return ToolResult(f"Error: {exc}")
        except (ValueError, TypeError) as exc:  # bad JSON / missing or extra args
            return ToolResult(f"Error: invalid arguments for {name}: {exc}")
        return out if isinstance(out, ToolResult) else ToolResult(str(out))


def build_default_registry(store: TicketStore, include_admin: bool = False) -> ToolRegistry:
    """Register the FlightAI tools around a TicketStore."""
    reg = ToolRegistry(include_admin=include_admin)

    def get_ticket_price(destination_city: str) -> ToolResult:
        print(f"Tool called: get_ticket_price({destination_city!r})", flush=True)
        price = store.get_price(destination_city)
        if price is None:
            return ToolResult("Unknown ticket price - we may not fly there",
                              {"city": destination_city})
        return ToolResult(f"The price of a return ticket to {destination_city} is {money(price)}",
                          {"city": destination_city})

    def book_ticket(destination_city: str, passenger_name: str) -> ToolResult:
        print(f"Tool called: book_ticket({destination_city!r}, {passenger_name!r})", flush=True)
        b = store.book(destination_city, passenger_name)
        return ToolResult(f"Booked ticket {b.ticket_id} for {b.passenger} to {destination_city} "
                          f"at {money(b.price)}", {"city": destination_city})

    def set_ticket_price(destination_city: str, price: float) -> ToolResult:
        print(f"Tool called: set_ticket_price({destination_city!r}, {price!r})", flush=True)
        store.set_price(destination_city, price)
        return ToolResult(f"Price for {destination_city} set to {money(price)}")

    city = {"type": "string", "description": "The city that the customer wants to travel to"}
    reg.register(Tool("get_ticket_price", "Get the price of a return ticket to the destination city.",
                      object_schema({"destination_city": city}, ["destination_city"]), get_ticket_price))
    reg.register(Tool("book_ticket", "Book a return ticket to the destination city for a named passenger. "
                      "Only call after the customer has confirmed the city and given their name.",
                      object_schema({"destination_city": city,
                                     "passenger_name": {"type": "string", "description": "Full name of the passenger"}},
                                    ["destination_city", "passenger_name"]), book_ticket))
    reg.register(Tool("set_ticket_price", "ADMIN: set the price of a return ticket to a destination city.",
                      object_schema({"destination_city": city,
                                     "price": {"type": "number", "description": "New price in US dollars"}},
                                    ["destination_city", "price"]), set_ticket_price, admin_only=True))
    return reg
