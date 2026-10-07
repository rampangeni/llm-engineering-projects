"""AutonomousPlanningAgent: an LLM drives the workflow by calling tools (scan / estimate / notify)."""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..config import Settings
from ..models import Deal, Opportunity
from .base import Agent

SCAN_FUNCTION = {
    "name": "scan_the_internet_for_bargains",
    "description": "Returns top bargains scraped from the internet along with the price each item is being offered for",
    "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
}

ESTIMATE_FUNCTION = {
    "name": "estimate_true_value",
    "description": "Given the description of an item, estimate how much it is actually worth",
    "parameters": {
        "type": "object",
        "properties": {"description": {"type": "string", "description": "The description of the item to be estimated"}},
        "required": ["description"],
        "additionalProperties": False,
    },
}

NOTIFY_FUNCTION = {
    "name": "notify_user_of_deal",
    "description": "Send the user a push notification about the single most compelling deal; only call this one time",
    "parameters": {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "The description of the item itself scraped from the internet"},
            "deal_price": {"type": "number", "description": "The price offered by this deal scraped from the internet"},
            "estimated_true_value": {"type": "number", "description": "The estimated actual value that this is worth"},
            "url": {"type": "string", "description": "The URL of this deal as scraped from the internet"},
        },
        "required": ["description", "deal_price", "estimated_true_value", "url"],
        "additionalProperties": False,
    },
}

SYSTEM_MESSAGE = "You find great deals on bargain products using your tools, and notify the user of the best bargain."
USER_MESSAGE = """
First, use your tool to scan the internet for bargain deals. Then for each deal, use your tool to estimate its true value.
Then pick the single most compelling deal where the price is much lower than the estimated true value, and use your tool to notify the user.
Then just reply OK to indicate success.
"""


class AutonomousPlanningAgent(Agent):
    name = "Autonomous Planning Agent"
    color = Agent.GREEN

    def __init__(self, scanner, ensemble, messenger, settings: Settings | None = None, client=None):
        self.log("Autonomous Planning Agent is initializing")
        self.settings = settings or Settings()
        self.MODEL = self.settings.planner_model
        self.scanner = scanner
        self.ensemble = ensemble
        self.messenger = messenger
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.openai = client
        self.memory: List[Opportunity] = []
        self.opportunity: Optional[Opportunity] = None
        self.log("Autonomous Planning Agent is ready")

    # ---- tools -------------------------------------------------------------------------------
    def scan_the_internet_for_bargains(self) -> str:
        self.log("Autonomous Planning agent is calling scanner")
        results = self.scanner.scan(memory=self.memory)
        return results.model_dump_json() if results else "No deals found"

    def estimate_true_value(self, description: str) -> str:
        self.log("Autonomous Planning agent is estimating value via Ensemble Agent")
        estimate = self.ensemble.price(description)
        return f"The estimated true value of {description} is {estimate}"

    def notify_user_of_deal(self, description: str, deal_price: float, estimated_true_value: float, url: str) -> str:
        if self.opportunity:
            self.log("Autonomous Planning agent is trying to notify the user a 2nd time; ignoring")
        else:
            self.log("Autonomous Planning agent is notifying user")
            self.messenger.notify(description, deal_price, estimated_true_value, url)
            deal = Deal(product_description=description, price=deal_price, url=url)
            self.opportunity = Opportunity(
                deal=deal, estimate=estimated_true_value, discount=estimated_true_value - deal_price
            )
        return "Notification sent ok"

    def get_tools(self) -> List[Dict]:
        return [{"type": "function", "function": f} for f in (SCAN_FUNCTION, ESTIMATE_FUNCTION, NOTIFY_FUNCTION)]

    def handle_tool_call(self, message) -> List[Dict]:
        """Execute every tool call in an assistant message; errors are returned to the model as text."""
        mapping = {
            "scan_the_internet_for_bargains": self.scan_the_internet_for_bargains,
            "estimate_true_value": self.estimate_true_value,
            "notify_user_of_deal": self.notify_user_of_deal,
        }
        results = []
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            tool = mapping.get(name)
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
                result = tool(**arguments) if tool else f"Unknown tool {name}"
            except Exception as exc:  # noqa: BLE001
                self.log(f"Tool {name} failed: {exc}")
                result = f"Tool {name} failed: {exc}"
            results.append({"role": "tool", "content": str(result), "tool_call_id": tool_call.id})
        return results

    # ---- the loop ------------------------------------------------------------------------------
    def plan(self, memory: List[Opportunity] = ()) -> Optional[Opportunity]:
        """Run the tool-calling loop until the model stops calling tools (or ``max_tool_steps`` is hit)."""
        self.log("Autonomous Planning Agent is kicking off a run")
        self.memory = list(memory)
        self.opportunity = None
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": USER_MESSAGE}]
        reply = ""
        for _ in range(self.settings.max_tool_steps):
            response = self.openai.chat.completions.create(model=self.MODEL, messages=messages, tools=self.get_tools())
            choice = response.choices[0]
            if choice.finish_reason == "tool_calls":
                messages.append(choice.message)
                messages.extend(self.handle_tool_call(choice.message))
            else:
                reply = choice.message.content
                break
        else:
            self.log(f"Autonomous Planning Agent stopped after {self.settings.max_tool_steps} steps")
        self.log(f"Autonomous Planning Agent completed with: {reply}")
        return self.opportunity
