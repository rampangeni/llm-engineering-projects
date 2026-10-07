"""MessagingAgent: Pushover push notifications; optional LLM-crafted message text."""

from __future__ import annotations

import requests

from ..config import Settings
from ..models import Opportunity
from .base import Agent

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


class MessagingAgent(Agent):
    name = "Messaging Agent"
    color = Agent.WHITE

    def __init__(self, settings: Settings | None = None, completion_fn=None, http_post=None):
        """
        :param completion_fn: litellm-style ``completion(model=, messages=)`` (injectable)
        :param http_post: ``requests.post``-style callable (injectable)
        """
        self.settings = settings or Settings()
        self.MODEL = self.settings.messaging_model
        self.log("Messaging Agent is initializing")
        self.pushover_user = self.settings.pushover_user
        self.pushover_token = self.settings.pushover_token
        self._completion = completion_fn
        self._post = http_post or requests.post
        mode = "live" if self.configured else "DRY-RUN (PUSHOVER_USER/PUSHOVER_TOKEN not set)"
        self.log(f"Messaging Agent is ready, Pushover mode: {mode}")

    @property
    def configured(self) -> bool:
        return bool(self.pushover_user and self.pushover_token)

    def push(self, text: str) -> bool:
        """Send via Pushover. Returns True when delivered; never raises on network errors."""
        if not self.configured:
            self.log(f"Dry run, would push: {text}")
            return False
        self.log("Messaging Agent is sending a push notification")
        payload = {"user": self.pushover_user, "token": self.pushover_token, "message": text, "sound": "cashregister"}
        try:
            response = self._post(PUSHOVER_URL, data=payload, timeout=10)
            ok = getattr(response, "status_code", 200) == 200
        except Exception as exc:  # noqa: BLE001
            self.log(f"Push failed: {exc}")
            return False
        if not ok:
            self.log(f"Push rejected with status {response.status_code}")
        return ok

    def alert(self, opportunity: Opportunity) -> bool:
        text = f"Deal Alert! Price=${opportunity.deal.price:.2f}, "
        text += f"Estimate=${opportunity.estimate:.2f}, "
        text += f"Discount=${opportunity.discount:.2f} :"
        text += opportunity.deal.product_description[:80] + "... "
        text += opportunity.deal.url
        ok = self.push(text)
        self.log("Messaging Agent has completed")
        return ok

    def craft_message(self, description: str, deal_price: float, estimated_true_value: float) -> str:
        user_prompt = "Please summarize this great deal in 2-3 sentences to be sent as an exciting push notification alerting the user about this deal.\n"
        user_prompt += f"Item Description: {description}\nOffered Price: {deal_price}\nEstimated true value: {estimated_true_value}"
        user_prompt += "\n\nRespond only with the 2-3 sentence message which will be used to alert & excite the user about this deal"
        completion = self._completion
        if completion is None:
            from litellm import completion  # lazy
        response = completion(model=self.MODEL, messages=[{"role": "user", "content": user_prompt}])
        return response.choices[0].message.content

    def notify(self, description: str, deal_price: float, estimated_true_value: float, url: str) -> bool:
        """Craft an exciting message with an LLM (falls back to a plain template) and push it."""
        self.log(f"Messaging Agent is using {self.MODEL} to craft the message")
        try:
            text = self.craft_message(description, deal_price, estimated_true_value)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Message crafting failed ({exc}); using plain template")
            text = f"Deal: {description[:100]} - offered at ${deal_price:.2f}, worth about ${estimated_true_value:.2f}."
        ok = self.push(text[:200] + "... " + url)
        self.log("Messaging Agent has completed")
        return ok
