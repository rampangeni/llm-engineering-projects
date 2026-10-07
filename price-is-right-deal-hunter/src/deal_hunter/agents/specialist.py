"""SpecialistAgent: the fine-tuned Llama on Modal (deployed by ../price-model-modal-service).

The Modal object is resolved BY NAME (app/class from config) - this project never imports the service
package. ``backend="mock"`` swaps in a deterministic offline stub for development.
"""

from __future__ import annotations

import hashlib

from ..config import Settings
from .base import Agent


class MockPricer:
    """Stable pseudo-price in [10, 510) from the text hash."""

    def price(self, description: str) -> float:
        return float(10 + int(hashlib.md5(description.encode("utf-8")).hexdigest()[:6], 16) % 500)


class ModalPricer:
    def __init__(self, app_name: str, class_name: str):
        import modal  # lazy; optional extra

        self._instance = modal.Cls.from_name(app_name, class_name)()

    def price(self, description: str) -> float:
        return float(self._instance.price.remote(description))


class SpecialistAgent(Agent):
    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self, settings: Settings | None = None, pricer=None):
        s = settings or Settings()
        if pricer is None:
            if s.specialist_backend == "mock":
                self.log("Specialist Agent is using the MOCK pricer (no Modal call)")
                pricer = MockPricer()
            else:
                self.log("Specialist Agent is initializing - connecting to modal")
                pricer = ModalPricer(s.modal_app_name, s.modal_class_name)
        self.pricer = pricer

    def price(self, description: str) -> float:
        self.log("Specialist Agent is calling remote fine-tuned model")
        result = self.pricer.price(description)
        self.log(f"Specialist Agent completed - predicting ${result:.2f}")
        return result
