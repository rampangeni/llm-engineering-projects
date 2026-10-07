"""EnsembleAgent: preprocess the text, price it with each model, combine.

Baseline behaviour: ``0.8 * frontier + 0.1 * specialist + 0.1 * neural_network``.
Extensions: weights from config, renormalised over the components that are available/succeeded, and an
optional conservative ``min`` strategy.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

from ..config import DEFAULT_WEIGHTS, Settings
from .base import Agent


class EnsembleAgent(Agent):
    name = "Ensemble Agent"
    color = Agent.YELLOW

    def __init__(
        self,
        components: Dict[str, object],
        weights: Optional[Dict[str, float]] = None,
        strategy: str = "weighted",
        preprocessor=None,
    ):
        """
        :param components: name ('frontier' | 'specialist' | 'nn') -> object with ``price(str) -> float``
        :param preprocessor: object with ``preprocess(str) -> str`` or None to skip rewriting
        """
        if not components:
            raise ValueError("EnsembleAgent needs at least one pricing component")
        if strategy not in ("weighted", "min"):
            raise ValueError("strategy must be 'weighted' or 'min'")
        self.log("Initializing Ensemble Agent")
        self.components = components
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.strategy = strategy
        self.preprocessor = preprocessor
        self.log(f"Ensemble Agent is ready with components {sorted(components)}")

    @classmethod
    def from_settings(cls, collection, settings: Settings, factories: Dict[str, Callable] | None = None) -> "EnsembleAgent":
        """Build components per ``Settings``. A component that fails to initialise is logged and skipped."""
        from .frontier import FrontierAgent
        from .neural_network import NeuralNetworkAgent
        from .specialist import SpecialistAgent

        factories = factories or {
            "specialist": lambda: SpecialistAgent(settings),
            "frontier": lambda: FrontierAgent(collection, settings),
            "nn": lambda: NeuralNetworkAgent(settings),
        }
        enabled = {
            "specialist": settings.enable_specialist,
            "frontier": settings.enable_frontier and collection is not None,
            "nn": settings.enable_nn,
        }
        components = {}
        for key, make in factories.items():
            if not enabled.get(key):
                logging.info(f"Ensemble: component {key!r} disabled")
                continue
            try:
                components[key] = make()
            except Exception as exc:  # noqa: BLE001
                logging.warning(f"Ensemble: component {key!r} unavailable ({exc}); continuing without it")
        preprocessor = None
        if settings.enable_preprocessor:
            from ..preprocessor import Preprocessor

            preprocessor = Preprocessor(settings.preprocessor_model)
        return cls(components, settings.ensemble_weights, settings.ensemble_strategy, preprocessor)

    def combine(self, estimates: Dict[str, float]) -> float:
        if self.strategy == "min":
            return min(estimates.values())
        total = sum(self.weights.get(k, 0.0) for k in estimates)
        if total <= 0:  # no weight configured for the survivors -> plain mean
            return sum(estimates.values()) / len(estimates)
        return sum(v * self.weights.get(k, 0.0) for k, v in estimates.items()) / total

    def price(self, description: str) -> float:
        text = description
        if self.preprocessor is not None:
            self.log("Running Ensemble Agent - preprocessing text")
            text = self.preprocessor.preprocess(description)
            self.log(f"Pre-processed text using {getattr(self.preprocessor, 'model_name', 'preprocessor')}")
        estimates: Dict[str, float] = {}
        for key, agent in self.components.items():
            try:
                estimates[key] = float(agent.price(text))
            except Exception as exc:  # noqa: BLE001
                self.log(f"Component {key!r} failed ({exc}); excluding it from this estimate")
        if not estimates:
            raise RuntimeError("All ensemble components failed")
        combined = self.combine(estimates)
        self.log(f"Ensemble Agent complete - returning ${combined:.2f}")
        return combined
