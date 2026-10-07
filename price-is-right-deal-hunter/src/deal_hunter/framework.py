"""DealAgentFramework: owns memory (surfaced opportunities), the vector store and the planner."""

from __future__ import annotations

import json
import logging
import os
from importlib import resources
from typing import List, Optional

from .config import Settings
from .log_utils import BG_BLUE, RESET, WHITE, init_logging
from .models import Opportunity


def load_seed_memory() -> List[dict]:
    return json.loads(resources.files("deal_hunter.data").joinpath("memory_seed.json").read_text())


def build_planner(settings: Settings, collection=None, demo: bool = False):
    """Wire scanner + ensemble + messenger into the planner chosen by ``settings.planner``.

    ``demo=True`` => fully offline: canned deals, mock specialist, no RAG/NN/preprocessor/Pushover,
    workflow planner. Useful to see the whole loop without any API key.
    """
    import dataclasses

    from .agents.autonomous_planning import AutonomousPlanningAgent
    from .agents.ensemble import EnsembleAgent
    from .agents.messaging import MessagingAgent
    from .agents.planning import PlanningAgent
    from .agents.scanner import DemoScannerAgent, ScannerAgent

    if demo:
        settings = dataclasses.replace(
            settings, specialist_backend="mock", enable_frontier=False, enable_nn=False,
            enable_preprocessor=False, pushover_user="", pushover_token="", planner="workflow",
        )
        scanner = DemoScannerAgent(settings)
    else:
        scanner = ScannerAgent(settings)
    ensemble = EnsembleAgent.from_settings(collection, settings)
    messenger = MessagingAgent(settings)
    if settings.planner == "autonomous":
        return AutonomousPlanningAgent(scanner, ensemble, messenger, settings)
    return PlanningAgent(scanner, ensemble, messenger, settings)


class DealAgentFramework:
    def __init__(self, settings: Settings | None = None, planner=None, collection=None, demo: bool = False):
        init_logging()
        self.settings = settings or Settings()
        self.demo = demo
        self.memory: List[Opportunity] = self.read_memory()
        self._collection = collection
        self.planner = planner

    # ---- memory -------------------------------------------------------------------------------
    def read_memory(self) -> List[Opportunity]:
        path = self.settings.memory_path
        if os.path.exists(path):
            with open(path, "r") as f:
                return [Opportunity(**item) for item in json.load(f)]
        return []

    def write_memory(self) -> None:
        with open(self.settings.memory_path, "w") as f:
            json.dump([o.model_dump() for o in self.memory], f, indent=2)

    @classmethod
    def reset_memory(cls, settings: Settings | None = None) -> None:
        """Truncate memory back to 2 deals (seeding from the packaged example if empty/missing)."""
        s = settings or Settings()
        data: List[dict] = []
        if os.path.exists(s.memory_path):
            with open(s.memory_path, "r") as f:
                data = json.load(f)
        data = data or load_seed_memory()
        with open(s.memory_path, "w") as f:
            json.dump(data[:2], f, indent=2)

    # ---- agents -------------------------------------------------------------------------------
    @property
    def collection(self):
        if self._collection is None and not self.demo:
            from .vectorstore import open_collection

            self._collection = open_collection(self.settings.chroma_path)
        return self._collection

    def log(self, message: str) -> None:
        logging.info(BG_BLUE + WHITE + "[Agent Framework] " + message + RESET)

    def init_agents_as_needed(self) -> None:
        if not self.planner:
            self.log("Initializing Agent Framework")
            self.planner = build_planner(self.settings, self.collection, demo=self.demo)
            self.log("Agent Framework is ready")

    def run(self) -> List[Opportunity]:
        self.init_agents_as_needed()
        logging.info("Kicking off Planning Agent")
        result: Optional[Opportunity] = self.planner.plan(memory=self.memory)
        logging.info(f"Planning Agent has completed and returned: {result}")
        if result:
            self.memory.append(result)
            self.write_memory()
        return self.memory

    @classmethod
    def get_plot_data(cls, settings: Settings | None = None, max_datapoints: int = 2000):
        from .vectorstore import get_plot_data, open_collection

        s = settings or Settings()
        return get_plot_data(open_collection(s.chroma_path), max_datapoints)
