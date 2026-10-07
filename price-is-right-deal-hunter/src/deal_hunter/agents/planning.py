"""PlanningAgent: the fixed workflow (scan -> price every deal -> alert on the best one)."""

from __future__ import annotations

from typing import List, Optional

from ..config import Settings
from ..models import Deal, Opportunity
from .base import Agent


class PlanningAgent(Agent):
    name = "Planning Agent"
    color = Agent.GREEN

    def __init__(self, scanner, ensemble, messenger, settings: Settings | None = None):
        """Agents are injected; use ``deal_hunter.framework.build_planner`` to wire them from settings."""
        self.log("Planning Agent is initializing")
        self.settings = settings or Settings()
        self.DEAL_THRESHOLD = self.settings.deal_threshold
        self.scanner = scanner
        self.ensemble = ensemble
        self.messenger = messenger
        self.log("Planning Agent is ready")

    def run(self, deal: Deal) -> Opportunity:
        self.log("Planning Agent is pricing up a potential deal")
        estimate = self.ensemble.price(deal.product_description)
        discount = estimate - deal.price
        self.log(f"Planning Agent has processed a deal with discount ${discount:.2f}")
        return Opportunity(deal=deal, estimate=estimate, discount=discount)

    def plan(self, memory: List[Opportunity] = ()) -> Optional[Opportunity]:
        """1) scan  2) price each deal  3) alert if the best discount beats the threshold.

        :param memory: opportunities surfaced in the past (their URLs are skipped)
        :return: the best Opportunity if it was alerted, else None
        """
        self.log("Planning Agent is kicking off a run")
        selection = self.scanner.scan(memory=list(memory))
        if not selection or not selection.deals:
            return None
        opportunities = [self.run(deal) for deal in selection.deals[:5]]
        opportunities.sort(key=lambda opp: opp.discount, reverse=True)
        best = opportunities[0]
        self.log(f"Planning Agent has identified the best deal has discount ${best.discount:.2f}")
        if best.discount > self.DEAL_THRESHOLD:
            self.messenger.alert(best)
            self.log("Planning Agent has completed a run")
            return best
        self.log("Planning Agent has completed a run (no deal above threshold)")
        return None
