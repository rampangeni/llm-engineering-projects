"""ScannerAgent: RSS feeds -> LLM (structured output) -> 5 best-described deals with clear prices."""

from __future__ import annotations

from typing import List, Optional

from ..config import Settings
from ..models import DealSelection, Opportunity
from ..scraping import ScrapedDeal
from .base import Agent

SAMPLE_DEALS = {
    "deals": [
        {
            "product_description": "The Hisense R6 Series 55R6030N is a 55-inch 4K UHD Roku Smart TV that offers stunning picture quality with 3840x2160 resolution. It features Dolby Vision HDR and HDR10 compatibility, and runs Roku's operating system with voice control through Google Assistant and Alexa. Three HDMI ports make connecting multiple devices simple.",
            "price": 178,
            "url": "https://www.dealnews.com/products/Hisense/Hisense-R6-Series-55-R6030-N-55-4-K-UHD-Roku-Smart-TV/484824.html?iref=rss-c142",
        },
        {
            "product_description": "The Poly Studio P21 is a 21.5-inch LED personal meeting display designed for remote work and video conferencing. It has a native 1080p resolution, a privacy shutter, stereo speakers, a 1080p webcam with manual pan/tilt/zoom, and 5W wireless charging for mobile devices.",
            "price": 30,
            "url": "https://www.dealnews.com/products/Poly-Studio-P21-21-5-1080-p-LED-Personal-Meeting-Display/378335.html?iref=rss-c39",
        },
        {
            "product_description": "The Lenovo IdeaPad Slim 5 laptop is powered by an AMD Ryzen 5 8645HS 6-core CPU with a 16-inch 1920x1080 touch display, 16GB of RAM and a 512GB SSD, built for everyday multitasking.",
            "price": 446,
            "url": "https://www.dealnews.com/products/Lenovo/Lenovo-Idea-Pad-Slim-5-7-th-Gen-Ryzen-5-16-Touch-Laptop/485068.html?iref=rss-c39",
        },
    ]
}


class ScannerAgent(Agent):
    SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list, by selecting deals that have the most detailed, high quality description and the most clear price.
    Respond strictly in JSON with no explanation, using this format. You should provide the price as a number derived from the description. If the price of a deal isn't clear, do not include that deal in your response.
    Most important is that you respond with the 5 deals that have the most detailed product description with price. It's not important to mention the terms of the deal; most important is a thorough description of the product.
    Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price. 
    """

    USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list, selecting those which have the most detailed, high quality product description and a clear price that is greater than 0.
    You should rephrase the description to be a summary of the product itself, not the terms of the deal.
    Remember to respond with a short paragraph of text in the product_description field for each of the 5 items that you select.
    Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price. 
    
    Deals:
    
    """

    USER_PROMPT_SUFFIX = "\n\nInclude exactly 5 deals, no more."

    name = "Scanner Agent"
    color = Agent.CYAN

    def __init__(self, settings: Settings | None = None, client=None, fetcher=None):
        """
        :param client: an OpenAI-compatible client (injectable for tests)
        :param fetcher: callable returning List[ScrapedDeal] (injectable for tests)
        """
        self.settings = settings or Settings()
        self.log("Scanner Agent is initializing")
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.openai = client
        self._fetcher = fetcher or (
            lambda: ScrapedDeal.fetch(self.settings.feeds, per_feed=self.settings.entries_per_feed)
        )
        self.log("Scanner Agent is ready")

    def fetch_deals(self, memory: List[Opportunity]) -> List[ScrapedDeal]:
        """New deals from the RSS feeds that are not already in memory."""
        self.log("Scanner Agent is about to fetch deals from RSS feed")
        urls = {opp.deal.url for opp in memory}
        result = [s for s in self._fetcher() if s.url not in urls]
        self.log(f"Scanner Agent received {len(result)} deals not already scraped")
        return result

    def make_user_prompt(self, scraped: List[ScrapedDeal]) -> str:
        return self.USER_PROMPT_PREFIX + "\n\n".join(s.describe() for s in scraped) + self.USER_PROMPT_SUFFIX

    def scan(self, memory: List[Opportunity] = ()) -> Optional[DealSelection]:
        """Ask the LLM (Structured Outputs) for the best deals; None if nothing new/usable."""
        scraped = self.fetch_deals(list(memory))
        if not scraped:
            return None
        self.log("Scanner Agent is calling OpenAI using Structured Outputs")
        completion = self.openai.chat.completions.parse(
            model=self.settings.scanner_model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self.make_user_prompt(scraped)},
            ],
            response_format=DealSelection,
            reasoning_effort="minimal",
        )
        result = completion.choices[0].message.parsed
        if result is None:  # refusal / parse failure
            self.log("Scanner Agent received no parsable selection")
            return None
        result.deals = [d for d in result.deals if d.price > 0]
        self.log(f"Scanner Agent received {len(result.deals)} selected deals with price>0 from OpenAI")
        return result

    def test_scan(self, memory: List[Opportunity] = ()) -> DealSelection:
        """Canned selection for offline demos/tests (no network, no LLM)."""
        return DealSelection(**SAMPLE_DEALS)


class DemoScannerAgent(ScannerAgent):
    """Offline scanner for demos: serves the canned selection, skipping URLs already in memory."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.openai = None
        self.log("Demo Scanner Agent is ready (canned deals, no network)")

    def scan(self, memory: List[Opportunity] = ()) -> Optional[DealSelection]:
        seen = {o.deal.url for o in memory}
        deals = [d for d in self.test_scan().deals if d.url not in seen]
        return DealSelection(deals=deals) if deals else None
