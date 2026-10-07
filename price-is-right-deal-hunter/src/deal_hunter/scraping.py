"""RSS + page scraping of deals (dealnews by default)."""

from __future__ import annotations

import logging
import re
import time
from typing import Dict, Iterable, List

import requests
from bs4 import BeautifulSoup

from .config import DEFAULT_FEEDS

log = logging.getLogger(__name__)


def extract(html_snippet: str) -> str:
    """Clean an RSS summary: take the ``snippet summary`` div text if present."""
    soup = BeautifulSoup(html_snippet, "html.parser")
    snippet_div = soup.find("div", class_="snippet summary")
    if snippet_div:
        description = snippet_div.get_text(" ", strip=True)
        description = BeautifulSoup(description, "html.parser").get_text()
        description = re.sub("<[^<]+?>", "", description)
        result = description.strip()
    else:
        result = html_snippet
    return result.replace("\n", " ")


class ScrapedDeal:
    """A deal from an RSS entry plus details scraped from the deal page."""

    category: str
    title: str
    summary: str
    url: str
    details: str
    features: str

    def __init__(self, entry: Dict, timeout: float = 15.0):
        self.title = entry["title"]
        self.summary = extract(entry.get("summary", ""))
        self.url = entry["links"][0]["href"]
        response = requests.get(self.url, timeout=timeout)
        soup = BeautifulSoup(response.content, "html.parser")
        section = soup.find("div", class_="content-section")
        content = section.get_text() if section else self.summary
        content = content.replace("\nmore", "").replace("\n", " ")
        if "Features" in content:
            self.details, self.features = content.split("Features", 1)
        else:
            self.details, self.features = content, ""
        self.truncate()

    def truncate(self) -> None:
        """Keep the prompt small."""
        self.title = self.title[:100]
        self.details = self.details[:500]
        self.features = self.features[:500]

    def __repr__(self) -> str:
        return f"<{self.title}>"

    def describe(self) -> str:
        return f"Title: {self.title}\nDetails: {self.details.strip()}\nFeatures: {self.features.strip()}\nURL: {self.url}"

    @classmethod
    def fetch(
        cls,
        feeds: Iterable[str] = DEFAULT_FEEDS,
        per_feed: int = 10,
        show_progress: bool = False,
        delay: float = 0.05,
    ) -> List["ScrapedDeal"]:
        """Retrieve deals from the feeds. A broken entry/feed is logged and skipped."""
        import feedparser

        deals: List[ScrapedDeal] = []
        feed_list = list(feeds)
        if show_progress:
            from tqdm import tqdm

            feed_list = tqdm(feed_list)
        for feed_url in feed_list:
            try:
                feed = feedparser.parse(feed_url)
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not parse feed %s: %s", feed_url, exc)
                continue
            for entry in feed.entries[:per_feed]:
                try:
                    deals.append(cls(entry))
                except Exception as exc:  # noqa: BLE001
                    log.warning("Skipping entry from %s: %s", feed_url, exc)
                    continue
                time.sleep(delay)
        return deals
