"""Scraping helpers (a minimal requests + BeautifulSoup scraper)."""

from __future__ import annotations

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
    )
}
TIMEOUT_SECONDS = 20


class ScrapeError(RuntimeError):
    """Raised when a page cannot be fetched."""


def _soup(url: str) -> BeautifulSoup:
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ScrapeError(f"Could not fetch {url}: {exc}") from exc
    return BeautifulSoup(response.content, "html.parser")


def fetch_website_contents(url: str, max_chars: int = 2_000) -> str:
    """Title + visible body text of ``url``, truncated to ``max_chars``."""
    soup = _soup(url)
    title = soup.title.string if soup.title and soup.title.string else "No title found"
    if soup.body:
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        text = soup.body.get_text(separator="\n", strip=True)
    else:
        text = ""
    return (title + "\n\n" + text)[:max_chars]


def fetch_website_links(url: str) -> list[str]:
    """All ``href`` values on the page (may be relative)."""
    soup = _soup(url)
    links = [a.get("href") for a in soup.find_all("a")]
    return [link for link in links if link]
