"""Prompt construction and the summarize call."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .config import Settings, load_settings, make_client
from .scraper import fetch_website_contents

SYSTEM_PROMPT = """
You are a snarky assistant that analyzes the contents of a website,
and provides a short, snarky, humorous summary, ignoring text that might be navigation related.
Respond in markdown. Do not wrap the markdown in a code block - respond just with the markdown.
"""

USER_PROMPT_PREFIX = """
Here are the contents of a website.
Provide a short summary of this website.
If it includes news or announcements, then summarize these too.

"""


def messages_for(website: str, system_prompt: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    """Build the chat messages for a scraped website's text."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": USER_PROMPT_PREFIX + website},
    ]


def summarize(
    url: str,
    settings: Settings | None = None,
    client: Any | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    max_chars: int = 2_000,
) -> str:
    """Fetch ``url`` and return the LLM's markdown summary."""
    settings = settings or load_settings()
    client = client or make_client(settings)
    website = fetch_website_contents(url, max_chars=max_chars)
    response = client.chat.completions.create(
        model=settings.model, messages=messages_for(website, system_prompt)
    )
    return response.choices[0].message.content or ""


def stream_summary(
    url: str,
    settings: Settings | None = None,
    client: Any | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    max_chars: int = 2_000,
) -> Iterator[str]:
    """Like :func:`summarize` but yields the cumulative text as it streams in."""
    settings = settings or load_settings()
    client = client or make_client(settings)
    website = fetch_website_contents(url, max_chars=max_chars)
    stream = client.chat.completions.create(
        model=settings.model, messages=messages_for(website, system_prompt), stream=True
    )
    text = ""
    for chunk in stream:
        if not chunk.choices:
            continue
        text += chunk.choices[0].delta.content or ""
        yield text
