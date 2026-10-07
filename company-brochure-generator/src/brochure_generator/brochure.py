"""Two-step pipeline: (1) LLM selects relevant links as JSON, (2) LLM writes the brochure."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any
from urllib.parse import urljoin, urlparse

from . import prompts, scraper
from .providers import Provider, brochure_model, get_provider, link_model, make_client

MAX_PROMPT_CHARS = 5_000


def parse_links_json(text: str) -> dict[str, list[dict[str, str]]]:
    """Parse the model's JSON answer; tolerate code fences / surrounding prose."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"Model did not return JSON: {text[:200]!r}") from None
        data = json.loads(match.group(0))
    links = data.get("links", []) if isinstance(data, dict) else []
    return {"links": [l for l in links if isinstance(l, dict) and l.get("url")]}


def normalise_links(base_url: str, links: list[dict[str, str]]) -> list[dict[str, str]]:
    """Resolve relative URLs against ``base_url``, keep http(s) only, de-duplicate."""
    seen, out = {base_url.rstrip("/")}, []
    for link in links:
        url = urljoin(base_url, link["url"])
        if urlparse(url).scheme not in ("http", "https") or url.rstrip("/") in seen:
            continue
        seen.add(url.rstrip("/"))
        out.append({"type": link.get("type", "page"), "url": url})
    return out


def select_relevant_links(url: str, provider: Provider, client: Any) -> dict[str, list[dict[str, str]]]:
    """Step 1: ask the LLM which links on ``url`` belong in a brochure."""
    model = link_model(provider)
    print(f"Selecting relevant links for {url} by calling {model}")
    links = scraper.fetch_website_links(url)
    kwargs: dict[str, Any] = {}
    if provider.json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompts.LINK_SYSTEM_PROMPT},
            {"role": "user", "content": prompts.links_user_prompt(url, links)},
        ],
        **kwargs,
    )
    result = parse_links_json(response.choices[0].message.content or "")
    result["links"] = normalise_links(url, result["links"])
    print(f"Found {len(result['links'])} relevant links")
    return result


def fetch_page_and_all_relevant_links(url: str, provider: Provider, client: Any) -> str:
    """Landing page text + text of every relevant page, as one string."""
    contents = scraper.fetch_website_contents(url)
    relevant = select_relevant_links(url, provider, client)
    result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"
    for link in relevant["links"]:
        result += f"\n\n### Link: {link['type']}\n"
        try:
            result += scraper.fetch_website_contents(link["url"])
        except scraper.ScrapeError as exc:  # one dead link should not kill the brochure
            result += f"(could not be fetched: {exc})"
    return result


def build_brochure_prompt(company_name: str, url: str, provider: Provider, client: Any,
                          max_chars: int = MAX_PROMPT_CHARS) -> str:
    prompt = prompts.brochure_user_prompt_header(company_name)
    prompt += fetch_page_and_all_relevant_links(url, provider, client)
    return prompt[:max_chars]


def stream_brochure(company_name: str, url: str, model: str = "GPT", tone: str = "professional",
                    client: Any | None = None, max_chars: int = MAX_PROMPT_CHARS) -> Iterator[str]:
    """Step 2 (streaming): yields the cumulative brochure markdown as tokens arrive.

    ``model`` is a provider key: GPT, Claude, Gemini or Ollama.
    """
    provider = get_provider(model)
    if tone not in prompts.TONES:
        raise ValueError(f"Unknown tone {tone!r}; choose from {list(prompts.TONES)}")
    client = client or make_client(provider)
    user_prompt = build_brochure_prompt(company_name, url, provider, client, max_chars)
    stream = client.chat.completions.create(
        model=brochure_model(provider),
        messages=[
            {"role": "system", "content": prompts.TONES[tone]},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )
    text = ""
    for chunk in stream:
        if not chunk.choices:
            continue
        text += chunk.choices[0].delta.content or ""
        yield text


def create_brochure(company_name: str, url: str, model: str = "GPT", tone: str = "professional",
                    client: Any | None = None) -> str:
    """Non-streaming convenience wrapper; returns the final markdown."""
    final = ""
    for final in stream_brochure(company_name, url, model, tone, client):
        pass
    return final
