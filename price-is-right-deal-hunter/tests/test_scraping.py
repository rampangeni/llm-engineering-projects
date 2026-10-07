import types

from deal_hunter import scraping
from deal_hunter.scraping import ScrapedDeal, extract

PAGE = b"<html><div class='content-section'>Great mic\nmore Features USB-C XLR</div></html>"


def entry(url="https://d/1", title="T" * 150, summary="<div class='snippet summary'>Nice <b>mic</b></div>"):
    return {"title": title, "summary": summary, "links": [{"href": url}]}


def test_extract_strips_html():
    assert extract("<div class='snippet summary'>Nice <b>mic</b></div>") == "Nice mic"
    assert extract("plain\ntext") == "plain text"


def test_scraped_deal_parses_and_truncates(monkeypatch):
    monkeypatch.setattr(scraping.requests, "get", lambda url, timeout: types.SimpleNamespace(content=PAGE))
    d = ScrapedDeal(entry())
    assert len(d.title) == 100
    assert d.url == "https://d/1"
    assert d.details.strip().startswith("Great mic") and "USB-C" in d.features
    assert "URL: https://d/1" in d.describe()


def test_scraped_deal_missing_content_section_falls_back(monkeypatch):
    monkeypatch.setattr(scraping.requests, "get", lambda url, timeout: types.SimpleNamespace(content=b"<html></html>"))
    assert ScrapedDeal(entry()).details == "Nice mic"


def test_fetch_skips_broken_entries(monkeypatch):
    import sys

    feed = types.SimpleNamespace(entries=[entry("https://d/1"), entry("https://d/boom"), entry("https://d/3")])
    monkeypatch.setitem(sys.modules, "feedparser", types.SimpleNamespace(parse=lambda url: feed))

    def get(url, timeout):
        if "boom" in url:
            raise ConnectionError("x")
        return types.SimpleNamespace(content=PAGE)

    monkeypatch.setattr(scraping.requests, "get", get)
    deals = ScrapedDeal.fetch(["feed1", "feed2"], per_feed=10, delay=0)
    assert [d.url for d in deals] == ["https://d/1", "https://d/3"] * 2
