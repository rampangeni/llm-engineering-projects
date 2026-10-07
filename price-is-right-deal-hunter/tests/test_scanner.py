from conftest import fake_openai, make_selection
from deal_hunter.agents.scanner import DemoScannerAgent, ScannerAgent
from deal_hunter.models import Opportunity


class Scraped:
    def __init__(self, url):
        self.url = url

    def describe(self):
        return f"Title: {self.url}"


def opp(url):
    from conftest import make_deal

    return Opportunity(deal=make_deal(url), estimate=1, discount=1)


def test_fetch_deals_excludes_memory(settings):
    agent = ScannerAgent(settings, client=object(), fetcher=lambda: [Scraped("a"), Scraped("b")])
    assert [s.url for s in agent.fetch_deals([opp("a")])] == ["b"]


def test_scan_uses_structured_output_and_filters_prices(settings):
    client = fake_openai(parsed=make_selection(10, 0, 5))
    agent = ScannerAgent(settings, client=client, fetcher=lambda: [Scraped("a")])
    result = agent.scan([])
    assert [d.price for d in result.deals] == [10, 5]
    call = client.calls["parse"][0]
    assert call["model"] == settings.scanner_model and call["response_format"].__name__ == "DealSelection"
    assert "Title: a" in call["messages"][1]["content"] and call["messages"][1]["content"].endswith("no more.")


def test_scan_returns_none_without_new_deals_and_skips_llm(settings):
    client = fake_openai(parsed=make_selection(1))
    agent = ScannerAgent(settings, client=client, fetcher=lambda: [Scraped("a")])
    assert agent.scan([opp("a")]) is None
    assert client.calls["parse"] == []


def test_scan_handles_unparsed_response(settings):
    agent = ScannerAgent(settings, client=fake_openai(parsed=None), fetcher=lambda: [Scraped("a")])
    assert agent.scan([]) is None


def test_demo_scanner_respects_memory(settings):
    agent = DemoScannerAgent(settings)
    first = agent.scan([])
    seen = [Opportunity(deal=d, estimate=1, discount=1) for d in first.deals]
    assert agent.scan(seen) is None
