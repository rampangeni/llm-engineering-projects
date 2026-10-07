from types import SimpleNamespace

import pytest


class FakeCompletions:
    def __init__(self, reply="A snarky summary."):
        self.reply = reply
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            words = self.reply.split(" ")
            return iter(
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=w + " "))]
                )
                for w in words
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))]
        )


@pytest.fixture
def fake_client():
    comp = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=comp))
    client.comp = comp
    return client


HTML = b"""<html><head><title>Hello Site</title></head><body>
<script>var x=1;</script><h1>Welcome</h1><p>Some text.</p></body></html>"""


@pytest.fixture
def fake_requests(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        return SimpleNamespace(content=HTML, raise_for_status=lambda: None)

    monkeypatch.setattr("webpage_summarizer.scraper.requests.get", fake_get)
