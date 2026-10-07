import types

import numpy as np
import pytest

from deal_hunter.config import Settings
from deal_hunter.models import Deal, DealSelection, Opportunity


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    for k in ["PUSHOVER_USER", "PUSHOVER_TOKEN", "ENSEMBLE_WEIGHTS", "PLANNER", "DEAL_THRESHOLD", "DEAL_FEEDS"]:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def settings(tmp_path):
    return Settings(memory_path=str(tmp_path / "memory.json"), chroma_path=str(tmp_path / "chroma"),
                    nn_weights_path=str(tmp_path / "missing.pth"))


def make_deal(url="https://x/1", price=100.0, desc="A thing"):
    return Deal(product_description=desc, price=price, url=url)


@pytest.fixture
def deal():
    return make_deal()


def make_selection(*prices):
    return DealSelection(deals=[make_deal(f"https://x/{i}", p, f"item {i}") for i, p in enumerate(prices)])


class FakePricer:
    def __init__(self, value=100.0, fail=False):
        self.value, self.fail, self.seen = value, fail, []

    def price(self, text):
        self.seen.append(text)
        if self.fail:
            raise RuntimeError("down")
        return self.value


class FakeEncoder:
    def encode(self, texts):
        return np.array([[float(len(t)), 1.0, 0.0] for t in texts])


class FakeCollection:
    """Minimal in-memory Chroma stand-in."""

    def __init__(self, docs=None, prices=None):
        self.docs, self.metas, self.ids, self.embeddings = [], [], [], []
        self.query_args = None
        if docs:
            self.docs, self.metas = list(docs), [{"price": p, "category": "Electronics"} for p in prices]

    def add(self, ids, documents, embeddings, metadatas):
        self.ids += ids
        self.docs += documents
        self.embeddings += embeddings
        self.metas += metadatas

    def count(self):
        return len(self.docs)

    def query(self, query_embeddings, n_results):
        self.query_args = (query_embeddings, n_results)
        return {"documents": [self.docs[:n_results]], "metadatas": [self.metas[:n_results]]}


def fake_openai(*responses, parsed=None):
    """OpenAI-like client: chat.completions.create pops scripted responses; .parse returns ``parsed``."""
    calls = {"create": [], "parse": []}
    queue = list(responses)

    def create(**kw):
        calls["create"].append(kw)
        return queue.pop(0)

    def parse(**kw):
        calls["parse"].append(kw)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(parsed=parsed))])

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create, parse=parse)))
    client.calls = calls
    return client


def text_response(text):
    msg = types.SimpleNamespace(content=text, tool_calls=None)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")])


def tool_response(*calls):
    """calls: (id, name, json_arguments)"""
    tcs = [types.SimpleNamespace(id=i, function=types.SimpleNamespace(name=n, arguments=a)) for i, n, a in calls]
    msg = types.SimpleNamespace(content=None, tool_calls=tcs)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="tool_calls")])
