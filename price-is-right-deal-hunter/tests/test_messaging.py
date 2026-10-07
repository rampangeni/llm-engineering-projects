import types

from deal_hunter.agents.messaging import MessagingAgent, PUSHOVER_URL
from deal_hunter.config import Settings
from deal_hunter.models import Deal, Opportunity

OPP = Opportunity(deal=Deal(product_description="Samsung TV " * 20, price=300, url="https://u"), estimate=1000, discount=700)


def agent(post=None, completion=None, **kw):
    s = Settings(pushover_user="u", pushover_token="t", **kw)
    return MessagingAgent(s, completion_fn=completion, http_post=post)


def test_dry_run_when_unconfigured():
    posts = []
    a = MessagingAgent(Settings(), http_post=lambda *a, **k: posts.append(1))
    assert a.push("hi") is False and posts == []


def test_push_payload():
    rec = {}

    def post(url, data, timeout):
        rec.update(url=url, data=data, timeout=timeout)
        return types.SimpleNamespace(status_code=200)

    assert agent(post).push("hello") is True
    assert rec["url"] == PUSHOVER_URL and rec["data"]["message"] == "hello" and rec["data"]["user"] == "u"


def test_push_failure_does_not_raise():
    def post(*a, **k):
        raise ConnectionError("offline")

    assert agent(post).push("x") is False
    assert agent(lambda *a, **k: types.SimpleNamespace(status_code=500)).push("x") is False


def test_alert_text():
    rec = {}
    a = agent(lambda url, data, timeout: rec.update(m=data["message"]) or types.SimpleNamespace(status_code=200))
    a.alert(OPP)
    m = rec["m"]
    assert "Price=$300.00" in m and "Estimate=$1000.00" in m and "Discount=$700.00" in m and m.endswith("https://u")


def test_notify_uses_llm_then_pushes():
    rec = {}

    def completion(model, messages):
        rec["model"], rec["prompt"] = model, messages[0]["content"]
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="WOW deal"))])

    a = agent(lambda url, data, timeout: rec.update(m=data["message"]) or types.SimpleNamespace(status_code=200), completion)
    assert a.notify("TV", 300, 1000, "https://u") is True
    assert rec["m"] == "WOW deal... https://u" and "Offered Price: 300" in rec["prompt"]
    assert rec["model"] == "claude-sonnet-4-5"


def test_notify_falls_back_when_llm_fails():
    rec = {}

    def completion(**kw):
        raise RuntimeError("no key")

    a = agent(lambda url, data, timeout: rec.update(m=data["message"]) or types.SimpleNamespace(status_code=200), completion)
    a.notify("TV", 300, 1000, "https://u")
    assert "offered at $300.00" in rec["m"]
