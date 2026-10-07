import json
from types import SimpleNamespace

import pytest

from brochure_generator import brochure, providers, scraper
from brochure_generator.cli import main


class FakeCompletions:
    def __init__(self, links_json):
        self.links_json = links_json
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        if kw.get("stream"):
            return iter(SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=w))])
                        for w in ["# Acme", " Brochure"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.links_json))])


@pytest.fixture
def fake(monkeypatch):
    links = json.dumps({"links": [{"type": "about page", "url": "/about"},
                                  {"type": "dup", "url": "https://acme.com/"},
                                  {"type": "mail", "url": "mailto:a@b.c"}]})
    comp = FakeCompletions(links)
    client = SimpleNamespace(chat=SimpleNamespace(completions=comp))
    monkeypatch.setattr(scraper, "fetch_website_links", lambda u: ["/about", "/privacy"])
    monkeypatch.setattr(scraper, "fetch_website_contents", lambda u, max_chars=2000: f"PAGE[{u}]")
    return client, comp


def test_parse_links_plain_and_fenced():
    assert brochure.parse_links_json('{"links": [{"type": "a", "url": "x"}]}')["links"][0]["url"] == "x"
    fenced = 'Sure!\n```json\n{"links": [{"type": "a", "url": "y"}]}\n```'
    assert brochure.parse_links_json(fenced)["links"][0]["url"] == "y"
    with pytest.raises(ValueError):
        brochure.parse_links_json("no json here")


def test_normalise_links():
    out = brochure.normalise_links("https://acme.com", [
        {"type": "about", "url": "/about"}, {"type": "x", "url": "https://acme.com/"},
        {"type": "m", "url": "mailto:a@b.c"}, {"type": "about2", "url": "/about"}])
    assert out == [{"type": "about", "url": "https://acme.com/about"}]


def test_select_links_uses_json_mode_for_gpt(fake):
    client, comp = fake
    res = brochure.select_relevant_links("https://acme.com", providers.PROVIDERS["GPT"], client)
    assert res["links"] == [{"type": "about page", "url": "https://acme.com/about"}]
    assert comp.calls[0]["response_format"] == {"type": "json_object"}
    assert comp.calls[0]["model"] == "gpt-5-nano"


def test_claude_has_no_json_mode(fake):
    client, comp = fake
    brochure.select_relevant_links("https://acme.com", providers.PROVIDERS["Claude"], client)
    assert "response_format" not in comp.calls[0]


def test_stream_brochure_end_to_end(fake):
    client, comp = fake
    parts = list(brochure.stream_brochure("Acme", "https://acme.com", "GPT", client=client))
    assert parts[-1] == "# Acme Brochure"
    final_call = comp.calls[-1]
    assert final_call["stream"] and final_call["model"] == "gpt-4.1-mini"
    user = final_call["messages"][1]["content"]
    assert "Acme" in user and "PAGE[https://acme.com/about]" in user
    assert len(user) <= brochure.MAX_PROMPT_CHARS


def test_humorous_tone_prompt(fake):
    client, comp = fake
    brochure.create_brochure("Acme", "https://acme.com", "GPT", "humorous", client)
    assert "humorous" in comp.calls[-1]["messages"][0]["content"]


def test_bad_inputs(fake):
    client, _ = fake
    with pytest.raises(ValueError):
        list(brochure.stream_brochure("A", "u", "Nope", client=client))
    with pytest.raises(ValueError):
        list(brochure.stream_brochure("A", "u", "GPT", tone="angry", client=client))


def test_missing_key(monkeypatch):
    monkeypatch.setattr(providers, "load_dotenv", lambda **k: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        providers.make_client(providers.PROVIDERS["Claude"])


def test_model_env_override(monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL", "claude-x")
    assert providers.brochure_model(providers.PROVIDERS["Claude"]) == "claude-x"


def test_cli_help():
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
