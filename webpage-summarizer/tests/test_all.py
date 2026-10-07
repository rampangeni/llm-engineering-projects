import pytest

from webpage_summarizer import config, scraper
from webpage_summarizer.cli import main
from webpage_summarizer.summarizer import messages_for, stream_summary, summarize


def test_scraper_extracts_title_and_text(fake_requests):
    text = scraper.fetch_website_contents("http://x")
    assert text.startswith("Hello Site")
    assert "Welcome" in text and "var x" not in text


def test_scraper_truncates(fake_requests):
    assert len(scraper.fetch_website_contents("http://x", max_chars=10)) == 10


def test_scraper_wraps_errors(monkeypatch):
    def boom(*a, **k):
        raise scraper.requests.ConnectionError("nope")

    monkeypatch.setattr("webpage_summarizer.scraper.requests.get", boom)
    with pytest.raises(scraper.ScrapeError):
        scraper.fetch_website_contents("http://x")


def test_messages_shape():
    msgs = messages_for("site text")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"].endswith("site text")


def test_summarize_uses_model(fake_requests, fake_client, monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    settings = config.Settings("openai", "my-model", None, "k")
    out = summarize("http://x", settings=settings, client=fake_client)
    assert out == "A snarky summary."
    assert fake_client.comp.calls[0]["model"] == "my-model"


def test_stream_summary_cumulative(fake_requests, fake_client):
    settings = config.Settings("ollama", "llama3.2", "u", "ollama")
    parts = list(stream_summary("http://x", settings=settings, client=fake_client))
    assert parts[-1].strip() == "A snarky summary."
    assert len(parts) > 1


def test_ollama_settings(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda **k: None)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    s = config.load_settings("ollama")
    assert s.model == "llama3.2" and s.base_url.endswith(":11434/v1")


def test_openai_requires_key(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda **k: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        config.make_client(config.load_settings("openai"))


def test_bad_backend():
    with pytest.raises(ValueError):
        config.load_settings("nope")


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    assert "--backend" in capsys.readouterr().out
