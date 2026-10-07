import sys
import types

from price_model_service.preprocessor import Preprocessor, SYSTEM_PROMPT


def fake_litellm(monkeypatch, record):
    def completion(**kw):
        record.update(kw)
        msg = types.SimpleNamespace(content="Title: Mic")
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=msg)],
            usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            _hidden_params={"response_cost": 0.002},
        )

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=completion))


def test_preprocess_tracks_usage(monkeypatch):
    rec = {}
    fake_litellm(monkeypatch, rec)
    p = Preprocessor("gpt-4o-mini")
    assert p.preprocess("raw text") == "Title: Mic"
    assert rec["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert rec["messages"][1]["content"] == "raw text"
    assert (p.total_input_tokens, p.total_output_tokens, p.total_cost) == (10, 5, 0.002)


def test_ollama_default_base_url_and_env_model(monkeypatch):
    p = Preprocessor()
    assert p.model_name == "ollama/llama3.2" and p.base_url == "http://localhost:11434"
    monkeypatch.setenv("PRICER_PREPROCESSOR_MODEL", "groq/openai/gpt-oss-20b")
    p = Preprocessor()
    assert p.reasoning_effort == "low" and p.base_url is None
