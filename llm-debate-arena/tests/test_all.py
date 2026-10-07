import json
from types import SimpleNamespace

import pytest

from llm_debate_arena import compare, litellm_tools, llm
from llm_debate_arena.cli import main
from llm_debate_arena.debate import (Bot, DebateConfig, DebateEngine, Turn, build_messages,
                                     config_from_dict, config_to_dict)
from llm_debate_arena.presets import PRESETS, classic, get_preset


def test_parse_model():
    assert llm.parse_model("anthropic:claude-haiku-4-5") == ("anthropic", "claude-haiku-4-5")
    assert llm.parse_model("ollama:llama3.2:1b") == ("ollama", "llama3.2:1b")
    assert llm.parse_model("gpt-4.1-mini") == ("openai", "gpt-4.1-mini")
    assert llm.parse_model("llama3.2:1b") == ("openai", "llama3.2:1b")


def test_build_messages_two_bots_classic_layout():
    cfg = classic()
    gpt, claude = cfg.bots
    m = build_messages(gpt, cfg.seed, multi_party=False)
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user"]  # leading user inserted
    assert m[2]["content"] == "Hi there" and m[3]["content"] == "Hi"
    c = build_messages(claude, cfg.seed, multi_party=False)
    assert [x["role"] for x in c] == ["system", "user", "assistant"]


def test_multiparty_prefix_and_merge():
    bots = [Bot("A", "x", "sa"), Bot("B", "x", "sb"), Bot("C", "x", "sc")]
    tr = [Turn("Moderator", "topic"), Turn("B", "b says"), Turn("C", "c says")]
    m = build_messages(bots[0], tr, multi_party=True)
    assert len(m) == 2 and m[1]["role"] == "user"
    assert "B: b says" in m[1]["content"] and "C: c says" in m[1]["content"]


def test_engine_round_robin_and_alternation():
    calls = []

    def chat(model, messages):
        calls.append((model, messages))
        return f"reply{len(calls)}"

    cfg = DebateConfig([Bot("A", "m1", "sa"), Bot("B", "m2", "sb")], rounds=2, topic="cats")
    eng = DebateEngine(cfg, chat)
    turns = list(eng.run())
    assert [t.speaker for t in turns] == ["A", "B", "A", "B"]
    assert [c[0] for c in calls] == ["m1", "m2", "m1", "m2"]
    assert "cats" in calls[0][1][1]["content"]
    # B's last prompt contains A's latest reply as final user message
    assert calls[3][1][-1] == {"role": "user", "content": "reply3"}


def test_config_roundtrip_and_validation():
    for name in PRESETS:
        cfg = get_preset(name)
        assert config_from_dict(json.loads(json.dumps(config_to_dict(cfg)))).bots[0].name == cfg.bots[0].name
    with pytest.raises(ValueError):
        config_from_dict({"bots": [{"name": "A", "model": "m", "system_prompt": "s"}]})
    with pytest.raises(ValueError):
        get_preset("nope")


class FakeRouter:
    def chat(self, spec, messages, **kw):
        if spec.startswith("bad"):
            raise llm.ProviderError("boom")
        return llm.Completion(spec, f"answer from {spec}", 5, 7, 0.1)


def test_compare_collects_errors():
    rows = compare.compare_models("hi", ["openai:a", "bad:b"], router=FakeRouter())
    assert rows[0].text == "answer from openai:a" and rows[0].error is None
    assert rows[1].error and "boom" in rows[1].error
    assert "ERROR" in compare.format_rows(rows)


def test_router_missing_key(monkeypatch):
    monkeypatch.setattr(llm, "load_dotenv", lambda **k: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(llm.ProviderError):
        llm.LLMRouter().client("anthropic")
    with pytest.raises(llm.ProviderError):
        llm.LLMRouter().client("nope")


def test_router_chat_with_fake_client():
    r = llm.LLMRouter()
    seen = {}

    def create(**kw):
        seen.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                               usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2))

    r._clients["ollama"] = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    c = r.chat("ollama:llama3.2:1b", [{"role": "user", "content": "x"}])
    assert c.text == "ok" and seen["model"] == "llama3.2:1b" and c.input_tokens == 3


def test_litellm_name_conversion():
    assert litellm_tools.to_litellm_name("anthropic:claude-haiku-4-5") == "anthropic/claude-haiku-4-5"
    assert litellm_tools.to_litellm_name("gemini/gemini-3.1-flash-lite") == "gemini/gemini-3.1-flash-lite"
    assert litellm_tools.to_litellm_name("ollama:llama3.2") == "ollama_chat/llama3.2"
    assert litellm_tools.to_litellm_name("gpt-4.1") == "openai/gpt-4.1"


def test_cache_demo_with_fake_completion():
    n = {"i": 0}

    def fake(model, messages):
        n["i"] += 1
        cached = 1000 if n["i"] == 3 else 0
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Speak, man."))],
            usage=SimpleNamespace(prompt_tokens=50 if n["i"] == 1 else 60000, completion_tokens=5,
                                  prompt_tokens_details=SimpleNamespace(cached_tokens=cached)),
            _hidden_params={"response_cost": 0.0001})

    res = litellm_tools.cache_demo(completion_fn=fake)
    assert [r["cached_tokens"] for r in res] == [0, 0, 1000]
    assert len(litellm_tools.load_hamlet()) > 100_000


def test_complete_with_cost_fake():
    fake = lambda model, messages: SimpleNamespace(  # noqa: E731
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2), _hidden_params={"response_cost": 0.01})
    c = litellm_tools.complete_with_cost("openai:gpt-4.1", [], completion_fn=fake)
    assert c.cost_cents == pytest.approx(1.0)


def test_cli(capsys):
    assert main(["presets"]) == 0
    assert "classic" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
