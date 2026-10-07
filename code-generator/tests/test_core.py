import shutil
from types import SimpleNamespace

import pytest

from code_generator import runner
from code_generator.generator import port, stream_port, strip_fences
from code_generator.models import ModelSpec, find_model
from code_generator.prompts import messages_for
from code_generator.samples import PI, PYTHON_HARD
from code_generator.system_info import pick_cpp_compile_command


class FakeClient:
    def __init__(self, text):
        self.text = text
        self.kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.kwargs = kw
        parts = [self.text[i : i + 5] for i in range(0, len(self.text), 5)]
        yield SimpleNamespace(choices=[])  # keep-alive chunk without choices
        for p in parts:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=p))])


def test_strip_fences():
    assert strip_fences("```cpp\nint main(){}\n```") == "int main(){}\n"


def test_prompts_mention_language_and_filename():
    m = messages_for("print(1)", "Rust", ["rustc", "main.rs"], {"os": "x"})
    assert "Rust" in m[0]["content"] and "main.rs" in m[1]["content"] and "print(1)" in m[1]["content"]


def test_stream_port_cumulative_and_reasoning_effort():
    fake = FakeClient("```cpp\nint main(){return 0;}\n```")
    out = list(stream_port("gpt-5", "x=1", "C++", ["c"], None, client_factory=lambda p: fake))
    assert out[-1] == "int main(){return 0;}\n"
    assert fake.kwargs["reasoning_effort"] == "high" and fake.kwargs["stream"] is True
    assert len(out) > 2


def test_no_reasoning_effort_for_others():
    fake = FakeClient("x")
    port("grok-4", "x", "C++", ["c"], client_factory=lambda p: fake)
    assert "reasoning_effort" not in fake.kwargs


def test_unknown_model():
    with pytest.raises(KeyError):
        find_model("nope")
    assert isinstance(find_model("qwen2.5-coder"), ModelSpec)


def test_samples_nonempty():
    assert "calculate" in PI and "max_subarray_sum" in PYTHON_HARD


def test_run_python_ok_error_timeout():
    assert runner.run_python("print('hi')").output.strip() == "hi"
    bad = runner.run_python("raise ValueError('boom')")
    assert not bad.ok and "boom" in bad.output
    t = runner.run_python("while True: pass", timeout=1)
    assert not t.ok and "Timed out" in t.output


def test_env_scrubbed(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    r = runner.run_python("import os; print(os.getenv('OPENAI_API_KEY'))")
    assert r.output.strip() == "None"


def test_pick_cpp_no_compiler(monkeypatch):
    monkeypatch.setattr("code_generator.system_info._which", lambda n: "")
    with pytest.raises(RuntimeError):
        pick_cpp_compile_command({"os": {}})


def test_compile_failure_reported():
    if not (shutil.which("g++") or shutil.which("clang++")):
        pytest.skip("no C++ compiler")
    res = runner.compile_and_run("this is not c++", "C++")
    assert not res.ok and res.output


def test_compile_and_run_cpp():
    if not (shutil.which("g++") or shutil.which("clang++")):
        pytest.skip("no C++ compiler")
    res = runner.compile_and_run('#include <cstdio>\nint main(){std::puts("42");}', "C++")
    assert res.ok and res.output.strip() == "42"


def test_speedup():
    assert runner.speedup(10, 0.5) == 20
    assert runner.speedup(1, 0) is None


def test_cli_parser():
    from code_generator.cli import build_parser

    a = build_parser().parse_args(["port", "f.py", "--language", "Rust", "--run"])
    assert a.language == "Rust" and a.run
