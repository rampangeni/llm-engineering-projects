from rag_assistant import app
from rag_assistant.cli import build_parser
from rag_assistant.config import Settings
from rag_assistant.documents import load_documents
from rag_assistant.pipelines import get_pipeline
from conftest import FakeEmbedder, FakeLLM


def test_cli_parser():
    a = build_parser().parse_args(["eval", "--pipeline", "pro", "--test", "3"])
    assert a.pipeline == "pro" and a.test == 3


def test_settings_env(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_PIPELINE", "pro")
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    s = Settings.from_env()
    assert s.pipeline == "pro" and s.knowledge_base_dir == tmp_path / "knowledge-base"


def test_chat_handler(settings, monkeypatch):
    p = get_pipeline("basic", settings, llm=FakeLLM("hello"), embedder=FakeEmbedder())
    p.ingest(load_documents(settings.knowledge_base_dir))
    monkeypatch.setitem(app._cache, "basic", p)
    history, ctx = app.chat([{"role": "user", "content": "Who founded it?"}], "basic")
    assert history[-1] == {"role": "assistant", "content": "hello"}
    assert "Source:" in ctx


def test_chat_handler_reports_errors(monkeypatch):
    class Boom:
        def answer(self, *a):
            raise RuntimeError("no db")

    monkeypatch.setitem(app._cache, "basic", Boom())
    history, _ = app.chat([{"role": "user", "content": "x"}], "basic")
    assert "no db" in history[-1]["content"]


def test_ui_builds():
    app.build_ui()
