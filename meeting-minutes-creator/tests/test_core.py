from types import SimpleNamespace

import pytest

from meeting_minutes.cli import build_parser, main
from meeting_minutes.config import resolve_backend
from meeting_minutes.minutes import ChatAPI
from meeting_minutes.pipeline import create_minutes
from meeting_minutes.prompts import SYSTEM_MESSAGE, build_messages
from meeting_minutes.transcribe import WhisperAPI


class FakeOpenAI:
    def __init__(self, text="# Minutes\n- item"):
        self.text = text
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=lambda **kw: "hello world "))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.last = None

    def _chat(self, **kw):
        self.last = kw
        yield SimpleNamespace(choices=[])
        for i in range(0, len(self.text), 4):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=self.text[i : i + 4]))])


def test_prompt_contains_transcript_and_markdown():
    m = build_messages("TRANSCRIPT", "a board meeting")
    assert m[0]["content"] == SYSTEM_MESSAGE
    assert "TRANSCRIPT" in m[1]["content"] and "markdown" in m[1]["content"] and "board meeting" in m[1]["content"]


def test_resolve_backend():
    assert resolve_backend("auto", cuda=True) == "local"
    assert resolve_backend("auto", cuda=False) == "api"
    assert resolve_backend("api", cuda=True) == "api"
    with pytest.raises(ValueError):
        resolve_backend("gpu")


def test_chat_api_streams_accumulated():
    fake = FakeOpenAI()
    out = list(ChatAPI(client=fake, model="m").stream("t"))
    assert out[-1] == "# Minutes\n- item" and len(out) > 1
    assert fake.last["stream"] is True and fake.last["model"] == "m"


def test_whisper_api(tmp_path):
    f = tmp_path / "a.mp3"
    f.write_bytes(b"x")
    assert WhisperAPI(client=FakeOpenAI(), model="whisper-1").transcribe(str(f)) == "hello world"


def test_whisper_api_missing_and_too_big(tmp_path):
    w = WhisperAPI(client=FakeOpenAI())
    with pytest.raises(FileNotFoundError):
        w.transcribe(str(tmp_path / "nope.mp3"))
    big = tmp_path / "big.wav"
    with big.open("wb") as fh:
        fh.truncate(26_000_000)
    with pytest.raises(ValueError, match="25 MB"):
        w.transcribe(str(big))


def test_pipeline_audio_end_to_end(tmp_path):
    f = tmp_path / "a.mp3"
    f.write_bytes(b"x")
    fake = FakeOpenAI("done")
    steps = list(
        create_minutes(
            audio_path=str(f), backend="api",
            transcriber=WhisperAPI(client=fake), generator=ChatAPI(client=fake, model="m"),
        )
    )
    assert steps[0] == ("hello world", "")
    assert steps[-1] == ("hello world", "done")


def test_pipeline_transcript_skips_stt():
    fake = FakeOpenAI("ok")
    steps = list(create_minutes(transcript="abc", backend="api", generator=ChatAPI(client=fake, model="m")))
    assert steps[-1] == ("abc", "ok") and all(t == "abc" for t, _ in steps)


def test_pipeline_requires_input():
    with pytest.raises(ValueError):
        list(create_minutes(backend="api"))


def test_cli_requires_input(capsys):
    assert main(["run"]) == 2
    assert build_parser().parse_args(["run", "x.mp3", "--backend", "api"]).backend == "api"
