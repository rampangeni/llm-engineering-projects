import base64
import json
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from airline_assistant import app, config
from airline_assistant.assistant import Assistant
from airline_assistant.cli import build_parser
from airline_assistant.media import Media
from airline_assistant.tickets import TicketError, TicketStore
from airline_assistant.tools import build_default_registry

SETTINGS = config.Settings("openai", "m", "img", "tts", "onyx", None, "k", False)


# ---- tickets
def test_store_prices_and_booking():
    s = TicketStore()
    assert s.get_price("London") == 799
    assert s.get_price("atlantis") is None
    s.set_price("Atlantis", 5000)
    assert s.get_price(" atlantis ") == 5000
    b = s.book("Paris", "Ada Lovelace")
    assert b.ticket_id == "FL-0001" and b.price == 899
    assert s.book("Paris", "Bob").ticket_id == "FL-0002"
    with pytest.raises(TicketError):
        s.book("Nowhere", "X")
    with pytest.raises(TicketError):
        s.book("Paris", " ")
    for bad in (0, -1, "9", True):
        with pytest.raises(TicketError):
            s.set_price("paris", bad)


# ---- tools
def test_registry_schemas_and_admin_gating():
    reg = build_default_registry(TicketStore())
    assert reg.names() == ["get_ticket_price", "book_ticket"]
    assert reg.schemas()[0]["function"]["parameters"]["required"] == ["destination_city"]
    assert "set_ticket_price" in build_default_registry(TicketStore(), include_admin=True).names()
    assert reg.dispatch("set_ticket_price", '{"destination_city":"paris","price":1}').content.startswith("Error: unknown")


def test_dispatch_paths():
    store = TicketStore()
    reg = build_default_registry(store, include_admin=True)
    r = reg.dispatch("get_ticket_price", json.dumps({"destination_city": "Tokyo"}))
    assert "$1,420" in r.content and r.meta["city"] == "Tokyo"
    assert "Unknown" in reg.dispatch("get_ticket_price", '{"destination_city":"Mars"}').content
    assert "FL-0001" in reg.dispatch("book_ticket", '{"destination_city":"paris","passenger_name":"Ann"}').content
    assert reg.dispatch("set_ticket_price", '{"destination_city":"paris","price":950}').content.endswith("$950")
    assert store.get_price("paris") == 950
    assert reg.dispatch("book_ticket", '{"destination_city":"Mars","passenger_name":"A"}').content.startswith("Error")
    assert "invalid arguments" in reg.dispatch("get_ticket_price", "{not json").content
    assert "invalid arguments" in reg.dispatch("get_ticket_price", '{"wrong": 1}').content


# ---- assistant with a scripted fake client
def _tool_msg(*calls):
    tcs = [SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=n, arguments=a))
           for i, (n, a) in enumerate(calls)]
    return SimpleNamespace(finish_reason="tool_calls",
                           message=SimpleNamespace(content=None, tool_calls=tcs))


def _final(text):
    return SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=text, tool_calls=None))


class ScriptedClient:
    def __init__(self, *choices):
        self.choices, self.calls = list(choices), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.images = SimpleNamespace(generate=self._image)
        self.audio = SimpleNamespace(speech=SimpleNamespace(create=lambda **k: SimpleNamespace(content=b"MP3")))

    def _create(self, **kw):
        self.calls.append(json.loads(json.dumps(kw, default=str)))
        return SimpleNamespace(choices=[self.choices.pop(0)])

    def _image(self, **kw):
        buf = BytesIO()
        Image.new("RGB", (4, 4), "red").save(buf, format="PNG")
        return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(buf.getvalue()).decode())])


def test_tool_loop_and_media():
    client = ScriptedClient(_tool_msg(("get_ticket_price", '{"destination_city":"Paris"}'),
                                      ("get_ticket_price", '{"destination_city":"London"}')),
                            _final("Paris is $899."))
    reg = build_default_registry(TicketStore())
    a = Assistant(client, SETTINGS, reg, Media(client, SETTINGS))
    reply = a.respond([{"role": "user", "content": "price to paris?"}])
    assert reply.text == "Paris is $899." and reply.cities == ["Paris", "London"]
    assert reply.audio == b"MP3" and reply.image.size == (4, 4)
    second = client.calls[1]["messages"]
    assert [m["role"] for m in second[-4:]] == ["user", "assistant", "tool", "tool"]
    assert second[-1]["tool_call_id"] == "c1" and "$799" in second[-1]["content"]
    assert client.calls[0]["tools"] and client.calls[0]["model"] == "m"


def test_tool_loop_is_bounded():
    endless = [_tool_msg(("get_ticket_price", '{"destination_city":"Paris"}')) for _ in range(20)]
    a = Assistant(ScriptedClient(*endless), SETTINGS, build_default_registry(TicketStore()))
    text, _ = a.complete([{"role": "user", "content": "x"}])
    assert "could not complete" in text


def test_media_failures_are_swallowed():
    broken = SimpleNamespace(images=SimpleNamespace(generate=lambda **k: 1 / 0),
                             audio=SimpleNamespace(speech=SimpleNamespace(create=lambda **k: 1 / 0)))
    m = Media(broken, SETTINGS)
    assert m.draw_city("Paris") is None and m.speak("hi") is None


# ---- UI helpers
def test_ui_helpers(tmp_path):
    assert app.put_message_in_chatbot("hi", []) == ("", [{"role": "user", "content": "hi"}])
    path = app.save_audio(b"abc")
    assert open(path, "rb").read() == b"abc" and app.save_audio(None) is None
    client = ScriptedClient(_final("Hello!"))
    a = Assistant(client, SETTINGS, build_default_registry(TicketStore()))
    hist, audio, image = app.make_chat_fn(a)([{"role": "user", "content": "hi"}])
    assert hist[-1] == {"role": "assistant", "content": "Hello!"} and audio is None and image is None
    app.build_ui(a)


def test_chat_fn_reports_errors():
    class Boom:
        media = None

        def respond(self, h):
            raise RuntimeError("down")

    hist, _, _ = app.make_chat_fn(Boom())([{"role": "user", "content": "hi"}])
    assert "down" in hist[-1]["content"]


# ---- config
def test_settings(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda **k: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ENABLE_ADMIN_TOOLS", raising=False)
    o = config.load_settings("ollama")
    assert not o.media_supported and o.model == "llama3.2" and not o.admin_tools
    with pytest.raises(RuntimeError):
        config.make_client(config.load_settings("openai"))
    with pytest.raises(ValueError):
        config.load_settings("x")
    monkeypatch.setenv("ENABLE_ADMIN_TOOLS", "1")
    assert config.load_settings("ollama").admin_tools


def test_cli_parser():
    assert build_parser().parse_args(["--chat", "--no-media"]).chat
