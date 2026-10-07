import json
from types import SimpleNamespace

import pytest

from conftest import make_item
from price_dataset.batch import Batch, BatchRunner, make_jsonl_line, missing_summaries, parse_output_line
from price_dataset.preprocessor import SYSTEM_PROMPT, Preprocessor


def test_jsonl_line_format():
    line = json.loads(make_jsonl_line(make_item(7, full="hello")))
    assert line["custom_id"] == "7" and line["method"] == "POST" and line["url"] == "/v1/chat/completions"
    assert line["body"]["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert line["body"]["messages"][1]["content"] == "hello"
    assert line["body"]["model"] == "openai/gpt-oss-20b"


def test_parse_output_line():
    raw = json.dumps({"custom_id": "3", "response": {"body": {"choices": [{"message": {"content": "Title: x"}}]}}})
    assert parse_output_line(raw) == (3, "Title: x")


class FakeClient:
    """Mimics the bits of the Groq SDK that BatchRunner touches."""

    def __init__(self, items_by_batch):
        self.sent = []
        self.files = SimpleNamespace(create=self._create, content=self._content)
        self.batches = SimpleNamespace(create=self._submit, retrieve=self._retrieve)
        self._n = 0

    def _create(self, file, purpose):
        self.sent.append(file.read().decode().splitlines())
        return SimpleNamespace(id=f"file{len(self.sent)}")

    def _submit(self, **kw):
        self._n += 1
        return SimpleNamespace(id=f"batch{self._n}")

    def _retrieve(self, batch_id):
        return SimpleNamespace(status="completed", output_file_id="out-" + batch_id)

    def _content(self, output_file_id):
        idx = int(output_file_id.replace("out-batch", "")) - 1
        lines = self.sent[idx]

        def write_to_file(path):
            with open(path, "w", encoding="utf-8") as f:
                for l in lines:
                    cid = json.loads(l)["custom_id"]
                    f.write(json.dumps({"custom_id": cid, "response": {"body": {"choices": [{"message": {"content": f"SUMMARY {cid}"}}]}}}) + "\n")

        return SimpleNamespace(write_to_file=write_to_file)


def test_batch_runner_end_to_end(tmp_path):
    items = [make_item(i, summary="") for i in range(5)]
    for it in items:
        it.summary = None
    client = FakeClient(None)
    runner = BatchRunner(items, tmp_path, client=client, batch_size=2)
    runner.create()
    assert [(b.start, b.end) for b in runner.batches] == [(0, 2), (2, 4), (4, 5)]
    runner.run()
    assert len(client.sent) == 3 and len(client.sent[2]) == 1

    # restart: new runner resumes from the JSON state
    runner2 = BatchRunner(items, tmp_path, client=client, batch_size=2)
    runner2.load()
    assert all(b.batch_id for b in runner2.batches)
    finished, total = runner2.fetch()
    assert (finished, total) == (3, 3)
    assert [i.summary for i in items] == [f"SUMMARY {i}" for i in range(5)]
    assert missing_summaries(items) == []


def test_run_is_idempotent_for_submitted_batches(tmp_path):
    items = [make_item(i) for i in range(2)]
    client = FakeClient(None)
    r = BatchRunner(items, tmp_path, client=client, batch_size=1)
    r.create()
    r.run()
    r.run()
    assert len(client.sent) == 2


def test_failed_batch_raises(tmp_path):
    b = Batch(0, 1, tmp_path)
    b.batch_id = "x"
    client = SimpleNamespace(batches=SimpleNamespace(retrieve=lambda i: SimpleNamespace(status="failed")))
    with pytest.raises(RuntimeError):
        b.is_ready(client)


def test_preprocessor_tracks_usage():
    def fake(**kw):
        assert kw["messages"][0]["role"] == "system"
        assert "reasoning_effort" in kw  # gpt-oss model
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            _hidden_params={"response_cost": 0.001},
            choices=[SimpleNamespace(message=SimpleNamespace(content="Title: x"))],
        )

    p = Preprocessor(completion_fn=fake)
    assert p.preprocess("text") == "Title: x"
    assert (p.total_input_tokens, p.total_output_tokens, p.total_cost) == (10, 5, 0.001)


def test_preprocessor_no_reasoning_for_other_models():
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1), _hidden_params={},
                               choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    Preprocessor("ollama/llama3.2", completion_fn=fake).preprocess("t")
    assert "reasoning_effort" not in seen
