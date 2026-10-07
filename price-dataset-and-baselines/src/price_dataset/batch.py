"""Groq Batch API runner for summarising items.

Divides items into chunks of 1,000, uploads one JSONL per chunk, submits the batches,
and later collects the outputs back into ``item.summary``.

Design notes:
* no module-level client / class-level mutable list - everything is instance state;
* the Groq client is injected (so it can be mocked) and created lazily;
* ``open(..., 'rb', encoding=...)`` (a ValueError in Python) fixed;
* resumable state is JSON, not pickle;
* results are matched on ``custom_id`` -> item.id through a dict, not list position.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from price_dataset.preprocessor import SYSTEM_PROMPT

MODEL = "openai/gpt-oss-20b"
BATCH_SIZE = 1_000


def make_jsonl_line(item, model: str = MODEL) -> str:
    """One Groq batch request line. custom_id is the item's integer id."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.full},
        ],
        "reasoning_effort": "low",
    }
    line = {
        "custom_id": str(item.id),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }
    return json.dumps(line)


def parse_output_line(line: str) -> tuple[int, str]:
    """-> (item id, summary text) from one line of the batch output file."""
    obj = json.loads(line)
    return int(obj["custom_id"]), obj["response"]["body"]["choices"][0]["message"]["content"]


def get_client():
    from groq import Groq

    return Groq(api_key=os.environ.get("GROQ_API_KEY"))


class Batch:
    """One chunk [start, end) of the item list."""

    def __init__(self, start: int, end: int, folder: Path):
        self.start = start
        self.end = end
        self.filename = f"{start}_{end}.jsonl"
        self.folder = Path(folder)
        self.file_id: str | None = None
        self.batch_id: str | None = None
        self.output_file_id: str | None = None
        self.done = False

    @property
    def input_path(self) -> Path:
        return self.folder / "batches" / self.filename

    @property
    def output_path(self) -> Path:
        return self.folder / "output" / self.filename

    def make_file(self, items) -> None:
        self.input_path.parent.mkdir(parents=True, exist_ok=True)
        with self.input_path.open("w", encoding="utf-8") as f:
            for item in items[self.start : self.end]:
                f.write(make_jsonl_line(item))
                f.write("\n")

    def send_file(self, client) -> None:
        with self.input_path.open("rb") as f:
            self.file_id = client.files.create(file=f, purpose="batch").id

    def submit(self, client) -> None:
        response = client.batches.create(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=self.file_id,
        )
        self.batch_id = response.id

    def is_ready(self, client) -> bool:
        response = client.batches.retrieve(self.batch_id)
        if response.status == "completed":
            self.output_file_id = response.output_file_id
            return True
        if response.status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch {self.batch_id} ({self.filename}) ended with status {response.status}")
        return False

    def fetch_output(self, client) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        client.files.content(self.output_file_id).write_to_file(str(self.output_path))

    def apply_output(self, by_id: dict) -> int:
        """Write summaries into the items. Returns how many were applied."""
        applied = 0
        with self.output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item_id, summary = parse_output_line(line)
                by_id[item_id].summary = summary
                applied += 1
        self.done = True
        return applied

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "file_id": self.file_id,
            "batch_id": self.batch_id,
            "output_file_id": self.output_file_id,
            "done": self.done,
        }


class BatchRunner:
    """Create / submit / fetch all batches for a list of items (ids must be assigned)."""

    def __init__(self, items, folder: str | Path, client=None, batch_size: int = BATCH_SIZE):
        self.items = items
        self.folder = Path(folder)
        self.client = client
        self.batch_size = batch_size
        self.batches: list[Batch] = []
        self.state_file = self.folder / "batches_state.json"

    def _client(self):
        if self.client is None:
            self.client = get_client()
        return self.client

    def create(self) -> None:
        self.batches = [
            Batch(start, min(start + self.batch_size, len(self.items)), self.folder)
            for start in range(0, len(self.items), self.batch_size)
        ]
        print(f"Created {len(self.batches)} batches")

    def run(self) -> None:
        """Write, upload and submit every batch that is not submitted yet; saves state as it goes."""
        client = self._client()
        for batch in self.batches:
            if batch.batch_id:
                continue
            batch.make_file(self.items)
            batch.send_file(client)
            batch.submit(client)
            self.save()
        print(f"Submitted {len(self.batches)} batches")

    def fetch(self) -> tuple[int, int]:
        """Collect finished batches. Returns (finished, total)."""
        client = self._client()
        by_id = {item.id: item for item in self.items}
        for batch in self.batches:
            if not batch.done and batch.is_ready(client):
                batch.fetch_output(client)
                batch.apply_output(by_id)
                self.save()
        finished = sum(b.done for b in self.batches)
        print(f"Finished {finished} of {len(self.batches)} batches")
        return finished, len(self.batches)

    def save(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps([b.to_dict() for b in self.batches], indent=2), encoding="utf-8")

    def load(self) -> None:
        raw = json.loads(self.state_file.read_text(encoding="utf-8"))
        self.batches = []
        for d in raw:
            b = Batch(d["start"], d["end"], self.folder)
            b.file_id, b.batch_id = d["file_id"], d["batch_id"]
            b.output_file_id, b.done = d["output_file_id"], d["done"]
            self.batches.append(b)
        print(f"Loaded {len(self.batches)} batches")

    def apply_finished_outputs(self) -> int:
        """Re-apply already-downloaded outputs (e.g. after a restart) without calling the API."""
        by_id = {item.id: item for item in self.items}
        n = 0
        for b in self.batches:
            if b.output_path.exists():
                n += b.apply_output(by_id)
        return n


def missing_summaries(items) -> list[int]:
    return [i for i, item in enumerate(items) if not item.summary]
