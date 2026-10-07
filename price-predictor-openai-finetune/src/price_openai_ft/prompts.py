"""Chat-format training data for OpenAI fine-tuning, and the matching inference prompt."""

from __future__ import annotations

import json
from pathlib import Path

INSTRUCTION = "Estimate the price of this product. Respond with the price, no explanation"


def user_message(summary: str) -> dict:
    return {"role": "user", "content": f"{INSTRUCTION}\n\n{summary}"}


def messages_for(item) -> list[dict]:
    """Training example: user asks, assistant answers with ``$<price to 2dp>``."""
    return [user_message(item.summary), {"role": "assistant", "content": f"${item.price:.2f}"}]


def test_messages_for(item) -> list[dict]:
    """Inference prompt - identical to training minus the assistant answer."""
    return [user_message(item.summary)]


test_messages_for.__test__ = False  # not a pytest test


def make_jsonl(items) -> str:
    """One ``{"messages": [...]}`` JSON object per line (no trailing newline)."""
    return "\n".join(json.dumps({"messages": messages_for(item)}) for item in items)


def write_jsonl(items, filename: str | Path) -> Path:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(make_jsonl(items) + "\n", encoding="utf-8")
    return path


def validate_jsonl(path: str | Path) -> int:
    """Light structural check of a training file; returns the number of examples."""
    n = 0
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            msgs = obj.get("messages")
            if not msgs or msgs[-1]["role"] != "assistant":
                raise ValueError(f"{path}:{lineno}: last message must be the assistant answer")
            if not msgs[-1]["content"].startswith("$"):
                raise ValueError(f"{path}:{lineno}: assistant content should look like '$12.34'")
            n += 1
    return n
