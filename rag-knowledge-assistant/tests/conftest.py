import hashlib
import json

import pytest

from rag_assistant.config import Settings
from rag_assistant.models import Chunks, RankOrder, AnswerEval


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder; no network."""

    name = "fake"
    DIM = 64

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.DIM
            for w in t.lower().split():
                h = int(hashlib.md5(w.strip(".,:?!#").encode()).hexdigest(), 16)
                v[h % self.DIM] += 1.0
            out.append(v)
        return out


class FakeLLM:
    """Scripted LLM. Records calls; understands the structured outputs used by the pipelines."""

    def __init__(self, text="fake answer"):
        self.text = text
        self.calls = []
        self.rank_order = None  # None => reverse order

    def chat(self, messages, response_model=None):
        self.calls.append((messages, response_model))
        if response_model is Chunks:
            doc = messages[0]["content"].split("Here is the document:")[1].split("Respond with")[0].strip()
            half = len(doc) // 2
            return Chunks.model_validate(
                {
                    "chunks": [
                        {"headline": "H1", "summary": "S1", "original_text": doc[: half + 10]},
                        {"headline": "H2", "summary": "S2", "original_text": doc[half:]},
                    ]
                }
            )
        if response_model is RankOrder:
            n = messages[1]["content"].count("# CHUNK ID:")
            return RankOrder(order=self.rank_order if self.rank_order is not None else list(range(n, 0, -1)))
        if response_model is AnswerEval:
            return AnswerEval(feedback="ok", accuracy=4, completeness=3, relevance=5)
        if "VERY short" in messages[0]["content"]:
            return "rewritten query"
        return self.text


@pytest.fixture
def kb(tmp_path):
    root = tmp_path / "data" / "knowledge-base"
    (root / "company").mkdir(parents=True)
    (root / "employees").mkdir()
    (root / "company" / "about.md").write_text("# About\n\nInsurellm was founded in 2015 by Avery Lancaster. " * 12)
    (root / "employees" / "avery.md").write_text("# Avery Lancaster\n\nCEO and founder of Insurellm in San Francisco.")
    (tmp_path / "data" / "tests.jsonl").write_text(
        json.dumps({"question": "Who founded Insurellm?", "keywords": ["Avery", "Lancaster"], "reference_answer": "Avery Lancaster.", "category": "direct_fact"})
        + "\n"
    )
    return root


@pytest.fixture
def settings(kb, tmp_path):
    return Settings(data_dir=tmp_path / "data")
