"""Embedding backends. Spec strings look like ``hf:all-MiniLM-L6-v2`` or ``openai:text-embedding-3-large``."""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-large", batch_size: int = 100):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.batch_size = batch_size
        self.name = f"openai:{model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            out.extend(e.embedding for e in resp.data)
        return out


class HFEmbedder:
    """Local sentence-transformers model (needs the ``local`` extra)."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "HF embeddings need sentence-transformers: pip install 'rag-knowledge-assistant[local]' "
                "or set BASIC_EMBEDDING=openai:text-embedding-3-large"
            ) from e
        self.model = SentenceTransformer(model)
        self.name = f"hf:{model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()


def make_embedder(spec: str, batch_size: int = 100) -> Embedder:
    provider, _, model = spec.partition(":")
    if provider == "openai":
        return OpenAIEmbedder(model or "text-embedding-3-large", batch_size)
    if provider == "hf":
        return HFEmbedder(model or "all-MiniLM-L6-v2")
    raise ValueError(f"Unknown embedding spec {spec!r}; use 'openai:<model>' or 'hf:<model>'")
