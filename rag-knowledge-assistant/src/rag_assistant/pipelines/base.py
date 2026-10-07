"""Pipeline interface shared by the basic and pro tiers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Document, Result


class Pipeline(ABC):
    name: str

    @abstractmethod
    def ingest(self, documents: list[Document]) -> int:
        """Chunk, embed and store documents; return the number of stored chunks."""

    @abstractmethod
    def fetch_context(self, question: str, history: list[dict] | None = None) -> list[Result]:
        """Return the final, ordered context chunks for a question."""

    @abstractmethod
    def answer(self, question: str, history: list[dict] | None = None) -> tuple[str, list[Result]]:
        """Return (answer, context chunks)."""
