"""Chroma vector store wrapper (persistent, one collection per pipeline directory)."""

from __future__ import annotations

from pathlib import Path

from .embeddings import Embedder
from .models import Result

COLLECTION = "docs"


class VectorStore:
    def __init__(self, path: Path, embedder: Embedder, collection: str = COLLECTION):
        import chromadb

        self.path = Path(path)
        self.embedder = embedder
        self.collection_name = collection
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))

    @property
    def collection(self):
        return self.client.get_or_create_collection(self.collection_name)

    def count(self) -> int:
        return self.collection.count()

    def rebuild(self, chunks: list[Result], batch_size: int = 200) -> int:
        """Drop and recreate the collection, then embed and add all chunks."""
        if self.collection_name in [c if isinstance(c, str) else c.name for c in self.client.list_collections()]:
            self.client.delete_collection(self.collection_name)
        col = self.client.get_or_create_collection(self.collection_name)
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.page_content for c in batch]
            col.add(
                ids=[str(i + j) for j in range(len(batch))],
                embeddings=self.embedder.embed(texts),
                documents=texts,
                metadatas=[c.metadata for c in batch],
            )
        return col.count()

    def query(self, text: str, k: int) -> list[Result]:
        col = self.collection
        if col.count() == 0:
            raise RuntimeError(f"Vector store at {self.path} is empty. Run `rag ingest` first.")
        vec = self.embedder.embed([text])[0]
        res = col.query(query_embeddings=[vec], n_results=k)
        return [Result(page_content=d, metadata=m) for d, m in zip(res["documents"][0], res["metadatas"][0])]

    def get_all(self) -> tuple[list[list[float]], list[str], list[dict]]:
        res = self.collection.get(include=["embeddings", "documents", "metadatas"])
        return [list(v) for v in res["embeddings"]], res["documents"], res["metadatas"]
