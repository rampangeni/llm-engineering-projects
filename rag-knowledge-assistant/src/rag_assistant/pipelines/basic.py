"""Basic tier: LangChain recursive splitter, single-vector retrieval, one LLM call."""

from __future__ import annotations

from ..config import Settings
from ..embeddings import Embedder, make_embedder
from ..llm import LLM, LiteLLMClient
from ..models import Document, Result
from ..store import VectorStore
from .base import Pipeline

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""


def create_chunks(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Result]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Result] = []
    for doc in documents:
        for piece in splitter.split_text(doc.text):
            chunks.append(Result(page_content=piece, metadata={"source": doc.source, "type": doc.type}))
    return chunks


def combined_question(question: str, history: list[dict] | None = None) -> str:
    """Retrieval query = all prior user turns + current question (cheap conversational memory)."""
    prior = "\n".join(m["content"] for m in (history or []) if m["role"] == "user")
    return prior + "\n" + question


class BasicPipeline(Pipeline):
    name = "basic"

    def __init__(
        self,
        settings: Settings,
        llm: LLM | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
    ):
        self.settings = settings
        self.llm = llm or LiteLLMClient(settings.basic_llm_model)
        self._embedder = embedder
        self._store = store

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            emb = self._embedder or make_embedder(
                self.settings.embedding_spec("basic"), self.settings.embed_batch_size
            )
            self._store = VectorStore(self.settings.db_dir("basic"), emb)
        return self._store

    def ingest(self, documents: list[Document]) -> int:
        chunks = create_chunks(documents, self.settings.chunk_size, self.settings.chunk_overlap)
        return self.store.rebuild(chunks)

    def fetch_context(self, question: str, history: list[dict] | None = None) -> list[Result]:
        return self.store.query(combined_question(question, history), self.settings.basic_retrieval_k)

    def answer(self, question: str, history: list[dict] | None = None) -> tuple[str, list[Result]]:
        history = history or []
        docs = self.fetch_context(question, history)
        context = "\n\n".join(d.page_content for d in docs)
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
        messages += [{"role": m["role"], "content": m["content"]} for m in history]
        messages.append({"role": "user", "content": question})
        return self.llm.chat(messages), docs
