"""Pro tier: LLM semantic chunking, query rewriting, multi-retrieval, LLM reranking."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ..config import Settings
from ..embeddings import Embedder, make_embedder
from ..llm import LLM, LiteLLMClient
from ..models import Chunks, Document, RankOrder, Result
from ..store import VectorStore
from .base import Pipeline

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""

RERANK_SYSTEM_PROMPT = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""

REWRITE_PROMPT = """
You are in a conversation with a user.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Since the conversation is contextual, understand the meaning of the user question and add details based on the history.
Condense everything in a single contextually-rich VERY short and specific question, most likely to surface content.

EXAMPLE:
user: Who is the founder? -> Query: who is the founder?
assistant: The founder is FooBar
user: What role covers? -> Query: What role FooBar covers?
...

IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""


def make_chunking_prompt(document: Document, avg_chunk_chars: int = 100) -> str:
    how_many = (len(document.text) // avg_chunk_chars) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document.type}
The document has been retrieved from: {document.source}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
This document should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document.text}

Respond with the chunks.
"""


def merge_chunks(primary: list[Result], secondary: list[Result]) -> list[Result]:
    """Union preserving order; secondary chunks are appended only if their text is new."""
    merged = list(primary)
    seen = {c.page_content for c in primary}
    for c in secondary:
        if c.page_content not in seen:
            merged.append(c)
            seen.add(c.page_content)
    return merged


def apply_rank_order(chunks: list[Result], order: list[int]) -> list[Result]:
    """Apply a 1-based LLM ranking defensively: drop invalid/duplicate ids, append omitted chunks."""
    result: list[Result] = []
    used: set[int] = set()
    for i in order:
        if 1 <= i <= len(chunks) and i not in used:
            used.add(i)
            result.append(chunks[i - 1])
    result.extend(c for idx, c in enumerate(chunks, start=1) if idx not in used)
    return result


def make_rag_messages(question: str, history: list[dict], chunks: list[Result]) -> list[dict]:
    context = "\n\n".join(f"Extract from {c.metadata['source']}:\n{c.page_content}" for c in chunks)
    return (
        [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
        + [{"role": m["role"], "content": m["content"]} for m in history]
        + [{"role": "user", "content": question}]
    )


class ProPipeline(Pipeline):
    name = "pro"

    def __init__(
        self,
        settings: Settings,
        llm: LLM | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
    ):
        self.settings = settings
        self.llm = llm or LiteLLMClient(settings.pro_llm_model)
        self._embedder = embedder
        self._store = store

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            emb = self._embedder or make_embedder(
                self.settings.embedding_spec("pro"), self.settings.embed_batch_size
            )
            self._store = VectorStore(self.settings.db_dir("pro"), emb)
        return self._store

    # ---- ingestion -------------------------------------------------------
    def chunk_document(self, document: Document) -> list[Result]:
        prompt = make_chunking_prompt(document, self.settings.pro_avg_chunk_chars)
        parsed = self.llm.chat([{"role": "user", "content": prompt}], response_model=Chunks)
        return [c.as_result(document) for c in parsed.chunks]

    def create_chunks(self, documents: list[Document]) -> list[Result]:
        """One LLM call per document, run on a small thread pool (set PRO workers to 1 on rate limits)."""
        with ThreadPoolExecutor(max_workers=self.settings.pro_workers) as pool:
            results = list(pool.map(self.chunk_document, documents))
        return [c for per_doc in results for c in per_doc]

    def ingest(self, documents: list[Document]) -> int:
        return self.store.rebuild(self.create_chunks(documents))

    # ---- retrieval -------------------------------------------------------
    def rewrite_query(self, question: str, history: list[dict] | None = None) -> str:
        prompt = REWRITE_PROMPT.format(history=history or [], question=question)
        return self.llm.chat([{"role": "system", "content": prompt}]).strip()

    def rerank(self, question: str, chunks: list[Result]) -> list[Result]:
        user = (
            f"The user has asked the following question:\n\n{question}\n\n"
            "Order all the chunks of text by relevance to the question, from most relevant to least relevant. "
            "Include all the chunk ids you are provided with, reranked.\n\nHere are the chunks:\n\n"
        )
        for i, c in enumerate(chunks, start=1):
            user += f"# CHUNK ID: {i}:\n\n{c.page_content}\n\n"
        user += "Reply only with the list of ranked chunk ids, nothing else."
        messages = [
            {"role": "system", "content": RERANK_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        ranking = self.llm.chat(messages, response_model=RankOrder)
        return apply_rank_order(chunks, ranking.order)

    def fetch_context(self, question: str, history: list[dict] | None = None) -> list[Result]:
        rewritten = self.rewrite_query(question, history)
        k = self.settings.pro_retrieval_k
        merged = merge_chunks(self.store.query(question, k), self.store.query(rewritten, k))
        return self.rerank(question, merged)[: self.settings.pro_final_k]

    def answer(self, question: str, history: list[dict] | None = None) -> tuple[str, list[Result]]:
        history = history or []
        chunks = self.fetch_context(question, history)
        return self.llm.chat(make_rag_messages(question, history, chunks)), chunks
