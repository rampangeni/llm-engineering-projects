"""Shared data models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A raw knowledge-base file."""

    source: str = Field(description="Path relative to the knowledge base root, e.g. company/about.md")
    type: str = Field(description="Top-level folder: company, contracts, employees, products")
    text: str


class Result(BaseModel):
    """A retrievable chunk: text plus metadata (always has 'source' and 'type')."""

    page_content: str
    metadata: dict


class Chunk(BaseModel):
    """An LLM-produced semantic chunk (pro pipeline)."""

    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query"
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def as_result(self, document: Document) -> Result:
        return Result(
            page_content=f"{self.headline}\n\n{self.summary}\n\n{self.original_text}",
            metadata={"source": document.source, "type": document.type},
        )


class Chunks(BaseModel):
    chunks: list[Chunk]


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


class TestQuestion(BaseModel):
    __test__ = False  # not a pytest class

    question: str
    keywords: list[str]
    reference_answer: str
    category: str


class RetrievalEval(BaseModel):
    mrr: float = Field(description="Mean Reciprocal Rank, averaged across keywords")
    ndcg: float = Field(description="nDCG with binary relevance, averaged across keywords")
    keywords_found: int
    total_keywords: int
    keyword_coverage: float = Field(description="Percentage of keywords found in the top-k")


class AnswerEval(BaseModel):
    feedback: str = Field(
        description="Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context"
    )
    accuracy: float = Field(
        description="1 (wrong; any wrong answer must score 1) to 5 (perfectly accurate). An acceptable answer would score 3."
    )
    completeness: float = Field(
        description="1 (missing key information) to 5 (all information from the reference answer is provided). Only 5 if ALL reference information is included."
    )
    relevance: float = Field(
        description="1 (off-topic) to 5 (directly addresses the question, no additional information). Only 5 if completely relevant with no extra information."
    )
