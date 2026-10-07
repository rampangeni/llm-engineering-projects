"""Runs retrieval and LLM-as-judge evaluation for a pipeline."""

from __future__ import annotations

from typing import Iterator

from ..llm import LLM
from ..models import AnswerEval, Result, RetrievalEval, TestQuestion
from ..pipelines.base import Pipeline
from .metrics import evaluate_retrieval_docs

JUDGE_SYSTEM = (
    "You are an expert evaluator assessing the quality of answers. Evaluate the generated answer by "
    "comparing it to the reference answer. Only give 5/5 scores for perfect answers."
)


def judge_prompt(test: TestQuestion, generated: str) -> str:
    return f"""Question:
{test.question}

Generated Answer:
{generated}

Reference Answer:
{test.reference_answer}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1."""


def evaluate_retrieval(pipeline: Pipeline, test: TestQuestion, k: int = 10) -> RetrievalEval:
    return evaluate_retrieval_docs(test, pipeline.fetch_context(test.question), k)


def evaluate_answer(
    pipeline: Pipeline, judge: LLM, test: TestQuestion
) -> tuple[AnswerEval, str, list[Result]]:
    generated, docs = pipeline.answer(test.question)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": judge_prompt(test, generated)},
    ]
    verdict = judge.chat(messages, response_model=AnswerEval)
    return verdict, generated, docs


def evaluate_all_retrieval(
    pipeline: Pipeline, tests: list[TestQuestion], k: int = 10
) -> Iterator[tuple[TestQuestion, RetrievalEval, float]]:
    for i, test in enumerate(tests, start=1):
        yield test, evaluate_retrieval(pipeline, test, k), i / len(tests)


def evaluate_all_answers(
    pipeline: Pipeline, judge: LLM, tests: list[TestQuestion]
) -> Iterator[tuple[TestQuestion, AnswerEval, float]]:
    for i, test in enumerate(tests, start=1):
        yield test, evaluate_answer(pipeline, judge, test)[0], i / len(tests)


def run_single(pipeline: Pipeline, judge: LLM, tests: list[TestQuestion], n: int) -> None:
    """Print a full report for test number ``n`` (used by ``rag eval --test N``)."""
    if not 0 <= n < len(tests):
        raise IndexError(f"test number must be between 0 and {len(tests) - 1}")
    t = tests[n]
    bar = "=" * 80
    print(f"\n{bar}\nTest #{n} [{t.category}]\n{bar}")
    print(f"Question: {t.question}\nKeywords: {t.keywords}\nReference: {t.reference_answer}")
    r = evaluate_retrieval(pipeline, t)
    print(f"\n-- Retrieval --\nMRR: {r.mrr:.4f}\nnDCG: {r.ndcg:.4f}")
    print(f"Keywords found: {r.keywords_found}/{r.total_keywords} ({r.keyword_coverage:.1f}%)")
    a, generated, _ = evaluate_answer(pipeline, judge, t)
    print(f"\n-- Answer --\n{generated}\n\nFeedback: {a.feedback}")
    print(f"Accuracy {a.accuracy:.2f}/5  Completeness {a.completeness:.2f}/5  Relevance {a.relevance:.2f}/5\n")


def summarize_retrieval(rows: list[tuple[TestQuestion, RetrievalEval]]) -> dict:
    n = len(rows) or 1
    return {
        "count": len(rows),
        "mrr": sum(r.mrr for _, r in rows) / n,
        "ndcg": sum(r.ndcg for _, r in rows) / n,
        "coverage": sum(r.keyword_coverage for _, r in rows) / n,
    }


def summarize_answers(rows: list[tuple[TestQuestion, AnswerEval]]) -> dict:
    n = len(rows) or 1
    return {
        "count": len(rows),
        "accuracy": sum(r.accuracy for _, r in rows) / n,
        "completeness": sum(r.completeness for _, r in rows) / n,
        "relevance": sum(r.relevance for _, r in rows) / n,
    }
