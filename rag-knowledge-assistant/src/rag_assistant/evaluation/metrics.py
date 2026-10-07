"""Retrieval metrics: keyword-based MRR, binary-relevance nDCG, keyword coverage."""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..models import Result, RetrievalEval, TestQuestion


def load_tests(path: Path) -> list[TestQuestion]:
    tests = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tests.append(TestQuestion(**json.loads(line)))
    return tests


def calculate_mrr(keyword: str, docs: list[Result]) -> float:
    """Reciprocal rank of the first chunk containing the keyword (case-insensitive), else 0."""
    kw = keyword.lower()
    for rank, doc in enumerate(docs, start=1):
        if kw in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    return sum(relevances[i] / math.log2(i + 2) for i in range(min(k, len(relevances))))


def calculate_ndcg(keyword: str, docs: list[Result], k: int = 10) -> float:
    kw = keyword.lower()
    rel = [1 if kw in d.page_content.lower() else 0 for d in docs[:k]]
    idcg = calculate_dcg(sorted(rel, reverse=True), k)
    return calculate_dcg(rel, k) / idcg if idcg > 0 else 0.0


def evaluate_retrieval_docs(test: TestQuestion, docs: list[Result], k: int = 10) -> RetrievalEval:
    mrr = [calculate_mrr(kw, docs) for kw in test.keywords]
    ndcg = [calculate_ndcg(kw, docs, k) for kw in test.keywords]
    found = sum(1 for s in mrr if s > 0)
    total = len(test.keywords)
    return RetrievalEval(
        mrr=sum(mrr) / total if total else 0.0,
        ndcg=sum(ndcg) / total if total else 0.0,
        keywords_found=found,
        total_keywords=total,
        keyword_coverage=found / total * 100 if total else 0.0,
    )
