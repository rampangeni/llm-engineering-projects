import math

from rag_assistant.documents import load_documents
from rag_assistant.evaluation.evaluator import evaluate_answer, evaluate_all_retrieval, summarize_retrieval
from rag_assistant.evaluation.metrics import calculate_mrr, calculate_ndcg, evaluate_retrieval_docs, load_tests
from rag_assistant.evaluation.dashboard import get_color
from rag_assistant.models import Result, TestQuestion
from rag_assistant.pipelines import get_pipeline
from conftest import FakeEmbedder, FakeLLM


def R(t):
    return Result(page_content=t, metadata={})


def test_mrr_and_ndcg():
    docs = [R("nothing"), R("Avery here"), R("avery again")]
    assert calculate_mrr("AVERY", docs) == 0.5
    assert calculate_mrr("zzz", docs) == 0.0
    # relevances [0,1,1]: dcg = 1/log2(3)+1/log2(4); ideal = 1 + 1/log2(3)
    expected = (1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3))
    assert abs(calculate_ndcg("avery", docs) - expected) < 1e-9
    assert calculate_ndcg("zzz", docs) == 0.0


def test_keyword_coverage():
    t = TestQuestion(question="q", keywords=["Avery", "Lancaster", "Mars"], reference_answer="r", category="c")
    r = evaluate_retrieval_docs(t, [R("Avery Lancaster")])
    assert r.keywords_found == 2 and abs(r.keyword_coverage - 200 / 3) < 1e-9 and r.mrr == 2 / 3


def test_bundled_tests_load():
    from rag_assistant.config import PROJECT_ROOT

    tests = load_tests(PROJECT_ROOT / "data" / "tests.jsonl")
    assert len(tests) == 150 and all(t.keywords for t in tests)


def test_eval_flow(settings):
    p = get_pipeline("basic", settings, llm=FakeLLM("a"), embedder=FakeEmbedder())
    p.ingest(load_documents(settings.knowledge_base_dir))
    tests = load_tests(settings.tests_file)
    rows = [(t, r) for t, r, _ in evaluate_all_retrieval(p, tests)]
    assert summarize_retrieval(rows)["coverage"] == 100.0
    verdict, gen, docs = evaluate_answer(p, FakeLLM(), tests[0])
    assert verdict.accuracy == 4 and gen == "a" and docs


def test_colors():
    assert get_color(0.95, "mrr") == "green" and get_color(0.8, "ndcg") == "orange"
    assert get_color(3.0, "accuracy") == "red"
