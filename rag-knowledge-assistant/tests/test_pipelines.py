from rag_assistant.documents import load_documents
from rag_assistant.pipelines import get_pipeline
from rag_assistant.pipelines.basic import combined_question, create_chunks
from rag_assistant.pipelines.pro import apply_rank_order, merge_chunks
from rag_assistant.models import Chunk, Document, Result
from conftest import FakeEmbedder, FakeLLM


def test_load_documents(settings):
    docs = load_documents(settings.knowledge_base_dir)
    assert {d.type for d in docs} == {"company", "employees"}
    assert "company/about.md" in {d.source for d in docs}


def test_basic_chunking_and_metadata(settings):
    docs = load_documents(settings.knowledge_base_dir)
    chunks = create_chunks(docs, 200, 50)
    assert len(chunks) > len(docs)
    assert all(set(c.metadata) == {"source", "type"} for c in chunks)


def test_combined_question():
    hist = [{"role": "user", "content": "Who is Avery?"}, {"role": "assistant", "content": "CEO"}]
    assert combined_question("Where?", hist) == "Who is Avery?\nWhere?"


def test_basic_end_to_end(settings):
    llm = FakeLLM("Avery founded it.")
    p = get_pipeline("basic", settings, llm=llm, embedder=FakeEmbedder())
    assert p.ingest(load_documents(settings.knowledge_base_dir)) > 0
    answer, ctx = p.answer("Who is Avery Lancaster?", [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
    assert answer == "Avery founded it."
    assert 0 < len(ctx) <= settings.basic_retrieval_k
    msgs = llm.calls[-1][0]
    assert msgs[0]["role"] == "system" and "Context:" in msgs[0]["content"]
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant", "user"]


def test_pro_helpers():
    a, b, c = (Result(page_content=t, metadata={}) for t in "abc")
    assert [r.page_content for r in merge_chunks([a, b], [b, c])] == ["a", "b", "c"]
    # invalid ids dropped, duplicates ignored, omitted chunks appended
    assert [r.page_content for r in apply_rank_order([a, b, c], [3, 3, 9, 1])] == ["c", "a", "b"]


def test_chunk_as_result():
    d = Document(source="company/x.md", type="company", text="t")
    r = Chunk(headline="H", summary="S", original_text="O").as_result(d)
    assert r.page_content == "H\n\nS\n\nO" and r.metadata == {"source": "company/x.md", "type": "company"}


def test_pro_end_to_end(settings):
    llm = FakeLLM("pro answer")
    p = get_pipeline("pro", settings, llm=llm, embedder=FakeEmbedder())
    assert p.ingest(load_documents(settings.knowledge_base_dir)) == 4  # 2 docs x 2 fake chunks
    answer, ctx = p.answer("Who founded Insurellm?")
    assert answer == "pro answer"
    assert 0 < len(ctx) <= settings.pro_final_k
    models = [rm for _, rm in llm.calls]
    assert any(m is not None and m.__name__ == "RankOrder" for m in models)
    assert "Extract from" in llm.calls[-1][0][0]["content"]
    assert p.rewrite_query("q") == "rewritten query"


def test_empty_store_raises(settings):
    p = get_pipeline("basic", settings, llm=FakeLLM(), embedder=FakeEmbedder())
    try:
        p.fetch_context("anything")
        assert False
    except RuntimeError as e:
        assert "ingest" in str(e)
