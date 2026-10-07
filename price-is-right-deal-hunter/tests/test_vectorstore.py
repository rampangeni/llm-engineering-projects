import numpy as np

from conftest import FakeCollection, FakeEncoder
from deal_hunter.vectorstore import build_vectorstore, color_for, get_plot_data


def rows(n, skip_summary_at=None):
    return ({"summary": None if i == skip_summary_at else f"summary {i}", "category": "Electronics", "price": float(i)} for i in range(n))


def test_build_batches_and_ids():
    coll = FakeCollection()
    added = build_vectorstore(rows(25), coll, FakeEncoder(), batch_size=10)
    assert added == 25 and coll.count() == 25
    assert coll.ids[0] == "doc_0" and coll.ids[-1] == "doc_24" and len(set(coll.ids)) == 25
    assert coll.metas[3] == {"category": "Electronics", "price": 3.0}
    assert len(coll.embeddings[0]) == 3


def test_build_limit_and_missing_summaries():
    coll = FakeCollection()
    assert build_vectorstore(rows(50, skip_summary_at=2), coll, FakeEncoder(), batch_size=4, limit=6) == 5
    assert "summary 2" not in coll.docs
    assert coll.ids == [f"doc_{i}" for i in range(5)]


def test_color_mapping():
    assert color_for("Appliances") == "red" and color_for("???") == "grey"


def test_plot_data_small_store_returns_empty():
    class C:
        def get(self, include, limit):
            return {"embeddings": [[0.0, 1.0]], "documents": ["a"], "metadatas": [{"category": "x"}]}

    docs, vecs, colors = get_plot_data(C())
    assert docs == [] and vecs.shape == (0, 3) and colors == []


def test_plot_data_runs_tsne():
    import pytest

    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)

    class C:
        def get(self, include, limit):
            n = 12
            return {"embeddings": rng.normal(size=(n, 6)).tolist(), "documents": [f"d{i}" for i in range(n)],
                    "metadatas": [{"category": "Electronics"}] * n}

    docs, vecs, colors = get_plot_data(C(), 12)
    assert vecs.shape == (12, 3) and colors[0] == "orange"
