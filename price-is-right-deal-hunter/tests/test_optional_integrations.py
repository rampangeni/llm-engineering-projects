"""Run only when the optional libraries are installed (chromadb / gradio+plotly); still fully offline."""

import pytest

from conftest import FakeEncoder
from deal_hunter.config import Settings
from deal_hunter.vectorstore import CATEGORIES, build_vectorstore, open_collection, reset_collection


def test_real_chroma_build_and_query(tmp_path):
    pytest.importorskip("chromadb")
    path = str(tmp_path / "db")
    coll = reset_collection(path)
    rows = [{"summary": f"item number {i}", "category": CATEGORIES[i % 8], "price": float(i + 1)} for i in range(30)]
    assert build_vectorstore(rows, coll, FakeEncoder(), batch_size=7) == 30
    assert open_collection(path).count() == 30  # persisted
    res = coll.query(query_embeddings=FakeEncoder().encode(["item number 3"]).tolist(), n_results=5)
    assert len(res["documents"][0]) == 5 and "price" in res["metadatas"][0][0]
    assert reset_collection(path).count() == 0  # --rebuild semantics


def test_gradio_app_builds_in_demo_mode(tmp_path):
    pytest.importorskip("gradio")
    pytest.importorskip("plotly")
    from deal_hunter.ui import App

    ui = App(Settings(memory_path=str(tmp_path / "m.json")), demo=True).build()
    assert ui is not None
