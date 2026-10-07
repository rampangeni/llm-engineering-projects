"""Tiny end-to-end DNN run on CPU; skipped when torch/sklearn are not installed."""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from conftest import make_item  # noqa: E402


def test_train_save_load_roundtrip(tmp_path):
    from price_dataset.dnn.runner import DeepNeuralNetworkPredictor, DeepNeuralNetworkRunner

    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    train = [make_item(i, price=5.0 + 15 * (i % 6), summary=f"{words[i % 6]} gadget {words[(i * 5) % 6]}") for i in range(120)]
    val = train[:20]
    runner = DeepNeuralNetworkRunner(train, val, num_layers=3, hidden_size=16, batch_size=16, device="cpu")
    runner.setup()
    hist = runner.train(epochs=2)
    assert len(hist) == 2 and hist[0]["lr"] == pytest.approx(1e-3)
    weights, meta = runner.save(tmp_path / "dnn.pth", extra_meta={"dataset": "unit"})
    assert weights.exists() and meta.name == "dnn.meta.json"
    m = json.loads(meta.read_text())
    assert m["hidden_size"] == 16 and m["y_std"] > 0 and m["dataset"] == "unit"

    predictor = DeepNeuralNetworkPredictor(weights, device="cpu")
    a = runner.inference(train[0])
    b = predictor(train[0].summary)
    assert a == pytest.approx(b, rel=1e-4) and a >= 0
