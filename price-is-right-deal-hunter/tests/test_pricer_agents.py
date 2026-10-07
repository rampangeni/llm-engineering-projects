import sys
import types

import pytest

from conftest import FakeCollection, FakeEncoder, fake_openai, text_response
from deal_hunter.agents.frontier import FrontierAgent
from deal_hunter.agents.neural_network import NeuralNetworkAgent
from deal_hunter.agents.specialist import MockPricer, SpecialistAgent
from deal_hunter.config import Settings


def test_specialist_mock_backend_is_deterministic():
    a = SpecialistAgent(Settings(specialist_backend="mock"))
    assert a.price("x") == a.price("x") == MockPricer().price("x")


def test_specialist_resolves_modal_by_name(monkeypatch):
    seen = {}

    class Inst:
        price = types.SimpleNamespace(remote=lambda text: seen.setdefault("text", text) and 321)

    fake = types.SimpleNamespace(Cls=types.SimpleNamespace(from_name=lambda app, cls: seen.update(names=(app, cls)) or (lambda: Inst())))
    monkeypatch.setitem(sys.modules, "modal", fake)
    agent = SpecialistAgent(Settings(modal_app_name="my-app", modal_class_name="P"))
    assert agent.price("hello") == 321.0
    assert seen["names"] == ("my-app", "P") and seen["text"] == "hello"


def test_frontier_rag_prompt_and_parsing(settings):
    coll = FakeCollection(["sim A", "sim B"], [10.0, 20.5])
    client = fake_openai(text_response("$1,299.99"))
    agent = FrontierAgent(coll, settings, client=client, encoder=FakeEncoder())
    assert agent.price("my product") == 1299.99
    assert coll.query_args[1] == 5
    call = client.calls["create"][0]
    content = call["messages"][0]["content"]
    assert "my product" in content and "sim A\nPrice is $10.00" in content and "Price is $20.50" in content
    assert call["model"] == settings.frontier_model and call["seed"] == 42 and call["reasoning_effort"] == "none"


def test_frontier_get_price_edge_cases():
    assert FrontierAgent.get_price("about $45 total") == 45.0
    assert FrontierAgent.get_price("no idea") == 0.0
    assert FrontierAgent.get_price(None) == 0.0


def test_nn_agent_with_injected_network(settings):
    net = types.SimpleNamespace(inference=lambda t: 12.5)
    assert NeuralNetworkAgent(settings, network=net).price("x") == 12.5


def test_nn_agent_missing_weights_message(settings):
    with pytest.raises(FileNotFoundError, match="price-dataset-and-baselines"):
        NeuralNetworkAgent(settings)


def test_dnn_forward_and_normalisation(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from deal_hunter.dnn import DeepNeuralNetwork, DeepNeuralNetworkInference

    net = DeepNeuralNetworkInference(num_layers=3, hidden_size=16)
    net.setup()
    path = tmp_path / "w.pth"
    torch.save(net.model.state_dict(), path)
    net2 = DeepNeuralNetworkInference(num_layers=3, hidden_size=16)
    net2.setup()
    net2.load(str(path))
    assert net2.inference("a fine microphone with usb") >= 0.0
    assert isinstance(DeepNeuralNetwork(10, 3, 8)(torch.zeros(1, 10)), torch.Tensor)
