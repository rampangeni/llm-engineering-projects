"""NeuralNetworkAgent: cheap local estimate from the week-6 deep neural network weights."""

from __future__ import annotations

import os

from ..config import Settings
from .base import Agent


class NeuralNetworkAgent(Agent):
    name = "Neural Network Agent"
    color = Agent.MAGENTA

    def __init__(self, settings: Settings | None = None, network=None):
        """:param network: object with ``inference(text) -> float`` (injectable for tests)"""
        s = settings or Settings()
        self.log("Neural Network Agent is initializing")
        if network is None:
            if not os.path.exists(s.nn_weights_path):
                raise FileNotFoundError(
                    f"Neural network weights not found at {s.nn_weights_path!r}. Produce deep_neural_network.pth with "
                    "../price-dataset-and-baselines (or set NN_WEIGHTS_PATH), or run with ENABLE_NN=0."
                )
            from ..dnn import DeepNeuralNetworkInference  # needs the `nn` extra

            network = DeepNeuralNetworkInference()
            network.setup()
            network.load(s.nn_weights_path)
        self.neural_network = network
        self.log("Neural Network Agent is ready and weights are loaded")

    def price(self, description: str) -> float:
        self.log("Neural Network Agent is starting a prediction")
        result = self.neural_network.inference(description)
        self.log(f"Neural Network Agent completed - predicting ${result:.2f}")
        return result
