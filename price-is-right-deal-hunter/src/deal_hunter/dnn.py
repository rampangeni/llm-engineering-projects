"""Deep residual MLP over hashed bag-of-words features (weights come from
../price-dataset-and-baselines as ``deep_neural_network.pth``). Needs the ``nn`` extra (torch, scikit-learn)."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import HashingVectorizer

# Target normalisation used at training time: y = (log(price + 1) - mean) / std
Y_STD = 1.0328539609909058
Y_MEAN = 4.434937953948975
N_FEATURES = 5000


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout_prob: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, num_layers: int = 10, hidden_size: int = 4096, dropout_prob: float = 0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), nn.Dropout(dropout_prob)
        )
        self.residual_blocks = nn.ModuleList(ResidualBlock(hidden_size, dropout_prob) for _ in range(num_layers - 2))
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.residual_blocks:
            x = block(x)
        return self.output_layer(x)


class DeepNeuralNetworkInference:
    def __init__(self, num_layers: int = 10, hidden_size: int = 4096):
        self.vectorizer = None
        self.model = None
        self.device = None
        self._arch = dict(num_layers=num_layers, hidden_size=hidden_size)
        np.random.seed(42)
        torch.manual_seed(42)

    def setup(self) -> None:
        self.vectorizer = HashingVectorizer(n_features=N_FEATURES, stop_words="english", binary=True)
        self.model = DeepNeuralNetwork(N_FEATURES, **self._arch)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logging.info(f"Neural Network is using {self.device}")
        self.model.to(self.device)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)

    def inference(self, text: str) -> float:
        self.model.eval()
        with torch.no_grad():
            vector = self.vectorizer.transform([text])
            vector = torch.FloatTensor(vector.toarray()).to(self.device)
            pred = self.model(vector)[0]
            result = torch.exp(pred * Y_STD + Y_MEAN) - 1
        return max(0.0, result.item())
