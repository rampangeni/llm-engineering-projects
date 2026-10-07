"""The residual MLP: 5000 -> 4096 -> 8 residual blocks -> 1 (≈289M parameters by default)."""

from __future__ import annotations

import torch.nn as nn

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
        out = self.block(x)
        out += x  # skip connection
        return self.relu(out)


class DeepNeuralNetwork(nn.Module):
    """``num_layers`` counts input layer + residual blocks + output layer, as in the reference architecture."""

    def __init__(self, input_size: int = N_FEATURES, num_layers: int = 10, hidden_size: int = 4096, dropout_prob: float = 0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_size, dropout_prob) for _ in range(num_layers - 2)]
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.residual_blocks:
            x = block(x)
        return self.output_layer(x)


def pick_device(requested: str = "auto"):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
