"""Target (price) transform used to train / invert the DNN. Pure Python - no torch.

Training target:  y = (log(price + 1) - y_mean) / y_std      (y_mean/y_std from TRAIN set)
Prediction:       price = max(0, exp(y_hat * y_std + y_mean) - 1)

For the full 800k training set the reference values are y_mean=4.434937953948975,
y_std=1.0328539609909058 (used by downstream inference code). They are saved next to the
weights in ``deep_neural_network.meta.json`` so consumers don't need the training data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_Y_MEAN = 4.434937953948975
DEFAULT_Y_STD = 1.0328539609909058


@dataclass(frozen=True)
class TargetScaler:
    y_mean: float
    y_std: float

    @classmethod
    def fit(cls, prices) -> "TargetScaler":
        logs = [math.log(p + 1) for p in prices]
        n = len(logs)
        mean = sum(logs) / n
        # torch.Tensor.std() is the unbiased (n-1) estimator - match it
        var = sum((v - mean) ** 2 for v in logs) / (n - 1)
        return cls(mean, math.sqrt(var))

    def encode(self, price: float) -> float:
        return (math.log(price + 1) - self.y_mean) / self.y_std

    def decode(self, y: float) -> float:
        return max(0.0, math.exp(y * self.y_std + self.y_mean) - 1)

    def to_dict(self) -> dict:
        return {"y_mean": self.y_mean, "y_std": self.y_std}
