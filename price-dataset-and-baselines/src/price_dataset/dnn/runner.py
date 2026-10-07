"""Training + inference wrappers for the deep neural network.

Produces (see README / docs):
  <out>/deep_neural_network.pth        torch ``state_dict`` of ``DeepNeuralNetwork``
  <out>/deep_neural_network.meta.json  y_mean/y_std + architecture + features (sidecar)

Memory note: the 800k x 5000 bag-of-words matrix is
kept SPARSE and densified one 64-row batch at a time (densifying everything would need
~16 GB of float32). Hyper-parameters, loss, optimiser, scheduler and seeds are fixed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import HashingVectorizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from price_dataset.dnn.model import N_FEATURES, DeepNeuralNetwork, pick_device
from price_dataset.dnn.targets import DEFAULT_Y_MEAN, DEFAULT_Y_STD, TargetScaler

WEIGHTS_NAME = "deep_neural_network.pth"
META_NAME = "deep_neural_network.meta.json"


def make_vectorizer() -> HashingVectorizer:
    """Stateless hashing bag-of-words; identical at train and inference time."""
    return HashingVectorizer(n_features=N_FEATURES, stop_words="english", binary=True)


def meta_path_for(weights_path: str | Path) -> Path:
    p = Path(weights_path)
    return p.with_name(p.stem + ".meta.json")


class DeepNeuralNetworkRunner:
    def __init__(
        self,
        train,
        val,
        *,
        num_layers: int = 10,
        hidden_size: int = 4096,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        batch_size: int = 64,
        t_max: int = 10,
        grad_clip: float = 1.0,
        device: str = "auto",
        seed: int = 42,
    ):
        self.train_data, self.val_data = train, val
        self.num_layers, self.hidden_size, self.dropout = num_layers, hidden_size, dropout
        self.lr, self.weight_decay, self.batch_size = lr, weight_decay, batch_size
        self.t_max, self.grad_clip, self.seed = t_max, grad_clip, seed
        self.device_name = device
        self.vectorizer = None
        self.model = None
        self.scaler: TargetScaler | None = None
        self.history: list[dict] = []
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # -------------------------------------------------------------- setup
    def setup(self) -> None:
        self.vectorizer = make_vectorizer()
        self.X_train = self.vectorizer.fit_transform([i.summary for i in self.train_data]).tocsr()
        self.y_train = np.array([float(i.price) for i in self.train_data], dtype=np.float32)
        self.X_val = self.vectorizer.transform([i.summary for i in self.val_data]).tocsr()
        self.y_val = np.array([float(i.price) for i in self.val_data], dtype=np.float32)

        self.scaler = TargetScaler.fit(self.y_train.tolist())
        log_train = np.log(self.y_train + 1)
        self.y_train_norm = ((log_train - self.scaler.y_mean) / self.scaler.y_std).astype(np.float32)
        self.y_val_norm = ((np.log(self.y_val + 1) - self.scaler.y_mean) / self.scaler.y_std).astype(np.float32)

        self.build_model()
        self.loss_function = nn.L1Loss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.t_max, eta_min=0)

    def build_model(self) -> None:
        self.device = pick_device(self.device_name)
        self.model = DeepNeuralNetwork(N_FEATURES, self.num_layers, self.hidden_size, self.dropout)
        total = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Deep Neural Network created with {total:,} parameters")
        print(f"Using {self.device}")
        self.model.to(self.device)

    def _dense(self, X) -> torch.Tensor:
        return torch.from_numpy(X.toarray().astype(np.float32, copy=False)).to(self.device)

    # -------------------------------------------------------------- train
    def train(self, epochs: int = 5) -> list[dict]:
        n = self.X_train.shape[0]
        for epoch in range(1, epochs + 1):
            self.model.train()
            order = np.random.permutation(n)
            losses = []
            for start in tqdm(range(0, n, self.batch_size), desc=f"epoch {epoch}/{epochs}"):
                idx = order[start : start + self.batch_size]
                batch_X = self._dense(self.X_train[idx])
                batch_y = torch.from_numpy(self.y_train_norm[idx]).unsqueeze(1).to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss_function(self.model(batch_X), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()
                losses.append(loss.item())

            val_loss, mae = self.validate()
            lr_now = self.scheduler.get_last_lr()[0]
            rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "val_mae": mae, "lr": lr_now}
            self.history.append(rec)
            print(f"Epoch [{epoch}/{epochs}]")
            print(f"Train Loss: {rec['train_loss']:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Val mean absolute error: ${mae:.2f}")
            print(f"Learning rate: {lr_now:.6f}")
            self.scheduler.step()
        return self.history

    def validate(self) -> tuple[float, float]:
        """-> (L1 loss in normalised log space, mean absolute error in dollars)."""
        self.model.eval()
        with torch.no_grad():
            out = self.model(self._dense(self.X_val))
            y_norm = torch.from_numpy(self.y_val_norm).unsqueeze(1).to(self.device)
            val_loss = self.loss_function(out, y_norm).item()
            dollars = torch.exp(out * self.scaler.y_std + self.scaler.y_mean) - 1
            mae = torch.abs(dollars - torch.from_numpy(self.y_val).unsqueeze(1).to(self.device)).mean().item()
        return val_loss, mae

    # ------------------------------------------------------- save / load / infer
    def save(self, path: str | Path, extra_meta: dict | None = None) -> tuple[Path, Path]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        meta = {
            "format": "torch state_dict of price_dataset.dnn.model.DeepNeuralNetwork",
            "n_features": N_FEATURES,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "vectorizer": {"type": "HashingVectorizer", "n_features": N_FEATURES, "stop_words": "english", "binary": True},
            **self.scaler.to_dict(),
            "train_size": len(self.train_data),
            "history": self.history,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **(extra_meta or {}),
        }
        mp = meta_path_for(path)
        mp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path, mp

    def inference(self, item_or_text) -> float:
        text = getattr(item_or_text, "summary", item_or_text)
        return _infer(self.model, self.vectorizer, self.scaler, self.device, text)


def _infer(model, vectorizer, scaler: TargetScaler, device, text: str) -> float:
    model.eval()
    with torch.no_grad():
        vec = torch.from_numpy(vectorizer.transform([text]).toarray().astype(np.float32)).to(device)
        pred = model(vec)[0].item()
    return scaler.decode(pred)


class DeepNeuralNetworkPredictor:
    """Inference-only: load weights (+ sidecar meta) - no training data required.

    If the sidecar is missing (e.g. weights downloaded without one) it
    falls back to the default constants for the full dataset (y_mean=4.4349, y_std=1.0329).
    """

    def __init__(self, weights_path: str | Path, meta_path: str | Path | None = None, device: str = "auto"):
        meta_file = Path(meta_path) if meta_path else meta_path_for(weights_path)
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        self.scaler = TargetScaler(meta.get("y_mean", DEFAULT_Y_MEAN), meta.get("y_std", DEFAULT_Y_STD))
        self.device = pick_device(device)
        self.vectorizer = make_vectorizer()
        self.model = DeepNeuralNetwork(
            meta.get("n_features", N_FEATURES), meta.get("num_layers", 10), meta.get("hidden_size", 4096), meta.get("dropout", 0.2)
        )
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)

    def __call__(self, item_or_text) -> float:
        text = getattr(item_or_text, "summary", item_or_text)
        return _infer(self.model, self.vectorizer, self.scaler, self.device, text)
