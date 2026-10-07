"""Environment-driven configuration. No heavy imports here."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_FEEDS = (
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
)
# Other feeds to try: Automotive c238, Home-Garden c196

DEFAULT_WEIGHTS = {"frontier": 0.8, "specialist": 0.1, "nn": 0.1}


def _flag(env, name: str, default: bool = True) -> bool:
    v = env.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def parse_weights(text: str) -> dict[str, float]:
    """'frontier:0.8,specialist:0.1,nn:0.1' -> dict. Raises ValueError on bad input."""
    weights: dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, value = part.partition(":")
        name = name.strip()
        if name not in DEFAULT_WEIGHTS:
            raise ValueError(f"Unknown ensemble component {name!r}; expected {sorted(DEFAULT_WEIGHTS)}")
        weights[name] = float(value)
    if not weights or any(w < 0 for w in weights.values()):
        raise ValueError("ENSEMBLE_WEIGHTS must be non-empty with non-negative values")
    return weights


@dataclass(frozen=True)
class Settings:
    # models
    scanner_model: str = "gpt-5-mini"
    frontier_model: str = "gpt-5.1"
    frontier_reasoning_effort: str = "none"
    planner_model: str = "gpt-5.1"
    messaging_model: str = "claude-sonnet-4-5"
    preprocessor_model: str = "ollama/llama3.2"
    # modal specialist
    specialist_backend: str = "modal"  # modal | mock
    modal_app_name: str = "pricer-service"
    modal_class_name: str = "Pricer"
    # behaviour
    planner: str = "workflow"  # workflow | autonomous
    deal_threshold: float = 50.0
    ensemble_weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    ensemble_strategy: str = "weighted"  # weighted | min
    enable_specialist: bool = True
    enable_frontier: bool = True
    enable_nn: bool = True
    enable_preprocessor: bool = True
    feeds: tuple = DEFAULT_FEEDS
    entries_per_feed: int = 10
    max_tool_steps: int = 10
    # pushover
    pushover_user: str = ""
    pushover_token: str = ""
    # paths
    chroma_path: str = "products_vectorstore"
    memory_path: str = "memory.json"
    nn_weights_path: str = "deep_neural_network.pth"
    hf_items_dataset: str = "rampangeni/items_full"
    # ui
    ui_refresh_seconds: int = 300
    ui_plot_points: int = 800

    @classmethod
    def from_env(cls, env=None) -> "Settings":
        e = os.environ if env is None else env
        d = cls()
        feeds = tuple(f.strip() for f in e.get("DEAL_FEEDS", "").split(",") if f.strip()) or d.feeds
        weights = parse_weights(e["ENSEMBLE_WEIGHTS"]) if e.get("ENSEMBLE_WEIGHTS") else dict(d.ensemble_weights)
        return cls(
            scanner_model=e.get("SCANNER_MODEL", d.scanner_model),
            frontier_model=e.get("FRONTIER_MODEL", d.frontier_model),
            frontier_reasoning_effort=e.get("FRONTIER_REASONING_EFFORT", d.frontier_reasoning_effort),
            planner_model=e.get("PLANNER_MODEL", d.planner_model),
            messaging_model=e.get("MESSAGING_MODEL", d.messaging_model),
            preprocessor_model=e.get("PRICER_PREPROCESSOR_MODEL", d.preprocessor_model),
            specialist_backend=e.get("SPECIALIST_BACKEND", d.specialist_backend),
            modal_app_name=e.get("MODAL_APP_NAME", d.modal_app_name),
            modal_class_name=e.get("MODAL_CLASS_NAME", d.modal_class_name),
            planner=e.get("PLANNER", d.planner),
            deal_threshold=float(e.get("DEAL_THRESHOLD", d.deal_threshold)),
            ensemble_weights=weights,
            ensemble_strategy=e.get("ENSEMBLE_STRATEGY", d.ensemble_strategy),
            enable_specialist=_flag(e, "ENABLE_SPECIALIST"),
            enable_frontier=_flag(e, "ENABLE_FRONTIER"),
            enable_nn=_flag(e, "ENABLE_NN"),
            enable_preprocessor=_flag(e, "ENABLE_PREPROCESSOR"),
            feeds=feeds,
            pushover_user=e.get("PUSHOVER_USER", ""),
            pushover_token=e.get("PUSHOVER_TOKEN", ""),
            chroma_path=e.get("CHROMA_PATH", d.chroma_path),
            memory_path=e.get("MEMORY_PATH", d.memory_path),
            nn_weights_path=e.get("NN_WEIGHTS_PATH", d.nn_weights_path),
            hf_items_dataset=e.get("HF_ITEMS_DATASET", d.hf_items_dataset),
            ui_refresh_seconds=int(e.get("UI_REFRESH_SECONDS", d.ui_refresh_seconds)),
            ui_plot_points=int(e.get("UI_PLOT_POINTS", d.ui_plot_points)),
        )
