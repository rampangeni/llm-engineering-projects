"""Metrics + tester for prompt/completion datapoints (no plotting).

Colour rule: green error < $40 or < 20%; orange < $80 or < 40%; red otherwise. hit rate = share green.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Callable

from tqdm import tqdm

from price_qlora.parse import extract_price


def color_for(error: float, truth: float) -> str:
    if error < 40 or error / truth < 0.2:
        return "green"
    if error < 80 or error / truth < 0.4:
        return "orange"
    return "red"


@dataclass
class Metrics:
    n: int
    average_error: float
    mse: float
    rmsle: float
    r2_percent: float
    hit_rate: float
    green: int
    orange: int
    red: int

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_line(self, title: str = "") -> str:
        return (
            f"{title} n={self.n}  avg error ${self.average_error:,.2f}  MSE {self.mse:,.0f}  RMSLE {self.rmsle:.3f}  "
            f"r2 {self.r2_percent:.1f}%  hit rate {self.hit_rate:.1%} "
            f"(green {self.green}/orange {self.orange}/red {self.red})"
        ).strip()


def compute_metrics(truths, guesses) -> Metrics:
    n = len(truths)
    errors = [abs(g - t) for g, t in zip(guesses, truths)]
    colors = [color_for(e, t) for e, t in zip(errors, truths)]
    mean = sum(truths) / n
    ss_tot = sum((t - mean) ** 2 for t in truths)
    ss_res = sum((t - g) ** 2 for t, g in zip(truths, guesses))
    rmsle = math.sqrt(sum((math.log1p(max(g, 0)) - math.log1p(max(t, 0))) ** 2 for t, g in zip(truths, guesses)) / n)
    return Metrics(
        n=n,
        average_error=sum(errors) / n,
        mse=ss_res / n,
        rmsle=rmsle,
        r2_percent=(1 - ss_res / ss_tot) * 100 if ss_tot else 0.0,
        hit_rate=colors.count("green") / n,
        green=colors.count("green"),
        orange=colors.count("orange"),
        red=colors.count("red"),
    )


def title_of(prompt: str) -> str:
    """First 'Title: ...' line of the summary inside a prompt (for readable reports)."""
    pieces = prompt.split("Title: ")
    title = pieces[1].split("\n")[0] if len(pieces) > 1 else pieces[0]
    return title if len(title) <= 40 else title[:40] + "..."


@dataclass
class Tester:
    """Runs ``predictor(prompt) -> str|float`` over datapoints ``{"prompt", "completion"}``.

    Sequential by default: one GPU model instance is not thread-safe for generate().
    """

    __test__ = False

    predictor: Callable
    data: list
    title: str = "model"
    size: int = 200
    quiet: bool = False
    titles: list = field(default_factory=list)
    guesses: list = field(default_factory=list)
    truths: list = field(default_factory=list)
    raw: list = field(default_factory=list)

    def __post_init__(self):
        self.size = min(self.size, len(self.data))

    def run(self) -> Metrics:
        for i in tqdm(range(self.size), disable=self.quiet):
            dp = self.data[i]
            raw = self.predictor(dp["prompt"])
            self.raw.append(raw)
            self.guesses.append(extract_price(raw) if isinstance(raw, str) else float(raw))
            self.truths.append(float(dp["completion"]))
            self.titles.append(title_of(dp["prompt"]))
        self.metrics = compute_metrics(self.truths, self.guesses)
        return self.metrics

    def rows(self) -> list[dict]:
        return [{"title": t, "truth": y, "guess": g, "raw": r} for t, y, g, r in zip(self.titles, self.truths, self.guesses, self.raw)]

    def dump(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"title": self.title, "metrics": self.metrics.to_dict(), "rows": self.rows()}, f, indent=2)
