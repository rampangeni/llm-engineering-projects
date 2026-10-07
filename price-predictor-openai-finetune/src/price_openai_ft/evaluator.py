"""Metrics + threaded tester (no plotting).

Colour rule: green error < $40 or < 20%; orange < $80 or < 40%; else red. hit rate = share green.
"""

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable

from tqdm import tqdm

_NUMBER = re.compile(r"[-+]?\d*\.\d+|\d+")


def post_process(value) -> float:
    if isinstance(value, str):
        value = value.replace("$", "").replace(",", "")
        m = _NUMBER.search(value)
        return float(m.group()) if m else 0
    return value


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
    return Metrics(
        n=n,
        average_error=sum(errors) / n,
        mse=ss_res / n,
        rmsle=math.sqrt(sum((math.log1p(max(g, 0)) - math.log1p(max(t, 0))) ** 2 for t, g in zip(truths, guesses)) / n),
        r2_percent=(1 - ss_res / ss_tot) * 100 if ss_tot else 0.0,
        hit_rate=colors.count("green") / n,
        green=colors.count("green"),
        orange=colors.count("orange"),
        red=colors.count("red"),
    )


@dataclass
class Tester:
    __test__ = False

    predictor: Callable
    data: list
    title: str = "model"
    size: int = 200
    workers: int = 5
    quiet: bool = False
    titles: list = field(default_factory=list)
    guesses: list = field(default_factory=list)
    truths: list = field(default_factory=list)
    raw: list = field(default_factory=list)

    def __post_init__(self):
        self.size = min(self.size, len(self.data))

    def _one(self, i):
        item = self.data[i]
        raw = self.predictor(item)
        return item.title, raw, post_process(raw), item.price

    def run(self) -> Metrics:
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for title, raw, guess, truth in tqdm(ex.map(self._one, range(self.size)), total=self.size, disable=self.quiet):
                self.titles.append(title)
                self.raw.append(raw)
                self.guesses.append(guess)
                self.truths.append(truth)
        self.metrics = compute_metrics(self.truths, self.guesses)
        return self.metrics

    def rows(self) -> list[dict]:
        """Per-datapoint results (for --save-predictions)."""
        return [
            {"title": t, "truth": y, "guess": g, "raw": r}
            for t, y, g, r in zip(self.titles, self.truths, self.guesses, self.raw)
        ]
