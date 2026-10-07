"""Evaluate a price predictor on test items: avg error, MSE, RMSLE, r2 and hit rate.

Colour rule: an estimate is
* green  if error < $40  or  error/truth < 20%
* orange if error < $80  or  error/truth < 40%
* red    otherwise
"Hit rate" = fraction of green estimates.

Metrics are pure Python (no sklearn/pandas) so they can be unit-tested anywhere.
Plotly charts are optional (``pip install .[plots]``) and written to HTML files instead
of ``fig.show()`` so they work headless.
"""

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from itertools import accumulate
from typing import Callable

from tqdm import tqdm

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red": RED, "orange": YELLOW, "green": GREEN}

WORKERS = 5
DEFAULT_SIZE = 200

_NUMBER = re.compile(r"[-+]?\d*\.\d+|\d+")


def post_process(value) -> float:
    """Extract a number from an LLM reply like '$1,299.00 approx' -> 1299.0 (0 if none)."""
    if isinstance(value, str):
        value = value.replace("$", "").replace(",", "")
        match = _NUMBER.search(value)
        return float(match.group()) if match else 0
    return value


def color_for(error: float, truth: float) -> str:
    if error < 40 or error / truth < 0.2:
        return "green"
    elif error < 80 or error / truth < 0.4:
        return "orange"
    return "red"


def rmsle(truths, guesses) -> float:
    """Root mean squared log error; negative guesses are clipped to 0."""
    n = len(truths)
    s = sum((math.log1p(max(g, 0)) - math.log1p(max(t, 0))) ** 2 for t, g in zip(truths, guesses))
    return math.sqrt(s / n)


def mean_squared_error(truths, guesses) -> float:
    return sum((g - t) ** 2 for t, g in zip(truths, guesses)) / len(truths)


def r2_score(truths, guesses) -> float:
    mean = sum(truths) / len(truths)
    ss_tot = sum((t - mean) ** 2 for t in truths)
    ss_res = sum((t - g) ** 2 for t, g in zip(truths, guesses))
    return 1 - ss_res / ss_tot if ss_tot else 0.0


@dataclass
class Metrics:
    n: int
    average_error: float
    mse: float
    rmsle: float
    r2_percent: float
    hit_rate: float  # fraction green, 0..1
    green: int
    orange: int
    red: int

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_line(self, title: str = "") -> str:
        return (
            f"{title} n={self.n}  avg error ${self.average_error:,.2f}  MSE {self.mse:,.0f}  "
            f"RMSLE {self.rmsle:.3f}  r2 {self.r2_percent:.1f}%  hit rate {self.hit_rate:.1%} "
            f"(green {self.green} / orange {self.orange} / red {self.red})"
        ).strip()


def compute_metrics(truths, guesses) -> Metrics:
    errors = [abs(g - t) for g, t in zip(guesses, truths)]
    colors = [color_for(e, t) for e, t in zip(errors, truths)]
    n = len(errors)
    return Metrics(
        n=n,
        average_error=sum(errors) / n,
        mse=mean_squared_error(truths, guesses),
        rmsle=rmsle(truths, guesses),
        r2_percent=r2_score(truths, guesses) * 100,
        hit_rate=colors.count("green") / n,
        green=colors.count("green"),
        orange=colors.count("orange"),
        red=colors.count("red"),
    )


def running_mean_ci(errors: list[float]):
    """Cumulative mean and 95% CI half-width after each datapoint (for the trend chart)."""
    x = list(range(1, len(errors) + 1))
    means = [s / i for s, i in zip(accumulate(errors), x)]
    squares = list(accumulate(e * e for e in errors))
    stds = [math.sqrt(max(sq / i - m**2, 0)) if i > 1 else 0 for i, sq, m in zip(x, squares, means)]
    ci = [1.96 * (sd / math.sqrt(i)) if i > 1 else 0 for i, sd in zip(x, stds)]
    return means, ci


@dataclass
class Tester:
    """Runs ``predictor(datapoint)`` over the first ``size`` datapoints (threaded)."""

    __test__ = False  # not a pytest test class

    predictor: Callable
    data: list
    title: str | None = None
    size: int = DEFAULT_SIZE
    workers: int = WORKERS
    truth_fn: Callable = lambda d: d.price
    title_fn: Callable = lambda d: d.title
    quiet: bool = False
    titles: list = field(default_factory=list)
    guesses: list = field(default_factory=list)
    truths: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    colors: list = field(default_factory=list)

    def __post_init__(self):
        self.title = self.title or self.make_title(self.predictor)
        self.size = min(self.size, len(self.data))

    @staticmethod
    def make_title(predictor) -> str:
        name = getattr(predictor, "__name__", "predictor")
        return name.replace("__", ".").replace("_", " ").title().replace("Gpt", "GPT")

    def run_datapoint(self, i: int):
        datapoint = self.data[i]
        guess = post_process(self.predictor(datapoint))
        truth = self.truth_fn(datapoint)
        error = abs(guess - truth)
        title = self.title_fn(datapoint)
        title = title if len(title) <= 40 else title[:40] + "..."
        return title, guess, truth, error, color_for(error, truth)

    def run(self) -> Metrics:
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for title, guess, truth, error, color in tqdm(
                ex.map(self.run_datapoint, range(self.size)), total=self.size, disable=self.quiet
            ):
                self.titles.append(title)
                self.guesses.append(guess)
                self.truths.append(truth)
                self.errors.append(error)
                self.colors.append(color)
                if not self.quiet:
                    print(f"{COLOR_MAP[color]}${error:.0f}{RESET} ", end="")
        if not self.quiet:
            print()
        self.metrics = compute_metrics(self.truths, self.guesses)
        return self.metrics

    def plot(self, path_prefix: str) -> list[str]:
        """Write <prefix>_scatter.html and <prefix>_trend.html (needs plotly + pandas)."""
        from price_dataset.plots import error_trend_chart, scatter_chart

        return [
            scatter_chart(self, f"{path_prefix}_scatter.html"),
            error_trend_chart(self, f"{path_prefix}_trend.html"),
        ]


def evaluate(function, data, size=DEFAULT_SIZE, workers=WORKERS, title=None, quiet=False, **kw) -> Metrics:
    """Convenience wrapper: run a Tester and print the summary line."""
    tester = Tester(function, data, title=title, size=size, workers=workers, quiet=quiet, **kw)
    metrics = tester.run()
    if not quiet:
        print(metrics.summary_line(tester.title))
    return metrics
