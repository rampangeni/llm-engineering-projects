"""Baseline price predictors (including the human baseline).

Every ``*_pricer`` factory takes training data and returns a ``predictor(item) -> float``
that :func:`price_dataset.evaluator.evaluate` can call. Heavy imports (sklearn, xgboost,
pandas) are lazy so the pure-logic parts import with no extras installed.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from price_dataset.items import Item

# ---------------------------------------------------------------- trivial ---------


def make_random_pricer(seed: int = 42):
    rng = random.Random(seed)

    def random_pricer(item: Item) -> float:
        return rng.randrange(1, 1000)

    return random_pricer


def make_constant_pricer(train: list[Item]):
    average = sum(i.price for i in train) / len(train)

    def constant_pricer(item: Item) -> float:
        return average

    constant_pricer.average = average
    return constant_pricer


# ----------------------------------------------------- linear on features ---------

FEATURE_COLUMNS = ["weight", "weight_unknown", "text_length"]


def get_features(item: Item) -> dict:
    """weight (lb), weight_unknown flag (weight == 0) and summary length in characters."""
    weight = item.weight or 0
    return {
        "weight": weight,
        "weight_unknown": 1 if weight == 0 else 0,
        "text_length": len(item.summary or ""),
    }


def features_matrix(items: list[Item]) -> list[list[float]]:
    return [[get_features(i)[c] for c in FEATURE_COLUMNS] for i in items]


def make_linear_regression_pricer(train: list[Item]):
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(features_matrix(train), [i.price for i in train])

    def linear_regression_pricer(item: Item) -> float:
        return float(model.predict(features_matrix([item]))[0])

    linear_regression_pricer.model = model
    return linear_regression_pricer


# --------------------------------------------------------- bag of words ----------


def make_bow_vectorizer(train: list[Item], max_features: int = 2000):
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
    X = vectorizer.fit_transform([i.summary for i in train])
    return vectorizer, X


def make_bow_linear_pricer(vectorizer, X, train: list[Item]):
    from sklearn.linear_model import LinearRegression

    regressor = LinearRegression()
    regressor.fit(X, [float(i.price) for i in train])

    def natural_language_linear_regression_pricer(item: Item) -> float:
        return max(float(regressor.predict(vectorizer.transform([item.summary]))[0]), 0)

    return natural_language_linear_regression_pricer


def make_random_forest_pricer(vectorizer, X, train: list[Item], subset: int = 15_000, n_jobs: int = 4):
    """Random forest (100 trees) on the bag-of-words matrix - first ``subset`` rows only (slow)."""
    from sklearn.ensemble import RandomForestRegressor

    prices = [float(i.price) for i in train]
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=n_jobs)
    rf.fit(X[:subset], prices[:subset])

    def random_forest(item: Item) -> float:
        return max(0, float(rf.predict(vectorizer.transform([item.summary]))[0]))

    random_forest.model = rf
    return random_forest


def make_xgboost_pricer(vectorizer, X, train: list[Item], n_estimators: int = 1000, n_jobs: int = 4):
    import xgboost as xgb

    model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=42, n_jobs=n_jobs, learning_rate=0.1)
    model.fit(X, [float(i.price) for i in train])

    def xg_boost(item: Item) -> float:
        return max(0, float(model.predict(vectorizer.transform([item.summary]))[0]))

    xg_boost.model = model
    return xg_boost


# ------------------------------------------------------------- human -------------


def export_human_csv(items: list[Item], path: str | Path, n: int = 100) -> None:
    """Write ``summary,0`` rows; a human fills in column 2 and saves it as human_out.csv."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for item in items[:n]:
            writer.writerow([item.summary, 0])


def read_human_csv(path: str | Path) -> list[float]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [float(row[1]) for row in csv.reader(f) if row]


def make_human_pricer(test: list[Item], predictions: list[float]):
    index = {id(item): i for i, item in enumerate(test[: len(predictions)])}

    def human_pricer(item: Item) -> float:
        return predictions[index[id(item)]]

    return human_pricer
