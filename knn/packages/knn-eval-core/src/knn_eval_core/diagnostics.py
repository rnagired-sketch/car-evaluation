"""k-NN diagnostics as plain data: the choice of K, learning curve, confusion, prediction cost."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, learning_curve, validation_curve

from knn_eval_core.dataset import Partition
from knn_eval_core.domain import CLASS_CODES, Seed
from knn_eval_core.evaluator import KnnEvaluator
from knn_eval_core.metrics import CvSettings
from knn_eval_core.neighbors import KnnSpec

Arrays3 = tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class Curve:
    """Training and cross-validation macro-F1 (mean and std over folds) against `x`."""

    x: np.ndarray
    train_mean: np.ndarray
    train_std: np.ndarray
    cv_mean: np.ndarray
    cv_std: np.ndarray

    @staticmethod
    def from_scores(x: np.ndarray, train_scores: np.ndarray, cv_scores: np.ndarray) -> Curve:
        return Curve(
            np.asarray(x),
            train_scores.mean(axis=1),
            train_scores.std(axis=1),
            cv_scores.mean(axis=1),
            cv_scores.std(axis=1),
        )


def k_curve(
    spec: KnnSpec,
    train: Partition,
    cv: CvSettings,
    seed: Seed,
    ks: Sequence[int] = (1, 3, 5, 7, 9, 11, 15, 21, 31),
) -> Curve:
    """Macro-F1 against K with the other settings held fixed (Lecture 3B, 'Choosing the value of k':
    a small K follows noise, a large K blurs the class boundaries)."""
    x = np.asarray([k for k in ks if k <= len(train) * (cv.n_splits - 1) // cv.n_splits])
    train_scores, cv_scores = validation_curve(
        dataclasses.replace(spec.params, k=1).build(),
        spec.inputs.transform(train.X),
        train.y,
        param_name="knn__n_neighbors",
        param_range=x,
        cv=cv.splitter(seed),
        scoring="f1_macro",
    )
    return Curve.from_scores(x, train_scores, cv_scores)


def learning_curve_of(
    spec: KnnSpec, data: Partition, seed: Seed, points: int = 8, n_splits: int = 5
) -> Curve:
    """Macro-F1 against the number of stored training cars: would more data help?"""
    smallest = max(spec.params.k, 1) / (len(data) * (n_splits - 1) / n_splits)
    sizes, train_scores, cv_scores = cast(
        Arrays3,
        learning_curve(
            spec.params.build(),
            spec.inputs.transform(data.X),
            data.y,
            train_sizes=np.linspace(max(0.1, smallest), 1.0, points),
            scoring="f1_macro",
            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
        ),
    )
    return Curve.from_scores(sizes, train_scores, cv_scores)


def confusion(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Counts with rows = true class and columns = predicted class, in `Acceptability` order."""
    return confusion_matrix(y_true, y_pred, labels=list(CLASS_CODES))


@dataclass(frozen=True)
class PredictionCost:
    """Lecture 3B, 'Limitations': k-NN is computationally expensive at prediction time,
    because every prediction compares the car with all stored training cars."""

    stored_cars: int
    predicted_cars: int
    seconds_per_car: float  # wall-clock; varies between machines and runs


def prediction_cost(evaluator: KnnEvaluator, X: pd.DataFrame, repeats: int = 5) -> PredictionCost:
    if len(X) == 0:
        raise ValueError("need at least one car to time")
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        evaluator.predict(X)
        best = min(best, time.perf_counter() - started)
    return PredictionCost(evaluator.training_size, len(X), best / len(X))
