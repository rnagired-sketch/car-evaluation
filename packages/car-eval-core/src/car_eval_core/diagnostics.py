"""Over/underfitting diagnostics (Lecture 3A) as plain data, ready for any plotting front end."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, learning_curve, validation_curve

from car_eval_core.dataset import Partition
from car_eval_core.domain import CLASS_CODES, CcpAlpha, Seed
from car_eval_core.metrics import CvSettings
from car_eval_core.tree import TreeSpec

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


def depth_curve(
    spec: TreeSpec, train: Partition, cv: CvSettings, seed: Seed, depths: Sequence[int] = range(1, 16)
) -> Curve:
    """Macro-F1 against max_depth, with the other tuned hyperparameters held fixed."""
    unlimited = dataclasses.replace(spec.params, max_depth=None, ccp_alpha=CcpAlpha(0.0))
    x = np.asarray(depths)
    train_scores, cv_scores = validation_curve(
        unlimited.build(seed),
        spec.features.transform(train.X),
        train.y,
        param_name="max_depth",
        param_range=x,
        cv=cv.splitter(seed),
        scoring="f1_macro",
    )
    return Curve.from_scores(x, train_scores, cv_scores)


def learning_curve_of(
    spec: TreeSpec, data: Partition, seed: Seed, points: int = 8, n_splits: int = 5
) -> Curve:
    """Macro-F1 against the number of training cars: would more data help?"""
    # Three arrays come back unless `return_times=True` is requested.
    sizes, train_scores, cv_scores = cast(
        Arrays3,
        learning_curve(
            spec.params.build(seed),
            spec.features.transform(data.X),
            data.y,
            train_sizes=np.linspace(0.1, 1.0, points),
            scoring="f1_macro",
            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
        ),
    )
    return Curve.from_scores(sizes, train_scores, cv_scores)


def confusion(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Counts with rows = true class and columns = predicted class, in `Acceptability` order."""
    return confusion_matrix(y_true, y_pred, labels=list(CLASS_CODES))
