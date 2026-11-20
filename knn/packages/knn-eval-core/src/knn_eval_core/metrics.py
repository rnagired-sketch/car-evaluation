"""Scores that matter for the proposal (the same definitions as the decision tree) and the CV scheme."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from knn_eval_core.domain import CLASS_CODES, Acceptability, Seed


@dataclass(frozen=True)
class Scores:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    false_acceptance: float  # share of truly 'unacc' cars rated acc/good/vgood
    false_rejection: float  # share of truly acceptable cars rated 'unacc'
    severe_errors: int  # predictions off by two or more ordinal classes

    @staticmethod
    def of(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> Scores:
        truth, pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
        if truth.shape != pred.shape:
            raise ValueError(f"shape mismatch: {truth.shape} vs {pred.shape}")
        if truth.size == 0:
            raise ValueError("cannot score an empty prediction")
        truly_unacc = truth == Acceptability.UNACC
        return Scores(
            accuracy=float(accuracy_score(truth, pred)),
            balanced_accuracy=_balanced_accuracy(truth, pred),
            macro_f1=float(
                # sklearn is unannotated; pyright infers `str` from the "warn" default.
                f1_score(truth, pred, average="macro", labels=list(CLASS_CODES), zero_division=0)  # pyright: ignore[reportArgumentType]
            ),
            false_acceptance=_share(pred[truly_unacc] != Acceptability.UNACC),
            false_rejection=_share(pred[~truly_unacc] == Acceptability.UNACC),
            severe_errors=int((np.abs(truth - pred) >= 2).sum()),
        )


def _balanced_accuracy(truth: np.ndarray, pred: np.ndarray) -> float:
    # Small subsets (one leaf, one fold) often miss classes; sklearn warns about that but the
    # mean recall over the classes that are present is still the number we want.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return float(balanced_accuracy_score(truth, pred))


def _share(flags: np.ndarray) -> float:
    return float(flags.mean()) if flags.size else float("nan")


@dataclass(frozen=True)
class CvSettings:
    n_splits: int = 5
    n_repeats: int = 3

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.n_repeats < 1:
            raise ValueError("n_repeats must be at least 1")

    def splitter(self, seed: Seed) -> RepeatedStratifiedKFold:
        # Repeating the k-fold split averages out the luck of one fold assignment.
        return RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=seed)
