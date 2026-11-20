"""The k-nearest-neighbor classifier and its hyperparameter search.

Lecture 3B names the three things k-NN needs (stored records, a distance metric, a K value),
the ways to turn neighbors into a class (majority vote, or votes weighted by distance, e.g.
1/d^2), the distance metrics to choose from (Euclidean, Manhattan, Minkowski, Hamming, cosine)
and the need to put features on a common scale. Each of those is a setting here, and the grid
search compares them on the training split by cross-validated macro-F1.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, assert_never

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler

from knn_eval_core.dataset import Partition
from knn_eval_core.domain import Seed
from knn_eval_core.features import InputSet
from knn_eval_core.metrics import CvSettings

logger = logging.getLogger(__name__)


class DistanceMetric(enum.Enum):
    """Distance between two encoded cars (Lecture 3B, 'Distance Metrics')."""

    EUCLIDEAN = "euclidean"  # straight-line distance; suits continuous features on one scale
    MANHATTAN = "manhattan"  # sum of absolute differences; suits grid-like (ordinal) data
    MINKOWSKI = "minkowski"  # generalises both: p = 1 Manhattan, p = 2 Euclidean
    HAMMING = "hamming"  # share of attributes that differ; the categorical distance
    COSINE = "cosine"  # 1 - cosine similarity; compares direction, not magnitude

    @property
    def needs_brute_force(self) -> bool:
        """sklearn's tree indexes do not support these metrics."""
        return self in (DistanceMetric.HAMMING, DistanceMetric.COSINE)


class Voting(enum.Enum):
    """How the K neighbors' labels become one rating (Lecture 3B, 'Nearest Neighbor Classification')."""

    MAJORITY = "majority"  # every neighbor has one vote
    INVERSE_DISTANCE = "inverse_distance"  # weight 1/d
    INVERSE_SQUARE = "inverse_square"  # weight 1/d^2, the lecture's example weight factor


class Scaling(enum.Enum):
    """Putting features on a common scale (Lecture 3B, 'Limitations': sensitive to feature scaling)."""

    STANDARD = "standard"  # zero mean, unit variance
    MINMAX = "minmax"  # each feature in [0, 1]
    NONE = "none"  # raw ranks; features with more levels weigh more


def inverse_square_weights(distances: np.ndarray) -> np.ndarray:
    """Weights 1/d^2 per neighbor. A neighbor at distance 0 is an exact match: when a row has
    one, only its exact matches vote."""
    distances = np.asarray(distances, dtype=float)
    exact = distances == 0
    with np.errstate(divide="ignore"):
        weights = 1.0 / distances**2
    rows_with_exact = exact.any(axis=1)
    weights[rows_with_exact] = exact[rows_with_exact].astype(float)
    return weights


def _identity(X: Any) -> Any:
    return X


@dataclass(frozen=True)
class KnnParams:
    k: int
    metric: DistanceMetric
    voting: Voting
    scaling: Scaling
    minkowski_p: float = 3.0  # only used by the Minkowski metric

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be at least 1")
        if self.minkowski_p < 1:
            raise ValueError("minkowski_p must be at least 1 (p < 1 is not a distance)")

    def build(self) -> Pipeline:
        """Scaler + classifier. k-NN has no random state: the result depends on the data only."""
        match self.scaling:
            case Scaling.STANDARD:
                scaler: Any = StandardScaler()
            case Scaling.MINMAX:
                scaler = MinMaxScaler()
            case Scaling.NONE:
                scaler = FunctionTransformer(_identity)
            case _:
                assert_never(self.scaling)
        match self.voting:
            case Voting.MAJORITY:
                weights: Any = "uniform"
            case Voting.INVERSE_DISTANCE:
                weights = "distance"
            case Voting.INVERSE_SQUARE:
                weights = inverse_square_weights
            case _:
                assert_never(self.voting)
        knn = KNeighborsClassifier(
            n_neighbors=self.k,
            metric=self.metric.value,
            # sklearn reads p only for the Minkowski metric; it accepts any real p >= 1.
            p=self.minkowski_p if self.metric is DistanceMetric.MINKOWSKI else 2,  # pyright: ignore[reportArgumentType]
            weights=weights,
            algorithm="brute" if self.metric.needs_brute_force else "auto",
        )
        return Pipeline([("scale", scaler), ("knn", knn)])

    def describe(self) -> str:
        metric = self.metric.value
        if self.metric is DistanceMetric.MINKOWSKI:
            metric += f" (p={self.minkowski_p:g})"
        return f"k={self.k}, {metric} distance, {self.voting.value} voting, {self.scaling.value} scaling"


@dataclass(frozen=True)
class KnnSpec:
    """Everything needed to rebuild the model: which inputs the distance uses and the parameters."""

    inputs: InputSet
    params: KnnParams

    def fit(self, data: Partition) -> Pipeline:
        """'Training' a k-NN model only scales and stores the training cars."""
        return self.params.build().fit(self.inputs.transform(data.X), data.y)

    def predict(self, model: Pipeline, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(model.predict(self.inputs.transform(X)))


@dataclass(frozen=True)
class SearchGrid:
    """Settings compared by the grid search."""

    k: tuple[int, ...] = (1, 3, 5, 7, 9, 11, 15, 21)
    metric: tuple[DistanceMetric, ...] = tuple(DistanceMetric)
    voting: tuple[Voting, ...] = tuple(Voting)
    scaling: tuple[Scaling, ...] = (Scaling.STANDARD,)
    minkowski_p: float = 3.0

    def __post_init__(self) -> None:
        for name in ("k", "metric", "voting", "scaling"):
            if not getattr(self, name):
                raise ValueError(f"search grid entry {name!r} is empty")
        if any(k < 1 for k in self.k):
            raise ValueError("k values must be >= 1")
        if self.minkowski_p < 1:
            raise ValueError("minkowski_p must be at least 1")

    @property
    def size(self) -> int:
        return len(self.k) * len(self.metric) * len(self.voting) * len(self.scaling)

    def candidates(self) -> list[KnnParams]:
        return [
            KnnParams(k, metric, voting, scaling, self.minkowski_p)
            for metric in self.metric
            for voting in self.voting
            for scaling in self.scaling
            for k in self.k
        ]


@dataclass(frozen=True)
class CandidateScore:
    params: KnnParams
    macro_f1_mean: float
    macro_f1_std: float
    balanced_accuracy_mean: float
    accuracy_mean: float


@dataclass(frozen=True)
class SearchResult:
    spec: KnnSpec
    cv_macro_f1_mean: float
    cv_macro_f1_std: float
    top_candidates: tuple[CandidateScore, ...]
    best_per_metric: Mapping[DistanceMetric, CandidateScore]  # the lecture's metric comparison
    k_scores: Mapping[int, float]  # CV macro-F1 per K with the best metric, voting and scaling


def tune_knn(
    inputs: InputSet, train: Partition, grid: SearchGrid, cv: CvSettings, seed: Seed, top: int = 5
) -> SearchResult:
    """Pick K, distance, voting and scaling by cross-validated macro-F1 on the training split.

    Every candidate is scored on the same folds, so the comparison between them is paired.
    """
    folds = cv.n_splits * cv.n_repeats
    logger.info(
        "grid search on %s: %d candidates x %d folds = %d fits",
        inputs.description,
        grid.size,
        folds,
        grid.size * folds,
    )
    X, y = inputs.transform(train.X), train.y
    splits = list(cv.splitter(seed).split(X, y))
    scoring = {"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy", "accuracy": "accuracy"}
    scores: list[CandidateScore] = []
    for params in grid.candidates():
        result = cross_validate(params.build(), X, y, cv=splits, scoring=scoring)
        scores.append(
            CandidateScore(
                params=params,
                macro_f1_mean=float(np.mean(result["test_macro_f1"])),
                macro_f1_std=float(np.std(result["test_macro_f1"])),
                balanced_accuracy_mean=float(np.mean(result["test_balanced_accuracy"])),
                accuracy_mean=float(np.mean(result["test_accuracy"])),
            )
        )
    # Best first; ties go to the smaller K (simpler, cheaper), then to grid order.
    ranked = sorted(
        range(len(scores)), key=lambda i: (-round(scores[i].macro_f1_mean, 12), scores[i].params.k, i)
    )
    best = scores[ranked[0]]
    best_per_metric: dict[DistanceMetric, CandidateScore] = {}
    for i in ranked:
        best_per_metric.setdefault(scores[i].params.metric, scores[i])
    k_scores = {
        s.params.k: s.macro_f1_mean
        for s in scores
        if dataclasses.replace(s.params, k=best.params.k) == best.params
    }
    logger.info(
        "best CV macro-F1 %.3f +/- %.3f with %s",
        best.macro_f1_mean,
        best.macro_f1_std,
        best.params.describe(),
    )
    for metric, score in best_per_metric.items():
        logger.info(
            "  %-10s best CV macro-F1 %.3f (%s)", metric.value, score.macro_f1_mean, score.params.describe()
        )
    return SearchResult(
        spec=KnnSpec(inputs, best.params),
        cv_macro_f1_mean=best.macro_f1_mean,
        cv_macro_f1_std=best.macro_f1_std,
        top_candidates=tuple(scores[i] for i in ranked[:top]),
        best_per_metric=best_per_metric,
        k_scores=dict(sorted(k_scores.items())),
    )
