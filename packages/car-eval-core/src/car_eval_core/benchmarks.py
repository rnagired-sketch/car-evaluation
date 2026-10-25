"""Reference models (majority class, the rev0 recipe, k-NN fallback) and the adequacy check."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from car_eval_core.dataset import DataSplits, Partition
from car_eval_core.domain import Seed
from car_eval_core.features import FeatureSet
from car_eval_core.metrics import CvSettings, Scores
from car_eval_core.tree import DevelopedTree

REV0_GRID: Final[dict[str, list[object]]] = {
    "max_depth": [4, 6, 8, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"],
}

KNN_GRID: Final[dict[str, list[object]]] = {
    "knn__n_neighbors": [1, 3, 5, 7, 9, 11, 15],
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2],  # Manhattan, Euclidean
}


def fit_majority_baseline(train: Partition) -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent").fit(train.X, train.y)


def fit_rev0_tree(train: Partition, seed: Seed) -> DecisionTreeClassifier:
    """The rev0 recipe: raw attributes, accuracy-scored 5-fold grid search, no pruning."""
    search = GridSearchCV(DecisionTreeClassifier(random_state=seed), REV0_GRID, cv=5, scoring="accuracy")
    return search.fit(train.X, train.y).best_estimator_


def fit_knn(features: FeatureSet, train: Partition, cv: CvSettings, seed: Seed) -> Pipeline:
    """k-NN on min-max scaled ranks (Lecture 3B: k-NN needs features on a common scale)."""
    pipeline = Pipeline([("scale", MinMaxScaler()), ("knn", KNeighborsClassifier())])
    search = GridSearchCV(pipeline, KNN_GRID, cv=cv.splitter(seed), scoring="f1_macro")
    return search.fit(features.transform(train.X), train.y).best_estimator_


SplitName = Literal["validation", "test"]


@dataclass(frozen=True)
class ComparisonRow:
    model: str
    split: SplitName
    scores: Scores


def tree_label(features: FeatureSet) -> str:
    return f"rev1 tree ({features.value})"


def compare_models(
    candidates: Mapping[FeatureSet, DevelopedTree], splits: DataSplits, cv: CvSettings, seed: Seed
) -> tuple[ComparisonRow, ...]:
    """Every model is fitted on the training split only, then scored on validation and test."""
    if not candidates:
        raise ValueError("compare_models needs at least one developed tree")
    best_features = max(candidates.values(), key=lambda d: d.validation.macro_f1).spec.features
    baseline = fit_majority_baseline(splits.train)
    rev0 = fit_rev0_tree(splits.train, seed)
    knn = fit_knn(best_features, splits.train, cv, seed)

    predictors: dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
        "majority baseline": lambda X: np.asarray(baseline.predict(X)),
        "rev0 tree": lambda X: np.asarray(rev0.predict(X)),
        f"k-NN ({best_features.value})": lambda X: np.asarray(knn.predict(best_features.transform(X))),
    }
    for features, developed in candidates.items():
        tree = developed.spec.fit(splits.train, seed)
        predictors[tree_label(features)] = lambda X, s=developed.spec, t=tree: s.predict(t, X)

    scored: tuple[tuple[SplitName, Partition], ...] = (
        ("validation", splits.validation),
        ("test", splits.test),
    )
    return tuple(
        ComparisonRow(name, split_name, Scores.of(part.y, predict(part.X)))
        for name, predict in predictors.items()
        for split_name, part in scored
    )


@dataclass(frozen=True)
class AdequacyTargets:
    """The proposal fixes no numbers; these defaults are the group's working assumption."""

    min_macro_f1: float = 0.90
    max_false_acceptance: float = 0.05

    def __post_init__(self) -> None:
        for name, value in dataclasses.asdict(self).items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}")


@dataclass(frozen=True)
class AdequacyCheck:
    """Proposal contingency: fall back to k-NN only if the tree is inadequate on validation."""

    validation: Scores
    targets: AdequacyTargets

    @property
    def passes(self) -> bool:
        return (
            self.validation.macro_f1 >= self.targets.min_macro_f1
            and self.validation.false_acceptance <= self.targets.max_false_acceptance
        )
