"""CART tree building (grid search) and pruning (cost-complexity, chosen on validation)."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from car_eval_core.dataset import DataSplits, Partition
from car_eval_core.domain import CcpAlpha, Seed
from car_eval_core.features import FeatureSet
from car_eval_core.metrics import CvSettings, Scores

Criterion = Literal["gini", "entropy"]
ClassWeight = Literal["balanced"] | None


@dataclass(frozen=True)
class SearchGrid:
    """Hyperparameter values tried during tree building. `None` depth means unlimited."""

    criterion: tuple[Criterion, ...] = ("gini", "entropy")
    max_depth: tuple[int | None, ...] = (6, 8, 10, 12, None)
    min_samples_split: tuple[int, ...] = (2, 5)
    min_samples_leaf: tuple[int, ...] = (1, 2, 4)
    class_weight: tuple[ClassWeight, ...] = (None, "balanced")

    def __post_init__(self) -> None:
        for name, values in dataclasses.asdict(self).items():
            if not values:
                raise ValueError(f"search grid entry {name!r} is empty")
        if any(d is not None and d < 1 for d in self.max_depth):
            raise ValueError("max_depth values must be >= 1 or None")
        if any(n < 2 for n in self.min_samples_split):
            raise ValueError("min_samples_split values must be >= 2")
        if any(n < 1 for n in self.min_samples_leaf):
            raise ValueError("min_samples_leaf values must be >= 1")

    @property
    def size(self) -> int:
        return int(np.prod([len(v) for v in dataclasses.asdict(self).values()]))

    def as_param_grid(self) -> dict[str, list[Any]]:
        return {name: list(values) for name, values in dataclasses.asdict(self).items()}


@dataclass(frozen=True)
class TreeHyperparameters:
    criterion: Criterion
    max_depth: int | None
    min_samples_split: int
    min_samples_leaf: int
    class_weight: ClassWeight
    ccp_alpha: CcpAlpha = CcpAlpha(0.0)

    def __post_init__(self) -> None:
        if self.ccp_alpha < 0:
            raise ValueError("ccp_alpha must be non-negative")

    @staticmethod
    def from_search(params: Mapping[str, Any]) -> TreeHyperparameters:
        return TreeHyperparameters(
            criterion=params["criterion"],
            max_depth=params["max_depth"],
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            class_weight=params["class_weight"],
        )

    def build(self, seed: Seed) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(**dataclasses.asdict(self), random_state=seed)


@dataclass(frozen=True)
class TreeSpec:
    """Everything needed to rebuild a tree: which inputs it sees and how it is grown."""

    features: FeatureSet
    params: TreeHyperparameters

    def with_alpha(self, alpha: CcpAlpha) -> TreeSpec:
        return dataclasses.replace(self, params=dataclasses.replace(self.params, ccp_alpha=alpha))

    def fit(self, data: Partition, seed: Seed) -> DecisionTreeClassifier:
        return self.params.build(seed).fit(self.features.transform(data.X), data.y)

    def predict(self, tree: DecisionTreeClassifier, X: pd.DataFrame) -> np.ndarray:
        return tree.predict(self.features.transform(X))


@dataclass(frozen=True)
class CandidateScore:
    params: TreeHyperparameters
    macro_f1_mean: float
    macro_f1_std: float
    balanced_accuracy_mean: float
    accuracy_mean: float


@dataclass(frozen=True)
class SearchResult:
    spec: TreeSpec
    cv_macro_f1_mean: float
    cv_macro_f1_std: float
    top_candidates: tuple[CandidateScore, ...]


def tune_tree(
    features: FeatureSet, train: Partition, grid: SearchGrid, cv: CvSettings, seed: Seed, top: int = 5
) -> SearchResult:
    """Tree-building phase: pick hyperparameters by cross-validated macro-F1 on the training split."""
    search = GridSearchCV(
        DecisionTreeClassifier(random_state=seed),
        grid.as_param_grid(),
        cv=cv.splitter(seed),
        scoring={"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy", "accuracy": "accuracy"},
        refit="macro_f1",  # pyright: ignore[reportArgumentType]  (sklearn is unannotated; a scorer name is valid)
    )
    search.fit(features.transform(train.X), train.y)
    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_macro_f1")
    candidates = tuple(
        CandidateScore(
            params=TreeHyperparameters.from_search(row["params"]),
            macro_f1_mean=float(row["mean_test_macro_f1"]),
            macro_f1_std=float(row["std_test_macro_f1"]),
            balanced_accuracy_mean=float(row["mean_test_balanced_accuracy"]),
            accuracy_mean=float(row["mean_test_accuracy"]),
        )
        for _, row in results.head(top).iterrows()
    )
    best = candidates[0]
    return SearchResult(
        spec=TreeSpec(features, TreeHyperparameters.from_search(search.best_params_)),
        cv_macro_f1_mean=best.macro_f1_mean,
        cv_macro_f1_std=best.macro_f1_std,
        top_candidates=candidates,
    )


@dataclass(frozen=True)
class PruningResult:
    alphas: np.ndarray
    validation_macro_f1: np.ndarray
    n_leaves: np.ndarray
    chosen_alpha: CcpAlpha


def prune_on_validation(spec: TreeSpec, splits: DataSplits, seed: Seed) -> PruningResult:
    """Tree-pruning phase: among the trees with the best validation macro-F1, keep the simplest.

    Each alpha on the minimal cost-complexity path removes the branches whose impurity
    decrease does not pay for their extra leaves; larger alpha means a smaller tree.
    """
    train_X = spec.features.transform(splits.train.X)
    path = spec.params.build(seed).cost_complexity_pruning_path(train_X, splits.train.y)
    alphas = np.unique(np.clip(path.ccp_alphas, 0.0, None))
    if alphas.size > 1:
        alphas = alphas[:-1]  # the largest alpha prunes the tree down to its root
    pruned = [spec.with_alpha(CcpAlpha(float(a))) for a in alphas]
    trees = [s.fit(splits.train, seed) for s in pruned]
    f1 = np.array(
        [
            Scores.of(splits.validation.y, s.predict(t, splits.validation.X)).macro_f1
            for s, t in zip(pruned, trees, strict=False)
        ]
    )
    leaves = np.array([t.get_n_leaves() for t in trees])
    chosen = CcpAlpha(float(alphas[np.isclose(f1, f1.max())].max()))
    return PruningResult(alphas, f1, leaves, chosen)


@dataclass(frozen=True)
class DevelopedTree:
    """A tree specification after both phases, with its validation scores."""

    search: SearchResult
    pruning: PruningResult
    spec: TreeSpec
    validation: Scores


def develop_tree(
    features: FeatureSet, splits: DataSplits, grid: SearchGrid, cv: CvSettings, seed: Seed
) -> DevelopedTree:
    search = tune_tree(features, splits.train, grid, cv, seed)
    pruning = prune_on_validation(search.spec, splits, seed)
    spec = search.spec.with_alpha(pruning.chosen_alpha)
    validation = Scores.of(
        splits.validation.y, spec.predict(spec.fit(splits.train, seed), splits.validation.X)
    )
    return DevelopedTree(search, pruning, spec, validation)
