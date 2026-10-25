"""The trained model: rating, confidence, decision path, purity and feature importance.

Typestate: a `CarEvaluator` only comes from `CarEvaluator.fit`, so there is no
unfitted model to call `predict` on by mistake.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from car_eval_core.dataset import CarDataset, Partition
from car_eval_core.domain import Acceptability, Car, Seed
from car_eval_core.features import FEATURE_NAMES, Condition, tighten
from car_eval_core.metrics import Scores, gini
from car_eval_core.tree import TreeSpec


@dataclass(frozen=True)
class Explanation:
    car: Car
    rating: Acceptability
    confidence: float  # Laplace-smoothed share of the leaf's training cars with this rating
    leaf_counts: Mapping[Acceptability, int]
    path: tuple[Condition, ...]  # one merged condition per feature, in first-seen order

    @property
    def leaf_support(self) -> int:
        return sum(self.leaf_counts.values())


@dataclass(frozen=True)
class TreePurity:
    root_gini: float
    weighted_leaf_gini: float
    pure_leaf_share: float
    n_leaves: int
    depth: int


@dataclass(frozen=True)
class Importances:
    impurity: Mapping[str, float]  # mean impurity decrease per model input
    permutation: Mapping[str, float]  # macro-F1 drop when one raw attribute is shuffled


class CarEvaluator:
    """A fitted, pruned decision tree that rates cars and explains its ratings."""

    def __init__(self, spec: TreeSpec, tree: DecisionTreeClassifier, training: Partition, seed: Seed):
        self.__spec = spec
        self.__tree = tree
        # Owned copy: later changes to the caller's frames cannot alter explanations.
        self.__training = Partition(training.X.copy(), training.y.copy())
        self.__seed = seed
        self.__training_leaves = tree.apply(spec.features.transform(training.X))

    @staticmethod
    def fit(spec: TreeSpec, training: Partition, seed: Seed) -> CarEvaluator:
        return CarEvaluator(spec, spec.fit(training, seed), training, seed)

    @property
    def tree(self) -> DecisionTreeClassifier:
        return self.__tree

    @property
    def spec(self) -> TreeSpec:
        return self.__spec

    @property
    def seed(self) -> Seed:
        return self.__seed

    @property
    def training_size(self) -> int:
        return len(self.__training)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Ratings for encoded raw attributes; derived model inputs are added here."""
        return self.__spec.predict(self.__tree, X)

    def rate(self, car: Car) -> Acceptability:
        return Acceptability(int(self.predict(car.encoded())[0]))

    def purity(self) -> TreePurity:
        """Unweighted class mix of the training cars that land in each leaf."""
        counts = pd.crosstab(self.__training_leaves, self.__training.y).to_numpy()
        leaf_gini = gini(counts / counts.sum(axis=1, keepdims=True))
        root = self.__training.y.value_counts(normalize=True).to_numpy()
        return TreePurity(
            root_gini=float(gini(root)),
            weighted_leaf_gini=float(np.average(leaf_gini, weights=counts.sum(axis=1))),
            pure_leaf_share=float(np.mean(np.isclose(leaf_gini, 0.0))),
            n_leaves=int(self.__tree.get_n_leaves()),
            depth=int(self.__tree.get_depth()),
        )

    def explain(self, car: Car) -> Explanation:
        x = self.__spec.features.transform(car.encoded())
        columns = self.__spec.features.columns
        structure = self.__tree.tree_
        conditions: dict[str, Condition] = {}
        for node in self.__tree.decision_path(x).indices:
            if structure.children_left[node] == structure.children_right[node]:  # leaf
                continue
            column = columns[structure.feature[node]]
            threshold = float(structure.threshold[node])
            step = self.__spec.features.condition(column, threshold, bool(x[column].iloc[0] <= threshold))
            conditions[column] = tighten(conditions[column], step) if column in conditions else step

        leaf = int(self.__tree.apply(x)[0])
        in_leaf = self.__training.y.to_numpy()[self.__training_leaves == leaf]
        counts = {a: int((in_leaf == a).sum()) for a in Acceptability}
        rating = Acceptability(int(self.__tree.predict(x)[0]))
        # Laplace smoothing: a pure leaf holding 2 cars earns far less confidence than one holding 200.
        confidence = (counts[rating] + 1) / (len(in_leaf) + len(Acceptability))
        return Explanation(car, rating, confidence, counts, tuple(conditions.values()))

    def with_new_labels(self, new_data: CarDataset) -> CarEvaluator:
        """Monitor-and-update step: refit with the same specification on old + new labels.

        Rows in `new_data` replace older rows with the same attributes.
        """
        X = pd.concat([self.__training.X, new_data.data.X], ignore_index=True)
        y = pd.concat([self.__training.y, new_data.data.y], ignore_index=True)
        keep = ~X.duplicated(keep="last")
        return CarEvaluator.fit(self.__spec, Partition(X[keep], y[keep]), self.__seed)

    def importances(self, data: Partition, repeats: int = 20) -> Importances:
        impurity = dict(
            zip(self.__spec.features.columns, map(float, self.__tree.feature_importances_), strict=False)
        )
        return Importances(impurity, self.permutation_importance(data, repeats))

    def permutation_importance(self, data: Partition, repeats: int = 20) -> dict[str, float]:
        """Drop in macro-F1 when one raw attribute is shuffled (concept scores follow along)."""
        if repeats < 1:
            raise ValueError("repeats must be at least 1")
        rng = np.random.default_rng(self.__seed)
        reference = Scores.of(data.y, self.predict(data.X)).macro_f1
        drops: dict[str, float] = {}
        for name in FEATURE_NAMES:
            shuffled_scores = [
                Scores.of(
                    data.y, self.predict(data.X.assign(**{name: rng.permutation(data.X[name].to_numpy())}))
                ).macro_f1
                for _ in range(repeats)
            ]
            drops[name] = reference - float(np.mean(shuffled_scores))
        return drops
