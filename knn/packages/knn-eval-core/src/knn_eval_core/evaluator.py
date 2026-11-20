"""The fitted k-NN model: ratings, vote-based confidence, the neighbors behind each rating.

Typestate: a `KnnEvaluator` only comes from `KnnEvaluator.fit`, so there is no unfitted
model to call `predict` on by mistake.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from knn_eval_core.dataset import CarDataset, Partition
from knn_eval_core.domain import Acceptability, Car, Seed
from knn_eval_core.features import FEATURE_NAMES
from knn_eval_core.metrics import Scores
from knn_eval_core.neighbors import KnnSpec


@dataclass(frozen=True)
class Neighbor:
    car: Car
    rating: Acceptability
    distance: float
    weight: float  # this neighbor's share of the vote, 0..1


@dataclass(frozen=True)
class Explanation:
    car: Car
    rating: Acceptability
    confidence: float  # the winning class's share of the (weighted) vote
    votes: Mapping[Acceptability, float]  # vote share per class, summing to 1
    neighbors: tuple[Neighbor, ...]  # nearest first


class KnnEvaluator:
    """A fitted k-NN model that rates cars and shows the training cars behind each rating."""

    def __init__(self, spec: KnnSpec, model: Pipeline, training: Partition, seed: Seed):
        self.__spec = spec
        self.__model = model
        # Owned copy: later changes to the caller's frames cannot alter explanations.
        self.__training = Partition(training.X.copy(), training.y.copy())
        self.__seed = seed

    @staticmethod
    def fit(spec: KnnSpec, training: Partition, seed: Seed) -> KnnEvaluator:
        return KnnEvaluator(spec, spec.fit(training), training, seed)

    @property
    def spec(self) -> KnnSpec:
        return self.__spec

    @property
    def model(self) -> Pipeline:
        return self.__model

    @property
    def seed(self) -> Seed:
        return self.__seed

    @property
    def training_size(self) -> int:
        return len(self.__training)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Ratings for encoded raw attributes; derived inputs and scaling are applied here."""
        return self.__spec.predict(self.__model, X)

    def rate(self, car: Car) -> Acceptability:
        return Acceptability(int(self.predict(car.encoded())[0]))

    def explain(self, car: Car) -> Explanation:
        x = self.__spec.inputs.transform(car.encoded())
        scaled = self.__model.named_steps["scale"].transform(x)
        knn = self.__model.named_steps["knn"]
        distances, indices = knn.kneighbors(scaled)
        distances, indices = distances[0], indices[0]
        probabilities = knn.predict_proba(scaled)[0]
        votes = {Acceptability(int(c)): float(p) for c, p in zip(knn.classes_, probabilities, strict=True)}
        votes = {a: votes.get(a, 0.0) for a in Acceptability}
        rating = Acceptability(int(knn.predict(scaled)[0]))
        raw_weights = self._weights(distances)
        shares = raw_weights / raw_weights.sum()
        neighbors = tuple(
            Neighbor(
                car=Car.from_ranks(self.__training.X.iloc[i]),
                rating=Acceptability(int(self.__training.y.iloc[i])),
                distance=float(d),
                weight=float(w),
            )
            for i, d, w in zip(indices, distances, shares, strict=True)
        )
        return Explanation(car, rating, votes[rating], votes, neighbors)

    def _weights(self, distances: np.ndarray) -> np.ndarray:
        weights = self.__model.named_steps["knn"].weights
        if weights == "uniform":
            return np.ones_like(distances)
        if weights == "distance":
            exact = distances == 0
            if exact.any():
                return exact.astype(float)
            return 1.0 / distances
        return np.asarray(weights(distances[np.newaxis, :]))[0]

    def with_new_labels(self, new_data: CarDataset) -> KnnEvaluator:
        """Monitor-and-update step: store old + new labelled cars (new rows replace old ones)."""
        X = pd.concat([self.__training.X, new_data.data.X], ignore_index=True)
        y = pd.concat([self.__training.y, new_data.data.y], ignore_index=True)
        keep = ~X.duplicated(keep="last")
        return KnnEvaluator.fit(self.__spec, Partition(X[keep], y[keep]), self.__seed)

    def permutation_importance(self, data: Partition, repeats: int = 20) -> dict[str, float]:
        """Drop in macro-F1 when one raw attribute is shuffled (concept scores follow along)."""
        if repeats < 1:
            raise ValueError("repeats must be at least 1")
        rng = np.random.default_rng(self.__seed)
        reference = Scores.of(data.y, self.predict(data.X)).macro_f1
        drops: dict[str, float] = {}
        for name in FEATURE_NAMES:
            shuffled = [
                Scores.of(
                    data.y, self.predict(data.X.assign(**{name: rng.permutation(data.X[name].to_numpy())}))
                ).macro_f1
                for _ in range(repeats)
            ]
            drops[name] = reference - float(np.mean(shuffled))
        return drops
