"""Instability check (Lecture 3B): hold the specification fixed and re-split the data."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from car_eval_core.benchmarks import fit_rev0_tree
from car_eval_core.dataset import CarDataset, DataSplits, SplitFractions
from car_eval_core.domain import Seed
from car_eval_core.evaluator import CarEvaluator
from car_eval_core.logs import log_every
from car_eval_core.metrics import Scores
from car_eval_core.tree import TreeSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StabilityStudy:
    seeds: tuple[Seed, ...]
    rev0_macro_f1: np.ndarray
    rev1_macro_f1: np.ndarray
    rev1_false_acceptance: np.ndarray
    rev1_leaves: np.ndarray
    root_features: Mapping[str, int]

    @property
    def rev1_wins(self) -> int:
        """Splits on which rev1 scores at least as well as rev0."""
        return int((self.rev1_macro_f1 >= self.rev0_macro_f1).sum())


def stability_study(
    dataset: CarDataset, spec: TreeSpec, fractions: SplitFractions, seeds: Sequence[Seed]
) -> StabilityStudy:
    if not seeds:
        raise ValueError("stability_study needs at least one seed")
    rev0, rev1, false_acceptance, leaves = [], [], [], []
    roots: Counter[str] = Counter()
    for index, seed in enumerate(seeds, start=1):
        splits = DataSplits.stratified(dataset, fractions, seed)
        rev0.append(
            Scores.of(splits.test.y, fit_rev0_tree(splits.train, seed).predict(splits.test.X)).macro_f1
        )
        model = CarEvaluator.fit(spec, splits.train, seed)
        scores = Scores.of(splits.test.y, model.predict(splits.test.X))
        rev1.append(scores.macro_f1)
        false_acceptance.append(scores.false_acceptance)
        leaves.append(model.tree.get_n_leaves())
        roots[spec.features.columns[model.tree.tree_.feature[0]]] += 1
        if log_every(index, len(seeds)):
            logger.info(
                "  split %d/%d (seed %d): rev0 macro-F1 %.3f, rev1 %.3f",
                index,
                len(seeds),
                seed,
                rev0[-1],
                rev1[-1],
            )
    return StabilityStudy(
        seeds=tuple(seeds),
        rev0_macro_f1=np.array(rev0),
        rev1_macro_f1=np.array(rev1),
        rev1_false_acceptance=np.array(false_acceptance),
        rev1_leaves=np.array(leaves),
        root_features=dict(roots.most_common()),
    )
