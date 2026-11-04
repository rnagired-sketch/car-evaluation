"""Instability check (Lecture 3B): hold the tree's specification fixed and re-split the data.

Decision trees can change shape when the training data changes slightly. Refitting the same
specification on many random splits shows how much its test scores, size and root split move.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from car_eval_core.dataset import CarDataset, DataSplits, SplitFractions
from car_eval_core.domain import Seed
from car_eval_core.evaluator import CarEvaluator
from car_eval_core.features import MODEL_COLUMNS
from car_eval_core.logs import log_every
from car_eval_core.metrics import Scores
from car_eval_core.tree import TreeSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StabilityStudy:
    seeds: tuple[Seed, ...]
    macro_f1: np.ndarray  # test macro-F1 per split
    false_acceptance: np.ndarray  # test false-acceptance rate per split
    leaves: np.ndarray  # tree size per split
    root_features: Mapping[str, int]  # how often each input was chosen for the root split


def stability_study(
    dataset: CarDataset, spec: TreeSpec, fractions: SplitFractions, seeds: Sequence[Seed]
) -> StabilityStudy:
    if not seeds:
        raise ValueError("stability_study needs at least one seed")
    macro_f1, false_acceptance, leaves = [], [], []
    roots: Counter[str] = Counter()
    for index, seed in enumerate(seeds, start=1):
        splits = DataSplits.stratified(dataset, fractions, seed)
        model = CarEvaluator.fit(spec, splits.train, seed)
        scores = Scores.of(splits.test.y, model.predict(splits.test.X))
        macro_f1.append(scores.macro_f1)
        false_acceptance.append(scores.false_acceptance)
        leaves.append(model.tree.get_n_leaves())
        roots[MODEL_COLUMNS[model.tree.tree_.feature[0]]] += 1
        if log_every(index, len(seeds)):
            logger.info(
                "  split %d/%d (seed %d): test macro-F1 %.3f, %d leaves",
                index,
                len(seeds),
                seed,
                scores.macro_f1,
                leaves[-1],
            )
    return StabilityStudy(
        seeds=tuple(seeds),
        macro_f1=np.array(macro_f1),
        false_acceptance=np.array(false_acceptance),
        leaves=np.array(leaves),
        root_features=dict(roots.most_common()),
    )
