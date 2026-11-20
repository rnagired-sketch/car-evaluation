"""Stability check: hold the k-NN settings fixed and re-split the data many times."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from knn_eval_core.dataset import CarDataset, DataSplits, SplitFractions
from knn_eval_core.domain import Seed
from knn_eval_core.evaluator import KnnEvaluator
from knn_eval_core.logs import log_every
from knn_eval_core.metrics import Scores
from knn_eval_core.neighbors import KnnSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StabilityStudy:
    seeds: tuple[Seed, ...]
    macro_f1: np.ndarray  # test macro-F1 per split
    false_acceptance: np.ndarray  # test false-acceptance rate per split


def stability_study(
    dataset: CarDataset, spec: KnnSpec, fractions: SplitFractions, seeds: Sequence[Seed]
) -> StabilityStudy:
    """Uses the same splits (seeds) as the decision tree's study, so the two are comparable."""
    if not seeds:
        raise ValueError("stability_study needs at least one seed")
    macro_f1, false_acceptance = [], []
    for index, seed in enumerate(seeds, start=1):
        splits = DataSplits.stratified(dataset, fractions, seed)
        model = KnnEvaluator.fit(spec, splits.train, seed)
        scores = Scores.of(splits.test.y, model.predict(splits.test.X))
        macro_f1.append(scores.macro_f1)
        false_acceptance.append(scores.false_acceptance)
        if log_every(index, len(seeds)):
            logger.info(
                "  split %d/%d (seed %d): test macro-F1 %.3f", index, len(seeds), seed, scores.macro_f1
            )
    return StabilityStudy(tuple(seeds), np.array(macro_f1), np.array(false_acceptance))
