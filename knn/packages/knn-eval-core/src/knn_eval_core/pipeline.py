"""End-to-end orchestration of the k-NN evaluation. Pure logic: no files, no printing.

It uses the same stratified 50/20/30 split, seed and scores as the decision tree application,
so the two models' results can be compared line by line. Progress is reported as log records
on `knn_eval_core.pipeline` (numbered stages with timings).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from knn_eval_core.dataset import CarDataset, DataSplits, SplitFractions
from knn_eval_core.diagnostics import (
    Curve,
    PredictionCost,
    confusion,
    k_curve,
    learning_curve_of,
    prediction_cost,
)
from knn_eval_core.domain import DEFAULT_SEED, Acceptability, Car, Seed
from knn_eval_core.evaluator import Explanation, KnnEvaluator
from knn_eval_core.features import InputSet
from knn_eval_core.logs import Stage, logged_stage
from knn_eval_core.metrics import CvSettings, Scores
from knn_eval_core.neighbors import SearchGrid, SearchResult, tune_knn
from knn_eval_core.stability import StabilityStudy, stability_study

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineSettings:
    seed: Seed = DEFAULT_SEED
    fractions: SplitFractions = SplitFractions()
    inputs: InputSet = InputSet.WITH_CONCEPTS  # the decision tree's inputs, for a like-for-like comparison
    grid: SearchGrid = SearchGrid()
    cv: CvSettings = CvSettings()
    stability_seeds: int = 20
    permutation_repeats: int = 20

    def __post_init__(self) -> None:
        if self.stability_seeds < 0:
            raise ValueError("stability_seeds must be >= 0 (0 skips the study)")
        if self.permutation_repeats < 1:
            raise ValueError("permutation_repeats must be >= 1")

    def stage_count(self) -> int:
        """Numbered stages `run_pipeline` reports: split, grid search, validation and test,
        diagnostics, and the optional stability study."""
        return 4 + int(self.stability_seeds > 0)


@dataclass(frozen=True)
class Misclassification:
    truth: Acceptability
    explanation: Explanation


@dataclass(frozen=True)
class PipelineResult:
    settings: PipelineSettings
    splits: DataSplits
    search: SearchResult
    evaluator: KnnEvaluator
    validation_scores: Scores
    test_predictions: np.ndarray
    test_scores: Scores
    confusion: np.ndarray
    importances: dict[str, float]
    k_curve: Curve
    learning_curve: Curve
    cost: PredictionCost
    stability: StabilityStudy | None
    misclassified: tuple[Misclassification, ...]


def run_pipeline(dataset: CarDataset, settings: PipelineSettings) -> PipelineResult:
    """Split, search the k-NN settings, check on validation, test once, then diagnose."""
    total = settings.stage_count()
    numbers = iter(range(1, total + 1))

    def stage(title: str) -> Stage:
        return Stage(next(numbers), total, title)

    started = time.perf_counter()
    logger.info("starting the k-NN evaluation: %d stages, seed %d", total, settings.seed)

    with logged_stage(logger, stage("splitting data 50/20/30 (stratified)")):
        splits = DataSplits.stratified(dataset, settings.fractions, settings.seed)
        logger.info(
            "train %d, validation %d, test %d cars",
            len(splits.train),
            len(splits.validation),
            len(splits.test),
        )

    with logged_stage(
        logger, stage(f"searching K, distance, voting and scaling on {settings.inputs.description}")
    ):
        search = tune_knn(settings.inputs, splits.train, settings.grid, settings.cv, settings.seed)

    with logged_stage(logger, stage("scoring the chosen k-NN on the validation and test splits")):
        evaluator = KnnEvaluator.fit(search.spec, splits.train, settings.seed)
        validation_scores = Scores.of(splits.validation.y, evaluator.predict(splits.validation.X))
        y_pred = evaluator.predict(splits.test.X)
        test_scores = Scores.of(splits.test.y, y_pred)
        wrong = y_pred != splits.test.y.to_numpy()
        misclassified = tuple(
            Misclassification(Acceptability(int(truth)), evaluator.explain(Car.from_ranks(row)))
            for (_, row), truth in zip(splits.test.X[wrong].iterrows(), splits.test.y[wrong], strict=True)
        )
        logger.info("validation macro-F1 %.3f", validation_scores.macro_f1)
        logger.info(
            "test macro-F1 %.3f, accuracy %.3f, %d of %d cars misclassified",
            test_scores.macro_f1,
            test_scores.accuracy,
            len(misclassified),
            len(splits.test),
        )

    with logged_stage(logger, stage("computing diagnostics (K curve, learning curve, importances, cost)")):
        curve_k = k_curve(search.spec, splits.train, settings.cv, settings.seed)
        learning = learning_curve_of(search.spec, splits.development, settings.seed)
        importances = evaluator.permutation_importance(splits.test, settings.permutation_repeats)
        cost = prediction_cost(evaluator, splits.test.X)
        logger.info(
            "prediction cost: %.1f microseconds per car against %d stored cars",
            cost.seconds_per_car * 1e6,
            cost.stored_cars,
        )

    stability = None
    if settings.stability_seeds:
        with logged_stage(logger, stage(f"stability study over {settings.stability_seeds} random splits")):
            seeds = [Seed(settings.seed + i) for i in range(settings.stability_seeds)]
            stability = stability_study(dataset, search.spec, settings.fractions, seeds)

    logger.info("k-NN evaluation finished in %.1f s", time.perf_counter() - started)
    return PipelineResult(
        settings=settings,
        splits=splits,
        search=search,
        evaluator=evaluator,
        validation_scores=validation_scores,
        test_predictions=y_pred,
        test_scores=test_scores,
        confusion=confusion(splits.test.y, y_pred),
        importances=importances,
        k_curve=curve_k,
        learning_curve=learning,
        cost=cost,
        stability=stability,
        misclassified=misclassified,
    )
