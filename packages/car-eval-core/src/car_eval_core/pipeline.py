"""End-to-end orchestration of the evaluation plan. Pure logic: no files, no printing.

Progress is reported as log records on `car_eval_core.pipeline` (numbered stages with timings).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from car_eval_core.dataset import CarDataset, DataSplits, SplitFractions
from car_eval_core.diagnostics import Curve, confusion, depth_curve, learning_curve_of
from car_eval_core.domain import DEFAULT_SEED, Acceptability, Car, Seed
from car_eval_core.evaluator import CarEvaluator, Explanation, Importances, TreePurity
from car_eval_core.features import MODEL_INPUTS
from car_eval_core.logs import Stage, logged_stage
from car_eval_core.metrics import CvSettings, Scores
from car_eval_core.stability import StabilityStudy, stability_study
from car_eval_core.tree import DevelopedTree, SearchGrid, develop_tree

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineSettings:
    seed: Seed = DEFAULT_SEED
    fractions: SplitFractions = SplitFractions()
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
        """Numbered stages `run_pipeline` reports: split, build and prune, test evaluation,
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
    developed: DevelopedTree  # grid search, pruning and the final specification
    evaluator: CarEvaluator
    test_predictions: np.ndarray
    test_scores: Scores
    confusion: np.ndarray
    purity: TreePurity
    importances: Importances
    depth_curve: Curve
    learning_curve: Curve
    stability: StabilityStudy | None
    misclassified: tuple[Misclassification, ...]


def run_pipeline(dataset: CarDataset, settings: PipelineSettings) -> PipelineResult:
    """Split, build and prune the tree, test it once, then diagnose it."""
    total = settings.stage_count()
    numbers = iter(range(1, total + 1))

    def stage(title: str) -> Stage:
        return Stage(next(numbers), total, title)

    started = time.perf_counter()
    logger.info("starting the evaluation plan: %d stages, seed %d", total, settings.seed)

    with logged_stage(logger, stage("splitting data 50/20/30 (stratified)")):
        splits = DataSplits.stratified(dataset, settings.fractions, settings.seed)
        logger.info(
            "train %d, validation %d, test %d cars",
            len(splits.train),
            len(splits.validation),
            len(splits.test),
        )

    with logged_stage(logger, stage(f"building and pruning the tree on {MODEL_INPUTS}")):
        developed = develop_tree(splits, settings.grid, settings.cv, settings.seed)
        logger.info("validation macro-F1 %.3f", developed.validation.macro_f1)

    with logged_stage(logger, stage("evaluating the tree on the test split")):
        evaluator = CarEvaluator.fit(developed.spec, splits.train, settings.seed)
        y_pred = evaluator.predict(splits.test.X)
        test_scores = Scores.of(splits.test.y, y_pred)
        wrong = y_pred != splits.test.y.to_numpy()
        misclassified = tuple(
            Misclassification(Acceptability(int(truth)), evaluator.explain(Car.from_ranks(row)))
            for (_, row), truth in zip(splits.test.X[wrong].iterrows(), splits.test.y[wrong], strict=False)
        )
        logger.info(
            "test macro-F1 %.3f, accuracy %.3f, %d of %d cars misclassified",
            test_scores.macro_f1,
            test_scores.accuracy,
            len(misclassified),
            len(splits.test),
        )

    with logged_stage(logger, stage("computing diagnostics (depth and learning curves, importances)")):
        depth = depth_curve(developed.spec, splits.train, settings.cv, settings.seed)
        learning = learning_curve_of(developed.spec, splits.development, settings.seed)
        importances = evaluator.importances(splits.test, settings.permutation_repeats)

    stability = None
    if settings.stability_seeds:
        with logged_stage(logger, stage(f"stability study over {settings.stability_seeds} random splits")):
            seeds = [Seed(settings.seed + i) for i in range(settings.stability_seeds)]
            stability = stability_study(dataset, developed.spec, settings.fractions, seeds)

    logger.info("evaluation plan finished in %.1f s", time.perf_counter() - started)
    return PipelineResult(
        settings=settings,
        splits=splits,
        developed=developed,
        evaluator=evaluator,
        test_predictions=y_pred,
        test_scores=test_scores,
        confusion=confusion(splits.test.y, y_pred),
        purity=evaluator.purity(),
        importances=importances,
        depth_curve=depth,
        learning_curve=learning,
        stability=stability,
        misclassified=misclassified,
    )
