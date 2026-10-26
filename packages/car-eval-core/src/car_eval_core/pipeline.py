"""End-to-end orchestration of the evaluation plan. Pure logic: no files, no printing.

Progress is reported as log records on `car_eval_core.pipeline` (numbered stages with timings).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from car_eval_core.benchmarks import AdequacyCheck, AdequacyTargets, ComparisonRow, compare_models
from car_eval_core.dataset import CarDataset, DataSplits, SplitFractions
from car_eval_core.diagnostics import Curve, confusion, depth_curve, learning_curve_of
from car_eval_core.domain import DEFAULT_SEED, Acceptability, Car, Seed
from car_eval_core.evaluator import CarEvaluator, Explanation, Importances, TreePurity
from car_eval_core.features import FeatureSet
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
    adequacy: AdequacyTargets = AdequacyTargets()
    feature_sets: tuple[FeatureSet, ...] = (FeatureSet.RAW, FeatureSet.WITH_CONCEPTS)
    compare_benchmarks: bool = True
    stability_seeds: int = 20
    permutation_repeats: int = 20

    def __post_init__(self) -> None:
        if not self.feature_sets:
            raise ValueError("feature_sets must name at least one feature set")
        if len(set(self.feature_sets)) != len(self.feature_sets):
            raise ValueError("feature_sets contains duplicates")
        if self.stability_seeds < 0:
            raise ValueError("stability_seeds must be >= 0 (0 skips the study)")
        if self.permutation_repeats < 1:
            raise ValueError("permutation_repeats must be >= 1")

    def stage_count(self) -> int:
        """Numbered stages `run_pipeline` reports: split, one per feature set, the optional
        benchmarks, test evaluation, diagnostics and the optional stability study."""
        return 3 + len(self.feature_sets) + int(self.compare_benchmarks) + int(self.stability_seeds > 0)


@dataclass(frozen=True)
class Misclassification:
    truth: Acceptability
    explanation: Explanation


@dataclass(frozen=True)
class PipelineResult:
    settings: PipelineSettings
    splits: DataSplits
    candidates: Mapping[FeatureSet, DevelopedTree]
    chosen: DevelopedTree
    comparison: tuple[ComparisonRow, ...]  # empty when benchmarks are switched off
    adequacy: AdequacyCheck
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
    """Split, build and prune trees per feature set, select on validation, test once, diagnose."""
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

    candidates: dict[FeatureSet, DevelopedTree] = {}
    for features in settings.feature_sets:
        with logged_stage(logger, stage(f"building and pruning a tree on {features.description}")):
            developed = develop_tree(features, splits, settings.grid, settings.cv, settings.seed)
            candidates[features] = developed
            logger.info("validation macro-F1 %.3f", developed.validation.macro_f1)
    chosen = max(candidates.values(), key=lambda d: d.validation.macro_f1)
    logger.info("selected on validation: %s", chosen.spec.features.description)

    comparison: tuple[ComparisonRow, ...] = ()
    if settings.compare_benchmarks:
        with logged_stage(logger, stage("fitting benchmark models")):
            comparison = compare_models(candidates, splits, settings.cv, settings.seed)

    with logged_stage(logger, stage("evaluating the selected tree on the test split")):
        evaluator = CarEvaluator.fit(chosen.spec, splits.train, settings.seed)
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
        adequacy = AdequacyCheck(chosen.validation, settings.adequacy)
        if not adequacy.passes:
            logger.warning(
                "the tree misses the adequacy targets on validation "
                "(macro-F1 %.3f, false acceptance %.3f); the proposal calls for the k-NN trade study",
                chosen.validation.macro_f1,
                chosen.validation.false_acceptance,
            )

    with logged_stage(logger, stage("computing diagnostics (depth and learning curves, importances)")):
        depth = depth_curve(chosen.spec, splits.train, settings.cv, settings.seed)
        learning = learning_curve_of(chosen.spec, splits.development, settings.seed)
        importances = evaluator.importances(splits.test, settings.permutation_repeats)

    stability = None
    if settings.stability_seeds:
        with logged_stage(logger, stage(f"stability study over {settings.stability_seeds} random splits")):
            seeds = [Seed(settings.seed + i) for i in range(settings.stability_seeds)]
            stability = stability_study(dataset, chosen.spec, settings.fractions, seeds)

    logger.info("evaluation plan finished in %.1f s", time.perf_counter() - started)
    return PipelineResult(
        settings=settings,
        splits=splits,
        candidates=candidates,
        chosen=chosen,
        comparison=comparison,
        adequacy=adequacy,
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
