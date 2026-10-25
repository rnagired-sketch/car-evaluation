"""End-to-end orchestration of the evaluation plan. Pure logic: no files, no printing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from car_eval_core.benchmarks import AdequacyCheck, AdequacyTargets, ComparisonRow, compare_models
from car_eval_core.dataset import CarDataset, DataSplits, SplitFractions
from car_eval_core.diagnostics import Curve, confusion, depth_curve, learning_curve_of
from car_eval_core.domain import DEFAULT_SEED, Acceptability, Car, Seed
from car_eval_core.evaluator import CarEvaluator, Explanation, Importances, TreePurity
from car_eval_core.features import FeatureSet
from car_eval_core.metrics import CvSettings, Scores
from car_eval_core.stability import StabilityStudy, stability_study
from car_eval_core.tree import DevelopedTree, SearchGrid, develop_tree

Progress = Callable[[str], None]


def _silent(_: str) -> None:
    pass


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


def run_pipeline(
    dataset: CarDataset, settings: PipelineSettings, progress: Progress = _silent
) -> PipelineResult:
    """Split, build and prune trees per feature set, select on validation, test once, diagnose."""
    progress("splitting data")
    splits = DataSplits.stratified(dataset, settings.fractions, settings.seed)

    candidates: dict[FeatureSet, DevelopedTree] = {}
    for features in settings.feature_sets:
        progress(f"building and pruning tree on {features.description}")
        candidates[features] = develop_tree(features, splits, settings.grid, settings.cv, settings.seed)
    chosen = max(candidates.values(), key=lambda d: d.validation.macro_f1)

    comparison: tuple[ComparisonRow, ...] = ()
    if settings.compare_benchmarks:
        progress("fitting benchmark models")
        comparison = compare_models(candidates, splits, settings.cv, settings.seed)

    progress("evaluating the selected tree on the test split")
    evaluator = CarEvaluator.fit(chosen.spec, splits.train, settings.seed)
    y_pred = evaluator.predict(splits.test.X)
    wrong = y_pred != splits.test.y.to_numpy()
    misclassified = tuple(
        Misclassification(Acceptability(int(truth)), evaluator.explain(Car.from_ranks(row)))
        for (_, row), truth in zip(splits.test.X[wrong].iterrows(), splits.test.y[wrong], strict=False)
    )

    progress("computing diagnostics")
    depth = depth_curve(chosen.spec, splits.train, settings.cv, settings.seed)
    learning = learning_curve_of(chosen.spec, splits.development, settings.seed)

    stability = None
    if settings.stability_seeds:
        progress(f"re-splitting {settings.stability_seeds} times for the stability study")
        seeds = [Seed(settings.seed + i) for i in range(settings.stability_seeds)]
        stability = stability_study(dataset, chosen.spec, settings.fractions, seeds)

    return PipelineResult(
        settings=settings,
        splits=splits,
        candidates=candidates,
        chosen=chosen,
        comparison=comparison,
        adequacy=AdequacyCheck(chosen.validation, settings.adequacy),
        evaluator=evaluator,
        test_predictions=y_pred,
        test_scores=Scores.of(splits.test.y, y_pred),
        confusion=confusion(splits.test.y, y_pred),
        purity=evaluator.purity(),
        importances=evaluator.importances(splits.test, settings.permutation_repeats),
        depth_curve=depth,
        learning_curve=learning,
        stability=stability,
        misclassified=misclassified,
    )
