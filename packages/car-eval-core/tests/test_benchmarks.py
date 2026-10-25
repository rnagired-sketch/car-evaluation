from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from car_eval_core.benchmarks import (
    REV0_GRID,
    AdequacyCheck,
    AdequacyTargets,
    ComparisonRow,
    compare_models,
    fit_knn,
    fit_majority_baseline,
    fit_rev0_tree,
    tree_label,
)
from car_eval_core.dataset import DataSplits
from car_eval_core.domain import Acceptability, Seed
from car_eval_core.features import FeatureSet
from car_eval_core.metrics import CvSettings, Scores
from car_eval_core.tree import DevelopedTree, SearchGrid, develop_tree

SEED = Seed(42)


@pytest.fixture(scope="module")
def candidates(
    splits: DataSplits, tiny_grid: SearchGrid, fast_cv: CvSettings
) -> dict[FeatureSet, DevelopedTree]:
    return {f: develop_tree(f, splits, tiny_grid, fast_cv, SEED) for f in FeatureSet}


@pytest.fixture(scope="module")
def rows(
    candidates: dict[FeatureSet, DevelopedTree], splits: DataSplits, fast_cv: CvSettings
) -> tuple[ComparisonRow, ...]:
    return compare_models(candidates, splits, fast_cv, SEED)


class TestReferenceModels:
    def test_majority_baseline_always_says_unacc(self, splits: DataSplits) -> None:
        baseline = fit_majority_baseline(splits.train)
        assert set(baseline.predict(splits.test.X)) == {Acceptability.UNACC}

    def test_rev0_recipe_picks_from_the_rev0_grid(self, splits: DataSplits) -> None:
        tree = fit_rev0_tree(splits.train, SEED)
        assert isinstance(tree, DecisionTreeClassifier)
        assert tree.max_depth in REV0_GRID["max_depth"]
        assert tree.class_weight is None
        assert tree.ccp_alpha == 0.0
        assert tuple(tree.feature_names_in_)  # pyright: ignore[reportAttributeAccessIssue]  (set by fit) == FeatureSet.RAW.columns

    def test_knn_scales_features_and_uses_the_feature_set(
        self, splits: DataSplits, fast_cv: CvSettings
    ) -> None:
        knn = fit_knn(FeatureSet.WITH_CONCEPTS, splits.train, fast_cv, SEED)
        assert [name for name, _ in knn.steps] == ["scale", "knn"]
        X = FeatureSet.WITH_CONCEPTS.transform(splits.validation.X)
        scaled = knn.named_steps["scale"].transform(FeatureSet.WITH_CONCEPTS.transform(splits.train.X))
        assert scaled.min() == pytest.approx(0.0)
        assert scaled.max() == pytest.approx(1.0)
        assert Scores.of(splits.validation.y, knn.predict(X)).macro_f1 > 0.5


class TestCompareModels:
    def test_one_row_per_model_and_split(self, rows: tuple[ComparisonRow, ...]) -> None:
        models = list(dict.fromkeys(r.model for r in rows))
        assert models == [
            "majority baseline",
            "rev0 tree",
            "k-NN (with_concepts)",
            tree_label(FeatureSet.RAW),
            tree_label(FeatureSet.WITH_CONCEPTS),
        ]
        assert len(rows) == 2 * len(models)
        assert {r.split for r in rows} == {"validation", "test"}

    def test_tree_rows_match_the_developed_trees(
        self, rows: tuple[ComparisonRow, ...], candidates: dict[FeatureSet, DevelopedTree]
    ) -> None:
        by_key = {(r.model, r.split): r.scores for r in rows}
        for features, developed in candidates.items():
            assert by_key[(tree_label(features), "validation")] == developed.validation

    def test_rev1_beats_rev0_and_the_baseline_on_test(self, rows: tuple[ComparisonRow, ...]) -> None:
        by_key = {(r.model, r.split): r.scores for r in rows}
        rev1 = by_key[(tree_label(FeatureSet.WITH_CONCEPTS), "test")].macro_f1
        assert rev1 > by_key[("rev0 tree", "test")].macro_f1 > by_key[("majority baseline", "test")].macro_f1

    def test_requires_candidates(self, splits: DataSplits, fast_cv: CvSettings) -> None:
        with pytest.raises(ValueError, match="at least one"):
            compare_models({}, splits, fast_cv, SEED)


def _scores(macro_f1: float, false_acceptance: float) -> Scores:
    return Scores(0.9, 0.9, macro_f1, false_acceptance, 0.0, 0)


class TestAdequacy:
    @pytest.mark.parametrize(
        ("macro_f1", "false_acceptance", "passes"),
        [(0.95, 0.01, True), (0.90, 0.05, True), (0.89, 0.01, False), (0.95, 0.06, False), (0.5, 0.5, False)],
    )
    def test_both_targets_must_be_met(self, macro_f1: float, false_acceptance: float, passes: bool) -> None:
        assert AdequacyCheck(_scores(macro_f1, false_acceptance), AdequacyTargets()).passes is passes

    def test_nan_false_acceptance_fails(self) -> None:
        assert not AdequacyCheck(_scores(0.99, float("nan")), AdequacyTargets()).passes

    def test_custom_targets(self) -> None:
        targets = AdequacyTargets(min_macro_f1=0.99, max_false_acceptance=0.0)
        assert not AdequacyCheck(_scores(0.98, 0.0), targets).passes
        assert AdequacyCheck(_scores(0.995, 0.0), targets).passes

    @pytest.mark.parametrize("field", ["min_macro_f1", "max_false_acceptance"])
    @pytest.mark.parametrize("value", [-0.1, 1.5])
    def test_targets_must_be_probabilities(self, field: str, value: float) -> None:
        with pytest.raises(ValueError, match=field):
            dataclasses.replace(AdequacyTargets(), **{field: value})


def test_rev0_grid_matches_original_code() -> None:
    assert REV0_GRID == {
        "max_depth": [4, 6, 8, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],
    }
    assert np.prod([len(v) for v in REV0_GRID.values()]) == 90
