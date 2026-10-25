from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from car_eval_core.dataset import DataSplits
from car_eval_core.domain import CcpAlpha, Seed
from car_eval_core.features import FeatureSet
from car_eval_core.metrics import CvSettings, Scores
from car_eval_core.tree import (
    PruningResult,
    SearchGrid,
    TreeHyperparameters,
    TreeSpec,
    develop_tree,
    prune_on_validation,
    tune_tree,
)

SEED = Seed(42)


class TestSearchGrid:
    def test_default_size(self) -> None:
        assert SearchGrid().size == 2 * 5 * 2 * 3 * 2

    def test_param_grid_uses_sklearn_names(self) -> None:
        grid = SearchGrid().as_param_grid()
        assert set(grid) == {
            "criterion",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "class_weight",
        }
        assert grid["max_depth"][-1] is None
        assert all(isinstance(v, list) for v in grid.values())

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"criterion": ()}, "empty"),
            ({"max_depth": (0,)}, "max_depth"),
            ({"min_samples_split": (1,)}, "min_samples_split"),
            ({"min_samples_leaf": (0,)}, "min_samples_leaf"),
        ],
    )
    def test_rejects_invalid_values(self, change: dict[str, tuple[object, ...]], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(SearchGrid(), **change)


class TestTreeHyperparameters:
    def test_build_passes_every_field_to_sklearn(self) -> None:
        params = TreeHyperparameters("gini", 7, 3, 2, "balanced", CcpAlpha(0.01))
        tree = params.build(Seed(5))
        assert tree.get_params()["criterion"] == "gini"
        assert tree.max_depth == 7
        assert tree.min_samples_split == 3
        assert tree.min_samples_leaf == 2
        assert tree.class_weight == "balanced"
        assert tree.ccp_alpha == 0.01
        assert tree.random_state == 5

    def test_from_search_ignores_alpha(self) -> None:
        params = TreeHyperparameters.from_search(
            {
                "criterion": "entropy",
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "class_weight": None,
            }
        )
        assert params == TreeHyperparameters("entropy", None, 2, 1, None)

    def test_rejects_negative_alpha(self) -> None:
        with pytest.raises(ValueError, match="ccp_alpha"):
            TreeHyperparameters("gini", None, 2, 1, None, CcpAlpha(-0.1))


class TestTreeSpec:
    def test_with_alpha_returns_new_spec(self, concept_spec: TreeSpec) -> None:
        pruned = concept_spec.with_alpha(CcpAlpha(0.02))
        assert pruned.params.ccp_alpha == 0.02
        assert concept_spec.params.ccp_alpha == 0.0
        assert pruned.features is concept_spec.features

    @pytest.mark.parametrize("features", list(FeatureSet))
    def test_fit_uses_the_feature_set_columns(self, features: FeatureSet, splits: DataSplits) -> None:
        spec = TreeSpec(features, TreeHyperparameters("gini", 3, 2, 1, None))
        tree = spec.fit(splits.train, SEED)
        assert tuple(tree.feature_names_in_)  # pyright: ignore[reportAttributeAccessIssue]  (set by fit) == features.columns
        assert spec.predict(tree, splits.test.X).shape == (len(splits.test),)

    def test_fit_is_deterministic(self, concept_spec: TreeSpec, splits: DataSplits) -> None:
        a = concept_spec.predict(concept_spec.fit(splits.train, SEED), splits.test.X)
        b = concept_spec.predict(concept_spec.fit(splits.train, SEED), splits.test.X)
        np.testing.assert_array_equal(a, b)


class TestTuneTree:
    def test_best_comes_from_the_grid_and_leads_the_ranking(
        self, splits: DataSplits, small_grid: SearchGrid, fast_cv: CvSettings
    ) -> None:
        result = tune_tree(FeatureSet.RAW, splits.train, small_grid, fast_cv, SEED, top=3)
        params = result.spec.params
        assert params.criterion in small_grid.criterion
        assert params.max_depth in small_grid.max_depth
        assert params.min_samples_leaf in small_grid.min_samples_leaf
        assert params.ccp_alpha == 0.0
        assert result.spec.features is FeatureSet.RAW
        assert len(result.top_candidates) == 3
        means = [c.macro_f1_mean for c in result.top_candidates]
        assert means == sorted(means, reverse=True)
        assert result.top_candidates[0].params == params
        assert result.cv_macro_f1_mean == means[0]

    def test_scores_are_probabilities(
        self, splits: DataSplits, tiny_grid: SearchGrid, fast_cv: CvSettings
    ) -> None:
        result = tune_tree(FeatureSet.RAW, splits.train, tiny_grid, fast_cv, SEED)
        candidate = result.top_candidates[0]
        for value in (candidate.macro_f1_mean, candidate.balanced_accuracy_mean, candidate.accuracy_mean):
            assert 0.0 <= value <= 1.0
        assert candidate.macro_f1_std >= 0.0

    def test_deep_unweighted_tree_beats_a_stump(self, splits: DataSplits, fast_cv: CvSettings) -> None:
        grid = SearchGrid(("gini",), (1, None), (2,), (1,), (None,))
        result = tune_tree(FeatureSet.RAW, splits.train, grid, fast_cv, SEED)
        assert result.spec.params.max_depth is None


@pytest.fixture(scope="module")
def pruning(splits: DataSplits, concept_spec: TreeSpec) -> PruningResult:
    return prune_on_validation(concept_spec, splits, SEED)


class TestPruning:
    def test_arrays_are_aligned(self, pruning: PruningResult) -> None:
        assert pruning.alphas.shape == pruning.validation_macro_f1.shape == pruning.n_leaves.shape

    def test_alphas_are_sorted_and_start_at_zero(self, pruning: PruningResult) -> None:
        assert pruning.alphas[0] == 0.0
        assert np.all(np.diff(pruning.alphas) > 0)

    def test_larger_alpha_never_grows_the_tree(self, pruning: PruningResult) -> None:
        assert np.all(np.diff(pruning.n_leaves) <= 0)
        assert pruning.n_leaves[-1] > 1  # the root-only tree is excluded

    def test_chooses_the_simplest_tree_with_the_best_score(self, pruning: PruningResult) -> None:
        best = pruning.validation_macro_f1.max()
        chosen = int(np.flatnonzero(pruning.alphas == pruning.chosen_alpha)[0])
        assert pruning.validation_macro_f1[chosen] == pytest.approx(best)
        assert not np.any(np.isclose(pruning.validation_macro_f1[chosen + 1 :], best))

    def test_validation_scores_can_be_reproduced(
        self, pruning: PruningResult, splits: DataSplits, concept_spec: TreeSpec
    ) -> None:
        spec = concept_spec.with_alpha(CcpAlpha(float(pruning.alphas[-1])))
        score = Scores.of(
            splits.validation.y, spec.predict(spec.fit(splits.train, SEED), splits.validation.X)
        )
        assert score.macro_f1 == pytest.approx(pruning.validation_macro_f1[-1])

    def test_single_class_training_data_keeps_alpha_zero(
        self, splits: DataSplits, concept_spec: TreeSpec
    ) -> None:
        unacc_only = splits.train.y == 0
        train = dataclasses.replace(splits.train, X=splits.train.X[unacc_only], y=splits.train.y[unacc_only])
        result = prune_on_validation(concept_spec, dataclasses.replace(splits, train=train), SEED)
        assert result.chosen_alpha == 0.0
        assert result.n_leaves.tolist() == [1]


class TestDevelopTree:
    def test_combines_both_phases(
        self, splits: DataSplits, tiny_grid: SearchGrid, fast_cv: CvSettings
    ) -> None:
        developed = develop_tree(FeatureSet.WITH_CONCEPTS, splits, tiny_grid, fast_cv, SEED)
        assert developed.spec == developed.search.spec.with_alpha(developed.pruning.chosen_alpha)
        refit = developed.spec.fit(splits.train, SEED)
        expected = Scores.of(splits.validation.y, developed.spec.predict(refit, splits.validation.X))
        assert developed.validation == expected

    def test_concept_features_outperform_raw_attributes(
        self, splits: DataSplits, tiny_grid: SearchGrid, fast_cv: CvSettings
    ) -> None:
        # Regression guard for the main finding of rev1.
        raw = develop_tree(FeatureSet.RAW, splits, tiny_grid, fast_cv, SEED)
        concepts = develop_tree(FeatureSet.WITH_CONCEPTS, splits, tiny_grid, fast_cv, SEED)
        assert concepts.validation.macro_f1 > raw.validation.macro_f1
        assert concepts.validation.macro_f1 >= 0.97
