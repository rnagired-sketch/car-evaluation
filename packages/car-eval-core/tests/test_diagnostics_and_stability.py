from __future__ import annotations

import numpy as np
import pytest

from car_eval_core.dataset import CarDataset, DataSplits, SplitFractions
from car_eval_core.diagnostics import Curve, confusion, depth_curve, learning_curve_of
from car_eval_core.domain import Seed
from car_eval_core.metrics import CvSettings
from car_eval_core.stability import StabilityStudy, stability_study
from car_eval_core.tree import TreeSpec

SEED = Seed(42)


class TestCurves:
    def test_from_scores_aggregates_over_folds(self) -> None:
        train = np.array([[1.0, 0.8], [0.6, 0.6]])
        cv = np.array([[0.5, 0.7], [0.4, 0.4]])
        curve = Curve.from_scores(np.array([1, 2]), train, cv)
        np.testing.assert_allclose(curve.train_mean, [0.9, 0.6])
        np.testing.assert_allclose(curve.train_std, [0.1, 0.0])
        np.testing.assert_allclose(curve.cv_mean, [0.6, 0.4])

    def test_depth_curve_shows_underfitting_at_depth_one(
        self, concept_spec: TreeSpec, splits: DataSplits, fast_cv: CvSettings
    ) -> None:
        curve = depth_curve(concept_spec, splits.train, fast_cv, SEED, depths=[1, 4, 12])
        assert curve.x.tolist() == [1, 4, 12]
        assert curve.cv_mean[0] < curve.cv_mean[-1]
        # A deep tree memorises its training folds: the gap to cross-validation is the variance.
        assert curve.train_mean[-1] == pytest.approx(1.0)
        assert curve.train_mean[-1] > curve.cv_mean[-1]

    def test_learning_curve_grows_training_size(self, concept_spec: TreeSpec, splits: DataSplits) -> None:
        curve = learning_curve_of(concept_spec, splits.development, SEED, points=4, n_splits=3)
        assert len(curve.x) == 4
        assert np.all(np.diff(curve.x) > 0)
        assert curve.cv_mean[-1] > curve.cv_mean[0]


class TestConfusion:
    def test_rows_are_truth_columns_are_predictions(self) -> None:
        matrix = confusion(np.array([0, 0, 1, 3]), np.array([0, 1, 1, 2]))
        expected = np.zeros((4, 4), dtype=int)
        expected[0, 0] = expected[0, 1] = expected[1, 1] = expected[3, 2] = 1
        np.testing.assert_array_equal(matrix, expected)

    def test_always_four_by_four(self) -> None:
        assert confusion(np.array([0]), np.array([0])).shape == (4, 4)


@pytest.fixture(scope="module")
def study(dataset: CarDataset, concept_spec: TreeSpec) -> StabilityStudy:
    return stability_study(dataset, concept_spec, SplitFractions(), [Seed(1), Seed(2), Seed(3)])


class TestStability:
    def test_one_entry_per_seed(self, study: StabilityStudy) -> None:
        assert study.seeds == (1, 2, 3)
        for values in (study.macro_f1, study.false_acceptance, study.leaves):
            assert values.shape == (3,)
        assert sum(study.root_features.values()) == 3

    def test_root_features_are_ranked(self, study: StabilityStudy) -> None:
        counts = list(study.root_features.values())
        assert counts == sorted(counts, reverse=True)

    def test_scores_stay_high_on_every_split(self, study: StabilityStudy) -> None:
        assert study.macro_f1.min() >= 0.9
        assert (study.leaves > 1).all()

    def test_requires_seeds(self, dataset: CarDataset, concept_spec: TreeSpec) -> None:
        with pytest.raises(ValueError, match="at least one seed"):
            stability_study(dataset, concept_spec, SplitFractions(), [])
