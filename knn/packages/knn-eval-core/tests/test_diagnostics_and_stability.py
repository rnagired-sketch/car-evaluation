from __future__ import annotations

import numpy as np
import pytest

from knn_eval_core.dataset import CarDataset, DataSplits, SplitFractions
from knn_eval_core.diagnostics import Curve, confusion, k_curve, learning_curve_of, prediction_cost
from knn_eval_core.domain import Seed
from knn_eval_core.evaluator import KnnEvaluator
from knn_eval_core.metrics import CvSettings
from knn_eval_core.neighbors import KnnSpec
from knn_eval_core.stability import StabilityStudy, stability_study

SEED = Seed(42)


class TestCurves:
    def test_from_scores_aggregates_over_folds(self) -> None:
        curve = Curve.from_scores(
            np.array([1, 2]), np.array([[1.0, 0.8], [0.6, 0.6]]), np.array([[0.5, 0.7], [0.4, 0.4]])
        )
        np.testing.assert_allclose(curve.train_mean, [0.9, 0.6])
        np.testing.assert_allclose(curve.cv_mean, [0.6, 0.4])

    def test_k_curve_peaks_inside_the_range(
        self, spec: KnnSpec, splits: DataSplits, fast_cv: CvSettings
    ) -> None:
        # Lecture 3B: too small a K follows noise, too large a K blurs the classes.
        curve = k_curve(spec, splits.train, fast_cv, SEED, ks=(1, 7, 101))
        assert curve.x.tolist() == [1, 7, 101]
        assert curve.cv_mean[1] > curve.cv_mean[0]
        assert curve.cv_mean[1] > curve.cv_mean[2]

    def test_k_curve_drops_k_larger_than_a_fold(
        self, spec: KnnSpec, splits: DataSplits, fast_cv: CvSettings
    ) -> None:
        assert k_curve(spec, splits.train, fast_cv, SEED, ks=(5, 10_000)).x.tolist() == [5]

    def test_learning_curve_grows_the_store(self, spec: KnnSpec, splits: DataSplits) -> None:
        curve = learning_curve_of(spec, splits.development, SEED, points=4, n_splits=3)
        assert len(curve.x) == 4
        assert np.all(np.diff(curve.x) > 0)
        assert curve.x[0] >= spec.params.k
        assert curve.cv_mean[-1] > curve.cv_mean[0]


class TestConfusionAndCost:
    def test_confusion_is_four_by_four(self) -> None:
        matrix = confusion(np.array([0, 0, 1, 3]), np.array([0, 1, 1, 2]))
        assert matrix.shape == (4, 4)
        assert matrix.sum() == 4

    def test_prediction_cost(self, evaluator: KnnEvaluator, splits: DataSplits) -> None:
        cost = prediction_cost(evaluator, splits.test.X, repeats=2)
        assert cost.stored_cars == len(splits.train)
        assert cost.predicted_cars == len(splits.test)
        assert 0 < cost.seconds_per_car < 0.1

    def test_prediction_cost_needs_cars(self, evaluator: KnnEvaluator, splits: DataSplits) -> None:
        with pytest.raises(ValueError, match="at least one"):
            prediction_cost(evaluator, splits.test.X.head(0))


@pytest.fixture(scope="module")
def study(dataset: CarDataset, spec: KnnSpec) -> StabilityStudy:
    return stability_study(dataset, spec, SplitFractions(), [Seed(42), Seed(43), Seed(44)])


class TestStability:
    def test_one_entry_per_seed(self, study: StabilityStudy) -> None:
        assert study.seeds == (42, 43, 44)
        assert study.macro_f1.shape == study.false_acceptance.shape == (3,)

    def test_first_seed_reproduces_the_main_split(
        self, study: StabilityStudy, evaluator: KnnEvaluator, splits: DataSplits
    ) -> None:
        from knn_eval_core.metrics import Scores

        assert study.macro_f1[0] == pytest.approx(
            Scores.of(splits.test.y, evaluator.predict(splits.test.X)).macro_f1
        )

    def test_scores_are_sensible(self, study: StabilityStudy) -> None:
        assert (study.macro_f1 > 0.8).all()
        assert (study.false_acceptance < 0.05).all()

    def test_requires_seeds(self, dataset: CarDataset, spec: KnnSpec) -> None:
        with pytest.raises(ValueError, match="at least one seed"):
            stability_study(dataset, spec, SplitFractions(), [])
