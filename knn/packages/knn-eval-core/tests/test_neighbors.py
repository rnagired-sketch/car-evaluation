from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler

from knn_eval_core.dataset import DataSplits
from knn_eval_core.domain import Seed
from knn_eval_core.features import InputSet
from knn_eval_core.metrics import CvSettings, Scores
from knn_eval_core.neighbors import (
    DistanceMetric,
    KnnParams,
    KnnSpec,
    Scaling,
    SearchGrid,
    SearchResult,
    Voting,
    inverse_square_weights,
    tune_knn,
)

SEED = Seed(42)


def params(**change: object) -> KnnParams:
    return dataclasses.replace(
        KnnParams(5, DistanceMetric.EUCLIDEAN, Voting.MAJORITY, Scaling.STANDARD), **change
    )


class TestLectureSettings:
    def test_every_lecture_distance_metric_is_available(self) -> None:
        assert [m.value for m in DistanceMetric] == [
            "euclidean",
            "manhattan",
            "minkowski",
            "hamming",
            "cosine",
        ]

    def test_voting_rules(self) -> None:
        assert [v.value for v in Voting] == ["majority", "inverse_distance", "inverse_square"]

    def test_only_hamming_and_cosine_need_brute_force(self) -> None:
        assert {m for m in DistanceMetric if m.needs_brute_force} == {
            DistanceMetric.HAMMING,
            DistanceMetric.COSINE,
        }


class TestInverseSquareWeights:
    def test_weights_are_one_over_d_squared(self) -> None:
        np.testing.assert_allclose(inverse_square_weights(np.array([[1.0, 2.0, 4.0]])), [[1.0, 0.25, 0.0625]])

    def test_exact_match_takes_the_whole_vote(self) -> None:
        weights = inverse_square_weights(np.array([[0.0, 0.5, 1.0], [0.5, 1.0, 2.0]]))
        np.testing.assert_allclose(weights[0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(weights[1], [4.0, 1.0, 0.25])

    def test_closer_neighbors_count_more(self) -> None:
        weights = inverse_square_weights(np.array([[0.3, 0.6, 0.9]]))[0]
        assert weights[0] > weights[1] > weights[2]


class TestKnnParams:
    @pytest.mark.parametrize(
        ("change", "message"), [({"k": 0}, "k must"), ({"minkowski_p": 0.5}, "minkowski_p")]
    )
    def test_rejects_invalid(self, change: dict[str, object], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            params(**change)

    @pytest.mark.parametrize(
        ("scaling", "kind"),
        [
            (Scaling.STANDARD, StandardScaler),
            (Scaling.MINMAX, MinMaxScaler),
            (Scaling.NONE, FunctionTransformer),
        ],
    )
    def test_scaling_step(self, scaling: Scaling, kind: type) -> None:
        assert isinstance(params(scaling=scaling).build().named_steps["scale"], kind)

    @pytest.mark.parametrize("metric", list(DistanceMetric))
    def test_classifier_settings(self, metric: DistanceMetric) -> None:
        knn = params(k=9, metric=metric, minkowski_p=3.0).build().named_steps["knn"]
        assert knn.n_neighbors == 9
        assert knn.metric == metric.value
        assert knn.p == (3.0 if metric is DistanceMetric.MINKOWSKI else 2)
        assert knn.algorithm == ("brute" if metric.needs_brute_force else "auto")

    def test_voting_maps_to_weights(self) -> None:
        assert params(voting=Voting.MAJORITY).build().named_steps["knn"].weights == "uniform"
        assert params(voting=Voting.INVERSE_DISTANCE).build().named_steps["knn"].weights == "distance"
        assert (
            params(voting=Voting.INVERSE_SQUARE).build().named_steps["knn"].weights is inverse_square_weights
        )

    def test_describe(self) -> None:
        text = params(metric=DistanceMetric.MINKOWSKI, minkowski_p=3.0).describe()
        assert text == "k=5, minkowski (p=3) distance, majority voting, standard scaling"


class TestKnnSpec:
    @pytest.mark.parametrize("inputs", list(InputSet))
    def test_fit_uses_the_input_columns(self, inputs: InputSet, splits: DataSplits) -> None:
        spec = KnnSpec(inputs, params())
        model = spec.fit(splits.train)
        assert tuple(model.feature_names_in_) == inputs.columns  # pyright: ignore[reportAttributeAccessIssue]
        assert spec.predict(model, splits.test.X).shape == (len(splits.test),)

    def test_is_deterministic(self, splits: DataSplits) -> None:
        spec = KnnSpec(InputSet.WITH_CONCEPTS, params(metric=DistanceMetric.COSINE))
        a = spec.predict(spec.fit(splits.train), splits.test.X)
        b = spec.predict(spec.fit(splits.train), splits.test.X)
        np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("metric", list(DistanceMetric))
    @pytest.mark.parametrize("voting", list(Voting))
    def test_every_combination_beats_a_majority_vote(
        self, metric: DistanceMetric, voting: Voting, splits: DataSplits
    ) -> None:
        spec = KnnSpec(InputSet.WITH_CONCEPTS, params(metric=metric, voting=voting))
        scores = Scores.of(splits.validation.y, spec.predict(spec.fit(splits.train), splits.validation.X))
        assert scores.macro_f1 > 0.4  # always predicting unacc scores 0.21


class TestSearchGrid:
    def test_default_covers_the_lecture_settings(self) -> None:
        grid = SearchGrid()
        assert grid.metric == tuple(DistanceMetric)
        assert grid.voting == tuple(Voting)
        assert grid.size == 8 * 5 * 3 * 1 == len(grid.candidates())

    def test_candidates_are_unique(self) -> None:
        assert len(set(SearchGrid().candidates())) == SearchGrid().size

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"k": ()}, "empty"),
            ({"metric": ()}, "empty"),
            ({"k": (0,)}, "k values"),
            ({"minkowski_p": 0.0}, "minkowski_p"),
        ],
    )
    def test_rejects_invalid(self, change: dict[str, object], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(SearchGrid(), **change)


@pytest.fixture(scope="module")
def search(splits: DataSplits, tiny_grid: SearchGrid, fast_cv: CvSettings) -> SearchResult:
    return tune_knn(InputSet.WITH_CONCEPTS, splits.train, tiny_grid, fast_cv, SEED)


class TestTuneKnn:
    def test_best_leads_the_ranking(self, search: SearchResult) -> None:
        means = [c.macro_f1_mean for c in search.top_candidates]
        assert means == sorted(means, reverse=True)
        assert search.top_candidates[0].params == search.spec.params
        assert search.cv_macro_f1_mean == means[0]

    def test_best_comes_from_the_grid(self, search: SearchResult, tiny_grid: SearchGrid) -> None:
        assert search.spec.params in tiny_grid.candidates()
        assert search.spec.inputs is InputSet.WITH_CONCEPTS

    def test_reports_the_best_of_every_metric(self, search: SearchResult, tiny_grid: SearchGrid) -> None:
        assert set(search.best_per_metric) == set(tiny_grid.metric)
        for metric, score in search.best_per_metric.items():
            assert score.params.metric is metric
        assert search.best_per_metric[search.spec.params.metric].params == search.spec.params

    def test_k_scores_vary_only_k(self, search: SearchResult, tiny_grid: SearchGrid) -> None:
        assert list(search.k_scores) == sorted(tiny_grid.k)
        assert search.k_scores[search.spec.params.k] == pytest.approx(search.cv_macro_f1_mean)

    def test_scores_are_probabilities(self, search: SearchResult) -> None:
        for c in search.top_candidates:
            assert 0 <= c.macro_f1_mean <= 1
            assert 0 <= c.accuracy_mean <= 1
            assert c.macro_f1_std >= 0

    def test_ties_go_to_the_smaller_k(self, splits: DataSplits, fast_cv: CvSettings) -> None:
        # With an exact-match-wins vote and k larger than any useful neighborhood, two K values
        # can tie; the simpler (smaller) K must win.
        grid = SearchGrid(k=(9, 3), metric=(DistanceMetric.EUCLIDEAN,), voting=(Voting.INVERSE_DISTANCE,))
        result = tune_knn(InputSet.WITH_CONCEPTS, splits.train, grid, fast_cv, SEED)
        if result.k_scores[3] == pytest.approx(result.k_scores[9], abs=1e-12):
            assert result.spec.params.k == 3
        else:
            assert result.spec.params.k == max(result.k_scores, key=lambda k: result.k_scores[k])

    def test_logs_the_metric_comparison(
        self, splits: DataSplits, tiny_grid: SearchGrid, fast_cv: CvSettings, record_logs: object
    ) -> None:
        with record_logs() as logs:  # type: ignore[operator]
            tune_knn(InputSet.RAW, splits.train, tiny_grid, fast_cv, SEED)
        messages = logs.messages()
        assert messages[0] == "grid search on 6 raw attributes: 4 candidates x 3 folds = 12 fits"
        assert sum("best CV macro-F1" in m for m in messages) == 1 + len(tiny_grid.metric)
