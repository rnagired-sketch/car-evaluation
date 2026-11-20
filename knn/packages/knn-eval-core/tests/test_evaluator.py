from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from knn_eval_core.dataset import COLUMN_NAMES, CarDataset, DataSplits, Partition
from knn_eval_core.domain import Acceptability, Car, Seed
from knn_eval_core.evaluator import Explanation, KnnEvaluator
from knn_eval_core.features import FEATURE_NAMES, InputSet
from knn_eval_core.neighbors import DistanceMetric, KnnParams, KnnSpec, Scaling, Voting

SEED = Seed(42)
CHEAP_SAFE_ROOMY = Car("low", "low", "4", "4", "big", "high")
UNSAFE = Car("low", "med", "5more", "more", "med", "low")


def fitted(splits: DataSplits, **change: object) -> KnnEvaluator:
    base = {
        "k": 7,
        "metric": DistanceMetric.EUCLIDEAN,
        "voting": Voting.MAJORITY,
        "scaling": Scaling.STANDARD,
    }
    base.update(change)
    return KnnEvaluator.fit(KnnSpec(InputSet.WITH_CONCEPTS, KnnParams(**base)), splits.train, SEED)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def sample_explanations(evaluator: KnnEvaluator, all_cars: list[Car]) -> list[Explanation]:
    return [evaluator.explain(car) for car in all_cars[::9]]


class TestFit:
    def test_exposes_spec_seed_and_size(self, evaluator: KnnEvaluator, splits: DataSplits) -> None:
        assert evaluator.training_size == len(splits.train)
        assert evaluator.seed == SEED
        assert evaluator.spec.params.k == 7

    def test_predictions_are_class_codes(self, evaluator: KnnEvaluator, splits: DataSplits) -> None:
        pred = evaluator.predict(splits.test.X)
        assert pred.shape == (len(splits.test),)
        assert set(np.unique(pred)) <= {int(a) for a in Acceptability}

    def test_rate_matches_predict(self, evaluator: KnnEvaluator) -> None:
        assert evaluator.rate(UNSAFE) == Acceptability(int(evaluator.predict(UNSAFE.encoded())[0]))

    def test_distance_voting_recalls_every_training_car(
        self, evaluator: KnnEvaluator, splits: DataSplits
    ) -> None:
        # Every training car is its own nearest neighbor at distance 0, which wins the 1/d vote.
        np.testing.assert_array_equal(evaluator.predict(splits.train.X), splits.train.y.to_numpy())

    def test_unsafe_car_is_unacceptable(self, evaluator: KnnEvaluator) -> None:
        assert evaluator.rate(UNSAFE) is Acceptability.UNACC


class TestExplain:
    def test_rating_matches_predict(
        self, evaluator: KnnEvaluator, sample_explanations: list[Explanation]
    ) -> None:
        X = pd.concat([e.car.encoded() for e in sample_explanations], ignore_index=True)
        assert [int(e.rating) for e in sample_explanations] == evaluator.predict(X).tolist()

    def test_k_neighbors_nearest_first(self, sample_explanations: list[Explanation]) -> None:
        for e in sample_explanations:
            assert len(e.neighbors) == 7
            distances = [n.distance for n in e.neighbors]
            assert distances == sorted(distances)
            assert all(d >= 0 for d in distances)

    def test_votes_and_weights_are_shares(self, sample_explanations: list[Explanation]) -> None:
        for e in sample_explanations:
            assert sum(e.votes.values()) == pytest.approx(1.0)
            assert sum(n.weight for n in e.neighbors) == pytest.approx(1.0)
            assert set(e.votes) == set(Acceptability)

    def test_vote_shares_are_the_neighbors_weights_per_class(
        self, sample_explanations: list[Explanation]
    ) -> None:
        for e in sample_explanations:
            for rating, share in e.votes.items():
                assert share == pytest.approx(
                    sum(n.weight for n in e.neighbors if n.rating is rating), abs=1e-9
                )

    def test_rating_is_the_largest_vote_and_confidence_its_share(
        self, sample_explanations: list[Explanation]
    ) -> None:
        for e in sample_explanations:
            assert e.confidence == pytest.approx(max(e.votes.values()))
            assert e.votes[e.rating] == pytest.approx(e.confidence)

    def test_majority_voting_gives_equal_weights(self, splits: DataSplits) -> None:
        explanation = fitted(splits, voting=Voting.MAJORITY).explain(UNSAFE)
        assert {round(n.weight, 9) for n in explanation.neighbors} == {round(1 / 7, 9)}
        assert explanation.confidence * 7 == pytest.approx(round(explanation.confidence * 7))

    @pytest.mark.parametrize("voting", [Voting.INVERSE_DISTANCE, Voting.INVERSE_SQUARE])
    def test_an_exact_training_match_decides_alone(self, splits: DataSplits, voting: Voting) -> None:
        car = Car.from_ranks(splits.train.X.iloc[0])
        explanation = fitted(splits, voting=voting).explain(car)
        assert explanation.neighbors[0].distance == 0.0
        assert explanation.neighbors[0].weight == pytest.approx(1.0)
        assert explanation.confidence == pytest.approx(1.0)
        assert explanation.rating is Acceptability(int(splits.train.y.iloc[0]))

    def test_inverse_square_favours_close_neighbors_more_than_inverse_distance(
        self, splits: DataSplits
    ) -> None:
        car = Car.from_ranks(splits.test.X.iloc[0])  # not stored, so no exact match
        linear = fitted(splits, voting=Voting.INVERSE_DISTANCE).explain(car).neighbors
        square = fitted(splits, voting=Voting.INVERSE_SQUARE).explain(car).neighbors
        assert linear[0].distance > 0
        assert square[0].weight > linear[0].weight

    def test_neighbors_are_training_cars(self, evaluator: KnnEvaluator, splits: DataSplits) -> None:
        stored = {Car.from_ranks(row) for _, row in splits.train.X.iterrows()}
        assert all(n.car in stored for n in evaluator.explain(CHEAP_SAFE_ROOMY).neighbors)


class TestUpdates:
    def test_new_label_overrides_the_old_one(self, evaluator: KnnEvaluator, splits: DataSplits) -> None:
        car = Car.from_ranks(splits.train.X.iloc[0])
        old = Acceptability(int(splits.train.y.iloc[0]))
        new = Acceptability.VGOOD if old is Acceptability.UNACC else Acceptability.UNACC
        frame = pd.DataFrame(
            [[getattr(car, f) for f in FEATURE_NAMES] + [new.label]], columns=list(COLUMN_NAMES)
        )
        updated = evaluator.with_new_labels(CarDataset.from_frame(frame)[0])
        assert updated.rate(car) is new
        assert updated.training_size == evaluator.training_size
        assert evaluator.rate(car) is old

    def test_unseen_cars_grow_the_store(self, evaluator: KnnEvaluator, splits: DataSplits) -> None:
        extra = CarDataset(Partition(splits.test.X.head(10), splits.test.y.head(10)))
        assert evaluator.with_new_labels(extra).training_size == evaluator.training_size + 10

    def test_owns_a_copy_of_its_training_data(self, spec: KnnSpec, splits: DataSplits) -> None:
        train = Partition(splits.train.X.copy(), splits.train.y.copy())
        model = KnnEvaluator.fit(spec, train, SEED)
        before = model.explain(CHEAP_SAFE_ROOMY)
        train.y.iloc[:] = int(Acceptability.ACC)
        assert model.explain(CHEAP_SAFE_ROOMY) == before


class TestImportance:
    def test_covers_raw_attributes_and_safety_matters(
        self, evaluator: KnnEvaluator, splits: DataSplits
    ) -> None:
        drops = evaluator.permutation_importance(splits.test, repeats=3)
        assert tuple(drops) == FEATURE_NAMES
        assert drops["safety"] > drops["doors"]

    def test_reproducible_and_validated(self, evaluator: KnnEvaluator, splits: DataSplits) -> None:
        assert evaluator.permutation_importance(splits.test, 2) == evaluator.permutation_importance(
            splits.test, 2
        )
        with pytest.raises(ValueError, match="repeats"):
            evaluator.permutation_importance(splits.test, 0)
