from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from car_eval_core.dataset import COLUMN_NAMES, CarDataset, DataSplits, Partition
from car_eval_core.domain import Acceptability, Car, CcpAlpha, Seed
from car_eval_core.evaluator import CarEvaluator, Explanation
from car_eval_core.features import CONCEPT_BY_NAME, FEATURE_NAMES, FeatureSet, LevelCondition, ScoreCondition
from car_eval_core.tree import TreeHyperparameters, TreeSpec

SEED = Seed(42)
CHEAP_SAFE_ROOMY = Car("low", "low", "4", "4", "big", "high")
TWO_SEATER = Car("low", "low", "4", "2", "big", "high")
UNSAFE = Car("low", "med", "5more", "more", "med", "low")


def holds(condition: LevelCondition | ScoreCondition, ranks: pd.Series) -> bool:
    match condition:
        case LevelCondition(feature):
            return condition.holds_for(int(ranks[feature.name]))
        case ScoreCondition(concept):
            return condition.holds_for(int(sum(ranks[s] for s in concept.sources)))


class TestFit:
    def test_exposes_spec_seed_and_size(
        self, evaluator: CarEvaluator, concept_spec: TreeSpec, splits: DataSplits
    ) -> None:
        assert evaluator.spec == concept_spec
        assert evaluator.seed == SEED
        assert evaluator.training_size == len(splits.train)

    def test_predict_returns_class_codes(self, evaluator: CarEvaluator, splits: DataSplits) -> None:
        pred = evaluator.predict(splits.test.X)
        assert pred.shape == (len(splits.test),)
        assert set(np.unique(pred)) <= {int(a) for a in Acceptability}

    def test_rate_matches_predict(self, evaluator: CarEvaluator) -> None:
        assert evaluator.rate(CHEAP_SAFE_ROOMY) == Acceptability(
            int(evaluator.predict(CHEAP_SAFE_ROOMY.encoded())[0])
        )

    @pytest.mark.parametrize(
        ("car", "expected"),
        [
            (CHEAP_SAFE_ROOMY, Acceptability.VGOOD),
            (TWO_SEATER, Acceptability.UNACC),
            (UNSAFE, Acceptability.UNACC),
        ],
    )
    def test_known_expert_rules(self, evaluator: CarEvaluator, car: Car, expected: Acceptability) -> None:
        # Two seats or low safety is always unacceptable in the expert model.
        assert evaluator.rate(car) is expected

    def test_fits_the_training_data_exactly_when_unpruned(
        self, evaluator: CarEvaluator, splits: DataSplits
    ) -> None:
        np.testing.assert_array_equal(evaluator.predict(splits.train.X), splits.train.y.to_numpy())


class TestPurity:
    def test_unpruned_tree_has_pure_leaves(self, evaluator: CarEvaluator) -> None:
        purity = evaluator.purity()
        assert purity.weighted_leaf_gini == pytest.approx(0.0)
        assert purity.pure_leaf_share == 1.0
        assert purity.n_leaves == evaluator.tree.get_n_leaves()
        assert purity.depth == evaluator.tree.get_depth()

    def test_root_gini_matches_class_mix(self, evaluator: CarEvaluator, splits: DataSplits) -> None:
        p = splits.train.y.value_counts(normalize=True).to_numpy()
        assert evaluator.purity().root_gini == pytest.approx(1 - np.sum(p**2))

    def test_pruned_tree_is_less_pure(self, concept_spec: TreeSpec, splits: DataSplits) -> None:
        pruned = CarEvaluator.fit(concept_spec.with_alpha(CcpAlpha(0.02)), splits.train, SEED).purity()
        assert 0.0 < pruned.weighted_leaf_gini < pruned.root_gini
        assert pruned.pure_leaf_share < 1.0


@pytest.fixture(scope="module")
def every_explanation(evaluator: CarEvaluator, all_cars: list[Car]) -> list[Explanation]:
    return [evaluator.explain(car) for car in all_cars]


class TestExplain:
    def test_path_conditions_hold_for_every_possible_car(self, every_explanation: list[Explanation]) -> None:
        for explanation in every_explanation:
            ranks = explanation.car.encoded().iloc[0]
            assert all(holds(c, ranks) for c in explanation.path), explanation.car

    def test_rating_matches_predict_for_every_possible_car(
        self, evaluator: CarEvaluator, every_explanation: list[Explanation]
    ) -> None:
        X = pd.concat([e.car.encoded() for e in every_explanation], ignore_index=True)
        assert [int(e.rating) for e in every_explanation] == evaluator.predict(X).tolist()

    def test_path_has_one_condition_per_feature(self, evaluator: CarEvaluator, all_cars: list[Car]) -> None:
        for car in all_cars[::37]:
            names = [
                c.feature.name if isinstance(c, LevelCondition) else c.concept.name
                for c in evaluator.explain(car).path
            ]
            assert len(names) == len(set(names))

    def test_path_mentions_concepts_only_for_concept_models(self, splits: DataSplits) -> None:
        raw = CarEvaluator.fit(
            TreeSpec(FeatureSet.RAW, TreeHyperparameters("entropy", None, 2, 1, None)), splits.train, SEED
        )
        assert all(isinstance(c, LevelCondition) for c in raw.explain(CHEAP_SAFE_ROOMY).path)

    def test_low_safety_is_explained_by_safety_alone(self, evaluator: CarEvaluator) -> None:
        explanation = evaluator.explain(UNSAFE)
        assert explanation.rating is Acceptability.UNACC
        assert [c for c in explanation.path if isinstance(c, LevelCondition) and c.feature.name == "safety"]

    def test_leaf_counts_and_confidence(self, evaluator: CarEvaluator) -> None:
        explanation = evaluator.explain(CHEAP_SAFE_ROOMY)
        n = explanation.leaf_support
        assert n == sum(explanation.leaf_counts.values()) > 0
        assert set(explanation.leaf_counts) == set(Acceptability)
        expected = (explanation.leaf_counts[explanation.rating] + 1) / (n + 4)
        assert explanation.confidence == pytest.approx(expected)

    def test_confidence_grows_with_leaf_support(self, evaluator: CarEvaluator, all_cars: list[Car]) -> None:
        explanations = [evaluator.explain(car) for car in all_cars[::11]]
        pure = [e for e in explanations if e.leaf_counts[e.rating] == e.leaf_support]
        small = min(pure, key=lambda e: e.leaf_support)
        large = max(pure, key=lambda e: e.leaf_support)
        assert small.confidence < large.confidence < 1.0

    def test_score_conditions_are_well_formed(self, evaluator: CarEvaluator, all_cars: list[Car]) -> None:
        for car in all_cars[::13]:
            for c in evaluator.explain(car).path:
                if isinstance(c, ScoreCondition):
                    assert 0 <= c.low <= c.high <= CONCEPT_BY_NAME[c.concept.name].max_score


class TestWithNewLabels:
    def test_new_label_overrides_the_old_one(self, evaluator: CarEvaluator, splits: DataSplits) -> None:
        car = Car.from_ranks(splits.train.X.iloc[0])
        old = Acceptability(int(splits.train.y.iloc[0]))
        new = Acceptability.VGOOD if old is Acceptability.UNACC else Acceptability.UNACC
        relabelled, _ = CarDataset.from_frame(
            pd.DataFrame([[getattr(car, f) for f in FEATURE_NAMES] + [new.label]], columns=list(COLUMN_NAMES))
        )
        updated = evaluator.with_new_labels(relabelled)
        assert updated.rate(car) is new
        assert updated.training_size == evaluator.training_size
        assert evaluator.rate(car) is old  # the original evaluator is untouched

    def test_unseen_cars_grow_the_training_set(self, evaluator: CarEvaluator, splits: DataSplits) -> None:
        extra = CarDataset(Partition(splits.test.X.head(10), splits.test.y.head(10)))
        assert evaluator.with_new_labels(extra).training_size == evaluator.training_size + 10


class TestImportances:
    def test_impurity_covers_model_inputs_and_sums_to_one(
        self, evaluator: CarEvaluator, splits: DataSplits
    ) -> None:
        importances = evaluator.importances(splits.test, repeats=2)
        assert tuple(importances.impurity) == FeatureSet.WITH_CONCEPTS.columns
        assert sum(importances.impurity.values()) == pytest.approx(1.0)

    def test_permutation_covers_raw_attributes(self, evaluator: CarEvaluator, splits: DataSplits) -> None:
        drops = evaluator.permutation_importance(splits.test, repeats=3)
        assert tuple(drops) == FEATURE_NAMES
        assert drops["safety"] > 0.2  # shuffling safety wrecks the ratings
        assert drops["safety"] > drops["doors"]

    def test_permutation_is_reproducible(self, evaluator: CarEvaluator, splits: DataSplits) -> None:
        assert evaluator.permutation_importance(splits.test, 2) == evaluator.permutation_importance(
            splits.test, 2
        )

    def test_rejects_zero_repeats(self, evaluator: CarEvaluator, splits: DataSplits) -> None:
        with pytest.raises(ValueError, match="repeats"):
            evaluator.permutation_importance(splits.test, repeats=0)


def test_evaluator_owns_a_copy_of_its_training_data(concept_spec: TreeSpec, splits: DataSplits) -> None:
    train = Partition(splits.train.X.copy(), splits.train.y.copy())
    model = CarEvaluator.fit(concept_spec, train, SEED)
    before = model.explain(CHEAP_SAFE_ROOMY)
    train.y.iloc[:] = int(Acceptability.ACC)  # the caller mutates its own data afterwards
    assert model.explain(CHEAP_SAFE_ROOMY) == before
