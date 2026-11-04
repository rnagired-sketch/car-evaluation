from __future__ import annotations

import pandas as pd
import pytest

from car_eval_core.features import (
    CONCEPT_BY_NAME,
    CONCEPTS,
    FEATURE_BY_NAME,
    FEATURE_NAMES,
    FEATURES,
    MODEL_COLUMNS,
    ConceptFeature,
    LevelCondition,
    OrdinalFeature,
    ScoreCondition,
    model_inputs,
    path_condition,
    tighten,
)

SAFETY = FEATURE_BY_NAME["safety"]
PRICE = CONCEPT_BY_NAME["price"]
COMFORT = CONCEPT_BY_NAME["comfort"]


class TestOrdinalFeature:
    def test_ranks_follow_level_order(self) -> None:
        assert SAFETY.ranks == {"low": 0, "med": 1, "high": 2}
        assert SAFETY.max_rank == 2

    def test_dataset_attributes_in_column_order(self) -> None:
        assert FEATURE_NAMES == ("buying", "maint", "doors", "persons", "lug_boot", "safety")
        assert [len(f.levels) for f in FEATURES] == [4, 4, 4, 3, 3, 3]

    @pytest.mark.parametrize("levels", [(), ("only",)])
    def test_rejects_fewer_than_two_levels(self, levels: tuple[str, ...]) -> None:
        with pytest.raises(ValueError, match="at least two"):
            OrdinalFeature("x", levels)

    def test_rejects_duplicate_levels(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            OrdinalFeature("x", ("a", "b", "a"))

    @pytest.mark.parametrize("feature", FEATURES, ids=lambda f: f.name)
    def test_every_cart_threshold_partitions_the_levels(self, feature: OrdinalFeature) -> None:
        everything = frozenset(range(len(feature.levels)))
        for cut in range(feature.max_rank):
            threshold = cut + 0.5
            left = feature.condition(threshold, goes_left=True)
            right = feature.condition(threshold, goes_left=False)
            assert left.admitted == frozenset(range(cut + 1))
            assert left.admitted | right.admitted == everything
            assert not left.admitted & right.admitted

    def test_condition_names_admitted_levels(self) -> None:
        assert SAFETY.condition(0.5, goes_left=False).levels == ("med", "high")
        assert SAFETY.condition(1.5, goes_left=True).levels == ("low", "med")


class TestConceptFeature:
    def test_definitions_follow_the_expert_hierarchy(self) -> None:
        assert PRICE.sources == ("buying", "maint")
        assert COMFORT.sources == ("doors", "persons", "lug_boot")
        assert [c.name for c in CONCEPTS] == ["price", "comfort"]

    def test_max_score_is_sum_of_max_ranks(self) -> None:
        assert PRICE.max_score == 3 + 3
        assert COMFORT.max_score == 3 + 2 + 2

    def test_score_sums_source_ranks(self) -> None:
        X = pd.DataFrame({"buying": [0, 3, 1], "maint": [0, 3, 2], "other": [9, 9, 9]})
        assert PRICE.score(X).tolist() == [0, 6, 3]
        assert PRICE.score(X).name == "price"

    def test_rejects_unknown_sources(self) -> None:
        with pytest.raises(ValueError, match="unknown source"):
            ConceptFeature("bad", ("buying", "colour"), "")

    def test_rejects_empty_sources(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            ConceptFeature("bad", (), "")

    @pytest.mark.parametrize(("threshold", "left", "right"), [(0.5, (0, 0), (1, 6)), (2.5, (0, 2), (3, 6))])
    def test_condition_splits_integer_scores(
        self, threshold: float, left: tuple[int, int], right: tuple[int, int]
    ) -> None:
        assert PRICE.condition(threshold, goes_left=True) == ScoreCondition(PRICE, *left)
        assert PRICE.condition(threshold, goes_left=False) == ScoreCondition(PRICE, *right)


class TestConditions:
    def test_level_condition_holds_for_admitted_ranks_only(self) -> None:
        condition = LevelCondition(SAFETY, frozenset({1, 2}))
        assert [condition.holds_for(r) for r in range(3)] == [False, True, True]

    def test_score_condition_bounds_are_inclusive(self) -> None:
        condition = ScoreCondition(PRICE, 2, 4)
        assert [condition.holds_for(s) for s in range(7)] == [False, False, True, True, True, False, False]

    def test_tighten_intersects_levels(self) -> None:
        merged = tighten(LevelCondition(SAFETY, frozenset({1, 2})), LevelCondition(SAFETY, frozenset({2})))
        assert merged == LevelCondition(SAFETY, frozenset({2}))

    def test_tighten_narrows_score_ranges(self) -> None:
        merged = tighten(ScoreCondition(PRICE, 0, 4), ScoreCondition(PRICE, 2, 6))
        assert merged == ScoreCondition(PRICE, 2, 4)

    def test_tighten_rejects_mixed_condition_kinds(self) -> None:
        with pytest.raises(TypeError, match="cannot combine"):
            tighten(LevelCondition(SAFETY, frozenset({0})), ScoreCondition(PRICE, 0, 1))

    def test_tighten_rejects_different_features(self) -> None:
        persons = FEATURE_BY_NAME["persons"]
        with pytest.raises(TypeError, match="cannot combine"):
            tighten(LevelCondition(SAFETY, frozenset({0})), LevelCondition(persons, frozenset({0})))
        with pytest.raises(TypeError, match="cannot combine"):
            tighten(ScoreCondition(PRICE, 0, 1), ScoreCondition(COMFORT, 0, 1))


class TestModelInputs:
    def test_columns_are_raw_attributes_then_concepts(self) -> None:
        assert (*FEATURE_NAMES, "price", "comfort") == MODEL_COLUMNS

    def test_adds_concept_scores_without_mutating_input(self) -> None:
        X = pd.DataFrame([[0, 1, 2, 1, 2, 0]], columns=list(FEATURE_NAMES))
        before = X.copy()
        out = model_inputs(X)
        assert tuple(out.columns) == MODEL_COLUMNS
        pd.testing.assert_frame_equal(X, before)

    def test_concept_columns_hold_rank_sums(self) -> None:
        X = pd.DataFrame([[1, 2, 3, 1, 2, 0]], columns=list(FEATURE_NAMES))
        out = model_inputs(X)
        assert out.loc[0, "price"] == 3
        assert out.loc[0, "comfort"] == 6

    def test_path_condition_dispatches_on_column(self) -> None:
        assert isinstance(path_condition("safety", 0.5, True), LevelCondition)
        assert isinstance(path_condition("price", 0.5, True), ScoreCondition)

    def test_path_condition_rejects_unknown_columns(self) -> None:
        with pytest.raises(KeyError, match="not a model input"):
            path_condition("colour", 0.5, True)
