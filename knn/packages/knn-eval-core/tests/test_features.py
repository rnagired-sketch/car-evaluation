from __future__ import annotations

import pandas as pd
import pytest

from knn_eval_core.features import (
    CONCEPTS,
    FEATURE_BY_NAME,
    FEATURE_NAMES,
    FEATURES,
    ConceptFeature,
    InputSet,
    OrdinalFeature,
)


class TestOrdinalFeature:
    def test_ranks_follow_level_order(self) -> None:
        assert FEATURE_BY_NAME["safety"].ranks == {"low": 0, "med": 1, "high": 2}

    def test_dataset_attributes_in_column_order(self) -> None:
        assert FEATURE_NAMES == ("buying", "maint", "doors", "persons", "lug_boot", "safety")
        assert [len(f.levels) for f in FEATURES] == [4, 4, 4, 3, 3, 3]

    @pytest.mark.parametrize("levels", [(), ("only",), ("a", "a")])
    def test_rejects_bad_levels(self, levels: tuple[str, ...]) -> None:
        with pytest.raises(ValueError, match=r"two levels|duplicate"):
            OrdinalFeature("x", levels)


class TestConcepts:
    def test_definitions(self) -> None:
        assert [(c.name, c.sources, c.max_score) for c in CONCEPTS] == [
            ("price", ("buying", "maint"), 6),
            ("comfort", ("doors", "persons", "lug_boot"), 7),
        ]

    def test_rejects_unknown_or_empty_sources(self) -> None:
        with pytest.raises(ValueError, match="unknown source"):
            ConceptFeature("bad", ("colour",), "")
        with pytest.raises(ValueError, match="at least one"):
            ConceptFeature("bad", (), "")


class TestInputSet:
    def test_values_are_config_names(self) -> None:
        assert [i.value for i in InputSet] == ["raw", "with_concepts"]

    def test_columns(self) -> None:
        assert InputSet.RAW.columns == FEATURE_NAMES
        assert InputSet.WITH_CONCEPTS.columns == (*FEATURE_NAMES, "price", "comfort")

    @pytest.mark.parametrize("inputs", list(InputSet))
    def test_transform_gives_declared_columns_and_never_mutates(self, inputs: InputSet) -> None:
        X = pd.DataFrame([[1, 2, 3, 1, 2, 0]], columns=list(FEATURE_NAMES))
        before = X.copy()
        out = inputs.transform(X)
        assert tuple(out.columns) == inputs.columns
        assert out is not X
        pd.testing.assert_frame_equal(X, before)

    def test_concept_scores_are_rank_sums(self) -> None:
        X = pd.DataFrame([[1, 2, 3, 1, 2, 0]], columns=list(FEATURE_NAMES))
        out = InputSet.WITH_CONCEPTS.transform(X)
        assert (out.loc[0, "price"], out.loc[0, "comfort"]) == (3, 6)

    def test_descriptions_differ(self) -> None:
        assert len({i.description for i in InputSet}) == 2
