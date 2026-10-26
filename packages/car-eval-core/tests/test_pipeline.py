from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from car_eval_core.dataset import CarDataset
from car_eval_core.domain import Acceptability
from car_eval_core.features import FeatureSet
from car_eval_core.metrics import CvSettings, Scores
from car_eval_core.pipeline import PipelineResult, PipelineSettings, run_pipeline
from car_eval_core.tree import SearchGrid


@pytest.fixture(scope="module")
def settings(tiny_grid: SearchGrid, fast_cv: CvSettings) -> PipelineSettings:
    return PipelineSettings(grid=tiny_grid, cv=fast_cv, stability_seeds=2, permutation_repeats=2)


@pytest.fixture(scope="module")
def run(
    dataset: CarDataset, settings: PipelineSettings, record_logs: Callable[..., Any]
) -> tuple[PipelineResult, list[str]]:
    with record_logs() as logs:
        result = run_pipeline(dataset, settings)
    return result, logs.messages()


STAGE = re.compile(r"^\[(\d+)/(\d+)\] (.+?)( \.\.\.| done in \d+\.\d s)$")


class TestSettings:
    def test_defaults_follow_the_proposal(self) -> None:
        settings = PipelineSettings()
        assert settings.feature_sets == (FeatureSet.RAW, FeatureSet.WITH_CONCEPTS)
        assert settings.fractions.train == 0.5
        assert settings.compare_benchmarks

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"feature_sets": ()}, "at least one"),
            ({"feature_sets": (FeatureSet.RAW, FeatureSet.RAW)}, "duplicates"),
            ({"stability_seeds": -1}, "stability_seeds"),
            ({"permutation_repeats": 0}, "permutation_repeats"),
        ],
    )
    def test_rejects_invalid(self, change: dict[str, object], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(PipelineSettings(), **change)

    @pytest.mark.parametrize(
        ("change", "stages"),
        [
            ({}, 7),
            ({"compare_benchmarks": False}, 6),
            ({"stability_seeds": 0}, 6),
            ({"feature_sets": (FeatureSet.RAW,), "compare_benchmarks": False, "stability_seeds": 0}, 4),
        ],
    )
    def test_stage_count(self, change: dict[str, object], stages: int) -> None:
        assert dataclasses.replace(PipelineSettings(), **change).stage_count() == stages


@pytest.mark.slow
class TestRunPipeline:
    def test_logs_every_stage_start_and_finish_in_order(
        self, run: tuple[PipelineResult, list[str]], settings: PipelineSettings
    ) -> None:
        _, messages = run
        stages = [match.groups() for m in messages if (match := STAGE.match(m))]
        total = settings.stage_count()
        assert len(stages) == 2 * total
        assert [int(number) for number, *_ in stages] == [n for n in range(1, total + 1) for _ in (0, 1)]
        assert {int(t) for _, t, *_ in stages} == {total}
        assert stages[0][2].startswith("splitting data")
        assert stages[-1][2].startswith("stability study over 2 random splits")

    def test_logs_scores_and_selection(self, run: tuple[PipelineResult, list[str]]) -> None:
        _, messages = run
        assert messages[0].startswith("starting the evaluation plan")
        assert messages[-1].startswith("evaluation plan finished in")
        assert any(m.startswith("selected on validation: ") for m in messages)
        assert any(m.startswith("test macro-F1 ") for m in messages)
        assert any(m.startswith("grid search on ") for m in messages)
        assert any(re.match(r"\s+split 2/2 \(seed 43\)", m) for m in messages)

    def test_develops_one_tree_per_feature_set(self, run: tuple[PipelineResult, list[str]]) -> None:
        result, _ = run
        assert list(result.candidates) == list(FeatureSet)

    def test_chosen_tree_has_best_validation_score(self, run: tuple[PipelineResult, list[str]]) -> None:
        result, _ = run
        best = max(d.validation.macro_f1 for d in result.candidates.values())
        assert result.chosen.validation.macro_f1 == best
        assert result.chosen.spec.features is FeatureSet.WITH_CONCEPTS
        assert result.evaluator.spec == result.chosen.spec

    def test_test_scores_are_consistent(self, run: tuple[PipelineResult, list[str]]) -> None:
        result, _ = run
        test = result.splits.test
        np.testing.assert_array_equal(result.test_predictions, result.evaluator.predict(test.X))
        assert result.test_scores == Scores.of(test.y, result.test_predictions)
        assert result.confusion.sum() == len(test)
        assert np.trace(result.confusion) == int((result.test_predictions == test.y.to_numpy()).sum())

    def test_misclassified_cars_are_exactly_the_wrong_predictions(
        self, run: tuple[PipelineResult, list[str]]
    ) -> None:
        result, _ = run
        wrong = int((result.test_predictions != result.splits.test.y.to_numpy()).sum())
        assert len(result.misclassified) == wrong
        for item in result.misclassified:
            assert item.truth is not item.explanation.rating
            assert isinstance(item.truth, Acceptability)

    def test_benchmarks_and_adequacy(
        self, run: tuple[PipelineResult, list[str]], settings: PipelineSettings
    ) -> None:
        result, _ = run
        assert len(result.comparison) == 2 * (3 + len(settings.feature_sets))
        assert result.adequacy.validation == result.chosen.validation
        assert result.adequacy.passes

    def test_diagnostics_and_stability(self, run: tuple[PipelineResult, list[str]]) -> None:
        result, _ = run
        assert len(result.depth_curve.x) == 15
        assert result.stability is not None
        assert len(result.stability.seeds) == 2
        assert result.purity.n_leaves == result.evaluator.tree.get_n_leaves()

    def test_optional_stages_can_be_switched_off(
        self, dataset: CarDataset, settings: PipelineSettings
    ) -> None:
        lean = dataclasses.replace(
            settings, compare_benchmarks=False, stability_seeds=0, feature_sets=(FeatureSet.RAW,)
        )
        result = run_pipeline(dataset, lean)
        assert result.comparison == ()
        assert result.stability is None
        assert result.chosen.spec.features is FeatureSet.RAW
