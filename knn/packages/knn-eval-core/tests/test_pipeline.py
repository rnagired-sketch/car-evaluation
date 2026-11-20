from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from knn_eval_core.dataset import CarDataset
from knn_eval_core.domain import Acceptability
from knn_eval_core.features import InputSet
from knn_eval_core.metrics import CvSettings, Scores
from knn_eval_core.neighbors import SearchGrid
from knn_eval_core.pipeline import PipelineResult, PipelineSettings, run_pipeline

STAGE = re.compile(r"^\[(\d+)/(\d+)\] (.+?)( \.\.\.| done in \d+\.\d s)$")


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


class TestSettings:
    def test_defaults_match_the_decision_tree(self) -> None:
        settings = PipelineSettings()
        assert settings.seed == 42
        assert (settings.fractions.train, settings.fractions.validation, settings.fractions.test) == (
            0.5,
            0.2,
            0.3,
        )
        assert settings.inputs is InputSet.WITH_CONCEPTS
        assert settings.stability_seeds == 20

    @pytest.mark.parametrize(
        ("change", "message"),
        [({"stability_seeds": -1}, "stability_seeds"), ({"permutation_repeats": 0}, "permutation")],
    )
    def test_rejects_invalid(self, change: dict[str, object], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(PipelineSettings(), **change)

    @pytest.mark.parametrize(("change", "stages"), [({}, 5), ({"stability_seeds": 0}, 4)])
    def test_stage_count(self, change: dict[str, object], stages: int) -> None:
        assert dataclasses.replace(PipelineSettings(), **change).stage_count() == stages


@pytest.mark.slow
class TestRunPipeline:
    def test_logs_every_stage_in_order(
        self, run: tuple[PipelineResult, list[str]], settings: PipelineSettings
    ) -> None:
        _, messages = run
        stages = [m.groups() for message in messages if (m := STAGE.match(message))]
        total = settings.stage_count()
        assert [int(n) for n, *_ in stages] == [n for n in range(1, total + 1) for _ in (0, 1)]
        assert messages[0].startswith("starting the k-NN evaluation")
        assert messages[-1].startswith("k-NN evaluation finished in")
        assert any(m.startswith("prediction cost: ") for m in messages)

    def test_scores_are_consistent(self, run: tuple[PipelineResult, list[str]]) -> None:
        result, _ = run
        test = result.splits.test
        np.testing.assert_array_equal(result.test_predictions, result.evaluator.predict(test.X))
        assert result.test_scores == Scores.of(test.y, result.test_predictions)
        assert result.confusion.sum() == len(test)
        assert result.evaluator.spec == result.search.spec
        assert result.validation_scores == Scores.of(
            result.splits.validation.y, result.evaluator.predict(result.splits.validation.X)
        )

    def test_misclassified_are_exactly_the_wrong_predictions(
        self, run: tuple[PipelineResult, list[str]]
    ) -> None:
        result, _ = run
        wrong = int((result.test_predictions != result.splits.test.y.to_numpy()).sum())
        assert len(result.misclassified) == wrong > 0
        for item in result.misclassified:
            assert item.truth is not item.explanation.rating
            assert isinstance(item.truth, Acceptability)

    def test_diagnostics_and_stability(self, run: tuple[PipelineResult, list[str]]) -> None:
        result, _ = run
        assert result.search.spec.params.k in result.k_curve.x
        assert result.cost.stored_cars == len(result.splits.train)
        assert result.stability is not None
        assert len(result.stability.seeds) == 2

    def test_stability_can_be_switched_off(self, dataset: CarDataset, settings: PipelineSettings) -> None:
        assert run_pipeline(dataset, dataclasses.replace(settings, stability_seeds=0)).stability is None
