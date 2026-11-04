from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from car_eval_core.domain import Acceptability, Car
from car_eval_core.evaluator import Explanation
from car_eval_core.features import CONCEPT_BY_NAME, FEATURE_BY_NAME, LevelCondition, ScoreCondition
from car_eval_io import app, plots, report
from car_eval_io.config import AppConfig
from car_eval_io.paths import RunPaths

CAR = Car("low", "low", "4", "4", "big", "high")


class TestDescriptions:
    def test_level_condition(self) -> None:
        condition = LevelCondition(FEATURE_BY_NAME["persons"], frozenset({1, 2}))
        assert report.describe_condition(condition) == "persons in {4, more}"

    def test_score_condition_range_and_single_value(self) -> None:
        price = CONCEPT_BY_NAME["price"]
        assert report.describe_condition(ScoreCondition(price, 0, 2)) == (
            "price score (buying + maint) in 0..2 of 0..6 (0 = cheapest)"
        )
        assert "in 3 of" in report.describe_condition(ScoreCondition(price, 3, 3))

    def test_car(self) -> None:
        assert (
            report.describe_car(CAR) == "buying=low, maint=low, doors=4, persons=4, lug_boot=big, safety=high"
        )

    def test_explanation(self) -> None:
        explanation = Explanation(
            car=CAR,
            rating=Acceptability.VGOOD,
            confidence=0.857,
            leaf_counts={a: (17 if a is Acceptability.VGOOD else 0) for a in Acceptability},
            path=(LevelCondition(FEATURE_BY_NAME["safety"], frozenset({2})),),
        )
        text = report.render_explanation(explanation)
        assert "Rating: VGOOD  (confidence 0.86)" in text
        assert "17 training cars -> {vgood: 17}" in text
        assert "- safety in {high}" in text

    def test_explanation_of_a_single_leaf_tree(self) -> None:
        counts = dict.fromkeys(Acceptability, 0)
        explanation = Explanation(CAR, Acceptability.UNACC, 0.5, counts, ())
        assert "(root is a leaf)" in report.render_explanation(explanation)


class TestRunOutputs:
    def test_console_report_has_every_section(
        self, finished_run: tuple[AppConfig, RunPaths, app.RunOutcome, str]
    ) -> None:
        _, _, _, text = finished_run
        for number in ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8."):
            assert f"\n{number} " in text
        assert text.startswith(f"\n{report.RULE}\n1. ")  # the report alone: progress goes to logging
        assert "\nprofile: default\n" in text

    def test_requested_cars_are_explained(
        self, finished_run: tuple[AppConfig, RunPaths, app.RunOutcome, str]
    ) -> None:
        config, _, outcome, _ = finished_run
        assert [e.car for e in outcome.explanations] == list(config.cars)
        assert outcome.explanations[1].rating is Acceptability.UNACC  # two seats

    def test_files_written(self, finished_run: tuple[AppConfig, RunPaths, app.RunOutcome, str]) -> None:
        _, paths, outcome, _ = finished_run
        assert {paths.report, paths.results, paths.model} <= set(outcome.files)
        assert all(p.is_file() for p in outcome.files)
        assert all(p.parent in (paths.output, paths.figures) for p in outcome.files)
        assert len(list(paths.figures.glob("*.png"))) == 7

    def test_results_json_is_strict_and_complete(
        self, finished_run: tuple[AppConfig, RunPaths, app.RunOutcome, str]
    ) -> None:
        _, paths, outcome, _ = finished_run
        data = json.loads(paths.results.read_text(), parse_constant=pytest.fail)
        assert data["validation_report"]["rows_kept"] == 1728
        assert data["split_counts"]["test"]["unacc"] == 363
        assert data["tree"]["spec"]["params"]["criterion"] == "entropy"
        assert data["tree"]["validation"]["macro_f1"] >= 0.97
        assert data["stability"]["macro_f1"]["mean"] >= 0.9
        assert data["test_scores"]["macro_f1"] == pytest.approx(outcome.result.test_scores.macro_f1)
        assert data["stability"]["splits"] == 2
        assert data["explanations"][0]["rating"] == outcome.explanations[0].rating.label
        first_step = data["explanations"][0]["path"][0]
        assert set(first_step) in ({"feature", "levels"}, {"concept", "sources", "low", "high", "max"})


class TestFigureSink:
    def test_saves_and_closes(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        sink = plots.FigureSink(tmp_path, show=False)
        with sink.figure("x") as fig:
            fig.subplots().plot([0, 1], [0, 1])
        assert sink.saved == [tmp_path / "x.png"]
        assert (tmp_path / "x.png").stat().st_size > 0
        assert plt.get_fignums() == []

    def test_nothing_saved_without_directory(self) -> None:
        sink = plots.FigureSink(None, show=False)
        with sink.figure("x") as fig:
            fig.subplots()
        assert sink.saved == []

    def test_failed_drawing_is_not_saved_but_is_freed(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        sink = plots.FigureSink(tmp_path, show=False)
        with pytest.raises(RuntimeError), sink.figure("broken"):
            raise RuntimeError("boom")
        assert not (tmp_path / "broken.png").exists()
        assert plt.get_fignums() == []

    def test_confusion_handles_empty_rows(self, tmp_path: Path) -> None:
        sink = plots.FigureSink(tmp_path, show=False)
        plots.confusion(sink, np.array([[5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]]))
        assert sink.saved == [tmp_path / "05_confusion_matrix.png"]
