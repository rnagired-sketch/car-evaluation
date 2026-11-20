from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from knn_eval_core.domain import Acceptability, Car
from knn_eval_core.evaluator import Explanation, Neighbor
from knn_eval_io import app, plots, report
from knn_eval_io.config import AppConfig
from knn_eval_io.paths import RunPaths

CAR = Car("low", "low", "4", "4", "big", "high")
NEAR = Car("low", "low", "4", "4", "big", "med")


class TestDescriptions:
    def test_car(self) -> None:
        assert (
            report.describe_car(CAR) == "buying=low, maint=low, doors=4, persons=4, lug_boot=big, safety=high"
        )

    def test_explanation_lists_votes_and_neighbors(self) -> None:
        explanation = Explanation(
            car=CAR,
            rating=Acceptability.VGOOD,
            confidence=0.75,
            votes={
                Acceptability.UNACC: 0.0,
                Acceptability.ACC: 0.0,
                Acceptability.GOOD: 0.25,
                Acceptability.VGOOD: 0.75,
            },
            neighbors=(
                Neighbor(CAR, Acceptability.VGOOD, 0.0, 0.75),
                Neighbor(NEAR, Acceptability.GOOD, 0.312, 0.25),
            ),
        )
        text = report.render_explanation(explanation)
        assert "Rating: VGOOD  (confidence 0.75)" in text
        assert "Vote shares: {good: 0.25, vgood: 0.75}" in text
        assert "Nearest 2 training cars:" in text
        assert "2. good   distance 0.312  weight 0.25  buying=low" in text


class TestRunOutputs:
    def test_console_report_has_every_section(
        self, finished_run: tuple[AppConfig, RunPaths, app.RunOutcome, str]
    ) -> None:
        _, _, _, text = finished_run
        for number in ("1.", "2.", "3.", "4.", "5.", "6.", "7."):
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
        assert sorted(p.name for p in paths.figures.glob("*.png")) == [
            "01_class_balance.png",
            "02_distance_metrics.png",
            "03_choosing_k.png",
            "04_learning_curve.png",
            "05_confusion_matrix.png",
            "06_feature_importance.png",
        ]

    def test_results_json_is_strict_and_complete(
        self, finished_run: tuple[AppConfig, RunPaths, app.RunOutcome, str]
    ) -> None:
        _, paths, outcome, _ = finished_run
        data = json.loads(paths.results.read_text(), parse_constant=pytest.fail)
        assert data["validation_report"]["rows_kept"] == 1728
        assert data["split_counts"]["test"]["unacc"] == 363
        assert data["model"] == "k-NN"
        knn = data["knn"]
        assert knn["spec"]["params"]["metric"] in {"euclidean", "cosine"}
        assert set(knn["best_per_metric"]) == {"euclidean", "cosine"}
        assert set(knn["k_scores"]) == {"5", "7"}
        assert knn["validation"]["macro_f1"] > 0.8
        assert data["test_scores"]["macro_f1"] == pytest.approx(outcome.result.test_scores.macro_f1)
        assert data["prediction_cost"]["stored_cars"] == 863
        assert data["stability"]["splits"] == 2
        first = data["explanations"][0]
        assert first["rating"] == outcome.explanations[0].rating.label
        assert len(first["neighbors"]) == outcome.result.search.spec.params.k
        assert set(first["neighbors"][0]) == {"car", "rating", "distance", "weight"}

    def test_report_shows_the_lecture_comparison(
        self, finished_run: tuple[AppConfig, RunPaths, app.RunOutcome, str]
    ) -> None:
        _, _, _, text = finished_run
        assert "Best setting per distance metric" in text
        assert "Choosing K" in text
        assert "prediction cost:" in text
        assert "model: k-nearest neighbors" in text


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
