from __future__ import annotations

import logging
from pathlib import Path

import pytest

from car_eval_core import pipeline
from car_eval_io import app
from car_eval_io.config import AppConfig
from car_eval_io.paths import RunPaths
from car_eval_io.sources import LocalFile


class TestRunPaths:
    def test_relative_folder_becomes_absolute_in_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        paths = RunPaths(Path("outputs"))
        assert paths.output == tmp_path.resolve() / "outputs"
        assert paths.output.is_absolute()

    def test_home_folder_is_expanded(self) -> None:
        assert RunPaths(Path("~/car-eval-out")).output == Path.home().resolve() / "car-eval-out"

    def test_every_file_lives_in_the_output_folder(self, tmp_path: Path) -> None:
        paths = RunPaths(tmp_path / "out")
        names = {
            paths.report: "report.txt",
            paths.results: "results.json",
            paths.model: "model.json",
            paths.explanations: "explanations.json",
            paths.log: "car-eval.log",
            paths.figures: "figures",
        }
        for path, name in names.items():
            assert path == paths.output / name

    def test_create_makes_nested_folders_and_is_repeatable(self, tmp_path: Path) -> None:
        paths = RunPaths(tmp_path / "a" / "b")
        assert paths.create() == tmp_path.resolve() / "a" / "b"
        assert paths.create().is_dir()

    def test_rejects_a_file(self, tmp_path: Path) -> None:
        file = tmp_path / "taken"
        file.write_text("x")
        with pytest.raises(ValueError, match="not a folder"):
            RunPaths(file)


def test_run_creates_the_folder_before_the_long_work_starts(
    tmp_path: Path, car_data: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The folder appears as soon as `run` starts, and the log names it as an absolute path."""
    paths = RunPaths(tmp_path / "results")

    def stop_here(*_: object, **__: object) -> None:
        assert paths.output.is_dir()
        raise RuntimeError("stop after the folder check")

    monkeypatch.setattr(app, "run_pipeline", stop_here)
    with caplog.at_level(logging.INFO, logger="car_eval_io"), pytest.raises(RuntimeError, match="stop"):
        app.run(AppConfig(), LocalFile(car_data), paths)
    assert f"results folder: {paths.output}" in caplog.messages
    assert pipeline.run_pipeline is not stop_here  # only the app's reference was replaced


def test_finished_run_files_are_all_under_the_output_folder(
    finished_run: tuple[AppConfig, RunPaths, app.RunOutcome, str],
) -> None:
    _, paths, outcome, _ = finished_run
    assert paths.output.is_absolute()
    assert outcome.files
    for path in outcome.files:
        assert path.is_file()
        assert path.is_relative_to(paths.output)
