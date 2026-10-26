"""The argparse entry point, driven exactly as `uv run car-eval ...` would drive it."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from car_eval_io import app
from car_eval_io.__main__ import (
    COMMANDS,
    DEFAULT_OUTPUT,
    EXIT_BAD_INPUT,
    EXIT_FAILURE,
    EXIT_INTERRUPTED,
    EXIT_OK,
    build_parser,
    data_source,
    main,
)
from car_eval_io.config import AppConfig, load_config
from car_eval_io.converters import CONVERTER
from car_eval_io.sources import LocalFile, UciDownload

CAR = "cars: [{buying: low, maint: low, doors: 4, persons: 4, lug_boot: big, safety: high}]"
FAST = """
figures: {save: false}
pipeline:
  feature_sets: [with_concepts]
  compare_benchmarks: false
  stability_seeds: 0
  permutation_repeats: 1
  cv: {n_splits: 3, n_repeats: 1}
  grid:
    criterion: [entropy]
    max_depth: [null]
    min_samples_split: [2]
    min_samples_leaf: [1]
    class_weight: [null]
"""


def parse(*argv: str) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def rejected(capsys: pytest.CaptureFixture[str], *argv: str) -> str:
    """Parse arguments that must fail; return argparse's error output."""
    with pytest.raises(SystemExit) as caught:
        build_parser().parse_args(list(argv))
    assert caught.value.code == EXIT_BAD_INPUT
    return capsys.readouterr().err


class TestParser:
    @pytest.mark.parametrize("command", list(COMMANDS))
    def test_every_command_requires_a_config(self, command: str, capsys: pytest.CaptureFixture[str]) -> None:
        assert "--config" in rejected(capsys, command)

    def test_run_with_local_data(self, car_data: Path) -> None:
        args = parse("run", "-c", "c.yaml", "--data", str(car_data), "--output", "results")
        assert args.config == Path("c.yaml")
        assert data_source(args) == LocalFile(car_data.resolve())
        assert args.output == Path("results").resolve()

    def test_run_with_download(self) -> None:
        assert data_source(parse("run", "-c", "c.yaml", "--download")) == UciDownload()

    def test_output_defaults_to_outputs_in_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        output = parse("run", "-c", "c.yaml", "--download").output
        assert output == tmp_path.resolve() / DEFAULT_OUTPUT
        assert output.is_absolute()

    def test_relative_paths_resolve_against_the_working_directory(
        self, car_data: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(car_data.parent)
        args = parse("run", "-c", "c.yaml", "-d", "car.data", "-o", "out")
        assert data_source(args) == LocalFile(car_data)
        assert args.output == car_data.parent / "out"

    def test_run_needs_a_data_source(self, capsys: pytest.CaptureFixture[str]) -> None:
        err = rejected(capsys, "run", "-c", "c.yaml")
        assert "one of the arguments -d/--data --download is required" in err

    def test_data_and_download_are_exclusive(
        self, car_data: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        err = rejected(capsys, "run", "-c", "c.yaml", "--data", str(car_data), "--download")
        assert "not allowed with" in err

    def test_data_file_must_exist(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        err = rejected(capsys, "run", "-c", "c.yaml", "--data", str(tmp_path / "none.data"))
        assert "no such file" in err

    def test_explain_data_is_optional(self) -> None:
        assert data_source(parse("explain", "-c", "c.yaml")) is None

    def test_show_config_takes_no_paths(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert "unrecognized arguments" in rejected(capsys, "show-config", "-c", "c.yaml", "--output", "x")

    @pytest.mark.parametrize(
        "argv", [[], ["train", "--config", "c.yaml"], ["run", "-c", "c.yaml", "--download", "--seed", "1"]]
    )
    def test_rejects_unknown_commands_and_options(
        self, argv: list[str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        rejected(capsys, *argv)


class TestMain:
    def test_bad_config_exits_2_with_message(
        self, write_config: Callable[[str], Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        argv = ["run", "--config", str(write_config("pipeline: {sed: 1}")), "--download"]
        assert main(argv) == EXIT_BAD_INPUT
        assert "unknown key(s): sed @ $.pipeline" in capsys.readouterr().err

    def test_output_that_is_a_file_exits_2(
        self, write_config: Callable[[str], Path], car_data: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        argv = ["run", "-c", str(write_config("")), "-d", str(car_data), "-o", str(car_data)]
        with pytest.raises(SystemExit) as caught:
            main(argv)
        assert caught.value.code == EXIT_BAD_INPUT
        assert "not a folder" in capsys.readouterr().err

    def test_explain_before_run_exits_1(
        self, write_config: Callable[[str], Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        argv = ["explain", "--config", str(write_config(CAR)), "--output", str(tmp_path / "out")]
        assert main(argv) == EXIT_FAILURE
        assert "car-eval run" in capsys.readouterr().err

    def test_explain_without_cars_exits_1(
        self, write_config: Callable[[str], Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["explain", "--config", str(write_config(""))]) == EXIT_FAILURE
        assert "no cars" in capsys.readouterr().err

    def test_show_config_round_trips(
        self, write_config: Callable[[str], Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = write_config("pipeline: {seed: 3, grid: {max_depth: [5, null]}}")
        assert main(["show-config", "--config", str(path)]) == EXIT_OK
        printed = yaml.safe_load(capsys.readouterr().out)
        assert printed["pipeline"]["grid"]["max_depth"] == [5, None]
        assert CONVERTER.structure(printed, AppConfig) == load_config(path)


@pytest.mark.slow
def test_run_then_explain_end_to_end(
    write_config: Callable[[str], Path], tmp_path: Path, car_data: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = str(write_config(FAST + CAR))
    output = tmp_path / "results"
    assert main(["run", "-c", config, "--data", str(car_data), "--output", str(output)]) == EXIT_OK
    run_out, run_err = capsys.readouterr()
    assert "Rating: VGOOD" in run_out
    assert " INFO " not in run_out  # standard output holds the report only
    assert "[1/4] splitting data" in run_err  # progress is logged to standard error
    assert "evaluation plan finished in" in run_err
    recipe = json.loads((output / "model.json").read_text())
    assert recipe["data"] == {"path": str(car_data.resolve()), "kind": "local"}
    assert not (output / "figures").exists()

    # Without --data, explain re-reads the source recorded in model.json.
    assert main(["explain", "-c", config, "--output", str(output)]) == EXIT_OK
    explained = json.loads((output / "explanations.json").read_text())
    assert explained[0]["rating"] == "vgood"
    assert "Rating: VGOOD" in capsys.readouterr().out

    # An explicit --data replaces the recorded source; a different dataset is refused.
    shortened = tmp_path / "short.data"
    shortened.write_text("".join(car_data.read_text().splitlines(keepends=True)[:-1]))
    assert main(["explain", "-c", config, "-o", str(output), "-d", str(shortened)]) == EXIT_FAILURE
    assert "1728 cleaned rows" in capsys.readouterr().err

    # Every command appended to one log file in the output folder, errors included.
    log = (output / "car-eval.log").read_text()
    assert log.index("car-eval run:") < log.index("car-eval explain:")
    assert log.count("car-eval explain:") == 2
    assert "ERROR   car_eval_io: the recipe was trained on 1728 cleaned rows" in log


class TestLoggedErrors:
    def test_config_errors_are_error_log_lines(
        self, write_config: Callable[[str], Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["show-config", "-c", str(write_config("pipeline: {sed: 1}"))])
        first, second = capsys.readouterr().err.splitlines()[:2]
        assert first.split(" ", 1)[1].startswith("ERROR   car_eval_io: invalid configuration ")
        assert second == "  - unknown key(s): sed @ $.pipeline"

    def test_unexpected_errors_are_logged_with_a_traceback(
        self,
        write_config: Callable[[str], Path],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def broken(config: AppConfig) -> None:
            raise RuntimeError("boom")

        monkeypatch.setattr(app, "show_config", broken)
        assert main(["show-config", "-c", str(write_config(""))]) == EXIT_FAILURE
        err = capsys.readouterr().err
        assert "ERROR   car_eval_io: unexpected error" in err
        assert "Traceback (most recent call last)" in err
        assert "RuntimeError: boom" in err

    def test_interrupt_is_a_warning_with_exit_130(
        self,
        write_config: Callable[[str], Path],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def interrupted(config: AppConfig) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(app, "show_config", interrupted)
        assert main(["show-config", "-c", str(write_config(""))]) == EXIT_INTERRUPTED
        assert "WARNING car_eval_io: interrupted" in capsys.readouterr().err

    def test_errors_reach_the_log_file(
        self, write_config: Callable[[str], Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        output = tmp_path / "out"
        main(["explain", "-c", str(write_config(CAR)), "-o", str(output)])
        capsys.readouterr()
        assert "ERROR   car_eval_io: cannot read model recipe" in (output / "car-eval.log").read_text()

    def test_log_file_can_be_switched_off(
        self, write_config: Callable[[str], Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        output = tmp_path / "out"
        main(["explain", "-c", str(write_config(f"logging: {{save: false}}\n{CAR}")), "-o", str(output)])
        assert "cannot read model recipe" in capsys.readouterr().err
        assert not (output / "car-eval.log").exists()

    def test_warning_level_hides_progress_but_not_errors(
        self, write_config: Callable[[str], Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        text = f"logging: {{level: warning, save: false}}\n{CAR}"
        main(["explain", "-c", str(write_config(text)), "-o", str(tmp_path / "out")])
        err = capsys.readouterr().err
        assert " INFO " not in err
        assert " ERROR " in err


@pytest.mark.slow
def test_module_entry_point_runs_as_a_subprocess(write_config: Callable[[str], Path]) -> None:
    path = write_config("pipeline: {seed: 9}")
    completed = subprocess.run(
        [sys.executable, "-m", "car_eval_io", "show-config", "--config", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == EXIT_OK, completed.stderr
    assert yaml.safe_load(completed.stdout)["pipeline"]["seed"] == 9
