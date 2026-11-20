"""Shared fixtures for the application layer. One small end-to-end run is shared by the session."""

from __future__ import annotations

import io
import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # headless: never open windows during tests

from knn_eval_io import app
from knn_eval_io.config import AppConfig, load_config
from knn_eval_io.paths import RunPaths
from knn_eval_io.sources import LocalFile

KNN_ROOT = Path(__file__).resolve().parents[3]  # the k-NN application's folder
CAR_DATA = KNN_ROOT.parent / "data" / "car.data"  # the repository's shared copy of the dataset
CONFIGS = KNN_ROOT / "configs"

FAST_PIPELINE = """
pipeline:
  grid:
    k: [5, 7]
    metric: [euclidean, cosine]
    voting: [inverse_distance]
  cv: {n_splits: 3, n_repeats: 1}
  stability_seeds: 2
  permutation_repeats: 2
"""

CARS = """
cars:
  - {buying: low, maint: low, doors: 4, persons: 4, lug_boot: big, safety: high}
  - {buying: low, maint: low, doors: 4, persons: 2, lug_boot: big, safety: high}
"""


@pytest.fixture(autouse=True)
def reset_logging() -> Iterator[None]:
    """`knn-eval` installs global logging handlers; undo them after every test."""
    yield
    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, "_knn_eval_handler", False)]:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.WARNING)
    for name in ("knn_eval_core", "knn_eval_io"):
        logging.getLogger(name).setLevel(logging.NOTSET)
    logging.captureWarnings(False)


@pytest.fixture(scope="session")
def car_data() -> Path:
    return CAR_DATA


@pytest.fixture(scope="session")
def configs_dir() -> Path:
    return CONFIGS


@pytest.fixture
def write_config(tmp_path: Path) -> Callable[[str], Path]:
    """Write a YAML config into this test's temporary folder and return its path."""

    def write(body: str) -> Path:
        path = tmp_path / "config.yaml"
        path.write_text(body, encoding="utf-8")
        return path

    return write


def fast_config_text(save_figures: bool = False) -> str:
    return f"figures: {{save: {str(save_figures).lower()}}}\n" + FAST_PIPELINE + CARS


FinishedRun = tuple[AppConfig, RunPaths, app.RunOutcome, str]


@pytest.fixture(scope="session")
def finished_run(tmp_path_factory: pytest.TempPathFactory) -> FinishedRun:
    """One small `run` shared by the session: config, output paths, outcome, console text."""
    folder = tmp_path_factory.mktemp("run")
    config_path = folder / "config.yaml"
    config_path.write_text(fast_config_text(save_figures=True), encoding="utf-8")
    config = load_config(config_path)
    paths = RunPaths(folder / "outputs")
    out = io.StringIO()
    outcome = app.run(config, LocalFile(CAR_DATA), paths, out)
    return config, paths, outcome, out.getvalue()
