"""The commands behind the CLI.

Each takes the loaded `AppConfig` (what to do), the paths given on the command line (where
to read and write), and an optional text stream for the report itself (standard output by
default, looked up at call time so redirection works). Progress goes to logging.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import pandas as pd
import yaml

from knn_eval_core.dataset import CarDataset, ValidationReport
from knn_eval_core.evaluator import Explanation
from knn_eval_core.pipeline import PipelineResult, run_pipeline
from knn_eval_io import plots, report
from knn_eval_io.artifacts import ModelRecipe, load_recipe, summarize, write_json
from knn_eval_io.config import AppConfig
from knn_eval_io.converters import CONVERTER
from knn_eval_io.paths import RunPaths
from knn_eval_io.sources import DataSource, describe, read_raw

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunOutcome:
    result: PipelineResult
    explanations: tuple[Explanation, ...]
    files: tuple[Path, ...]


def load_dataset(source: DataSource) -> tuple[CarDataset, ValidationReport]:
    logger.info("loading data from %s", describe(source))
    return CarDataset.from_frame(read_raw(source))


def run(config: AppConfig, data: DataSource, paths: RunPaths, out: TextIO | None = None) -> RunOutcome:
    """Full k-NN evaluation: report, figures, `results.json` and the `model.json` recipe."""
    out = out or sys.stdout
    started = time.perf_counter()
    logger.info("knn-eval run: profile %r", config.profile)
    logger.info("results folder: %s", paths.create())
    dataset, validation = load_dataset(data)
    result = run_pipeline(dataset, config.pipeline)
    logger.info("explaining %d car(s) from the config", len(config.cars))
    explanations = tuple(result.evaluator.explain(car) for car in config.cars)

    text = report.render_run(config.profile, describe(data), validation, result, explanations)
    print(text, file=out)

    paths.create()  # again, in case it was removed while the pipeline ran
    paths.report.write_text(text + "\n", encoding="utf-8")
    recipe = ModelRecipe(
        data=data,
        rows=len(dataset),
        seed=config.pipeline.seed,
        fractions=config.pipeline.fractions,
        spec=result.search.spec,
    )
    files = [
        paths.report,
        write_json(paths.results, summarize(validation, result, explanations)),
        write_json(paths.model, recipe),
    ]
    if config.figures.save or config.figures.show:
        logger.info("drawing figures (save=%s, show=%s)", config.figures.save, config.figures.show)
        sink = plots.FigureSink(paths.figures if config.figures.save else None, config.figures.show)
        counts = pd.Series({a.label: n for a, n in dataset.data.class_counts().items()})
        files += plots.all_figures(sink, counts, result)

    for path in files:
        logger.info("  wrote %s", path.relative_to(paths.output))
    logger.info(
        "wrote %d files to %s; finished in %.1f s", len(files), paths.output, time.perf_counter() - started
    )
    return RunOutcome(result, explanations, tuple(files))


def explain(
    config: AppConfig, paths: RunPaths, data: DataSource | None = None, out: TextIO | None = None
) -> tuple[Explanation, ...]:
    """Rate and explain `config.cars` with the model a previous `run` saved in `paths.output`.

    The data is re-read from `data` when given, otherwise from the source recorded in the model.
    """
    out = out or sys.stdout
    if not config.cars:
        raise ValueError("the config lists no cars to explain; add them under `cars:`")
    logger.info("knn-eval explain: %d car(s), model from %s", len(config.cars), paths.model)
    recipe = load_recipe(paths.model)
    dataset, _ = load_dataset(data if data is not None else recipe.data)
    logger.info("rebuilding the k-NN model from the recipe (%s)", recipe.spec.params.describe())
    evaluator = recipe.rebuild(dataset)
    explanations = tuple(evaluator.explain(car) for car in config.cars)
    for explanation in explanations:
        print(report.render_explanation(explanation), file=out)
    path = write_json(paths.explanations, explanations, tuple[Explanation, ...])
    logger.info("wrote %s", path)
    return explanations


def show_config(config: AppConfig, out: TextIO | None = None) -> None:
    """Print the fully resolved configuration, defaults included."""
    out = out or sys.stdout
    print(yaml.safe_dump(CONVERTER.unstructure(config), sort_keys=False), file=out, end="")
