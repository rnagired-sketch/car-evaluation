"""Files a k-NN run leaves behind: the model recipe and a machine-readable results summary.

The fitted model is not pickled. `model.json` records how to rebuild it (data source, seed,
split fractions, inputs and k-NN settings); "fitting" a k-NN model only stores the training
cars, so rebuilding is instant and the recipe stays human-readable and safe to load.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knn_eval_core.dataset import CarDataset, DataSplits, SplitFractions, ValidationReport
from knn_eval_core.diagnostics import PredictionCost
from knn_eval_core.domain import Acceptability, Seed
from knn_eval_core.evaluator import Explanation, KnnEvaluator
from knn_eval_core.metrics import Scores
from knn_eval_core.neighbors import CandidateScore, DistanceMetric, KnnSpec
from knn_eval_core.pipeline import Misclassification, PipelineResult
from knn_eval_io.converters import CONVERTER, describe_errors
from knn_eval_io.sources import DataSource

logger = logging.getLogger(__name__)


class ArtifactError(Exception):
    pass


@dataclass(frozen=True)
class ModelRecipe:
    data: DataSource
    rows: int  # rows kept after cleaning; guards against rebuilding from different data
    seed: Seed
    fractions: SplitFractions
    spec: KnnSpec

    def rebuild(self, dataset: CarDataset) -> KnnEvaluator:
        if len(dataset) != self.rows:
            raise ArtifactError(
                f"the recipe was trained on {self.rows} cleaned rows, this dataset has {len(dataset)}"
            )
        splits = DataSplits.stratified(dataset, self.fractions, self.seed)
        return KnnEvaluator.fit(self.spec, splits.train, self.seed)


@dataclass(frozen=True)
class ModelSummary:
    spec: KnnSpec
    cv_macro_f1_mean: float
    cv_macro_f1_std: float
    best_per_metric: Mapping[DistanceMetric, CandidateScore]
    k_scores: Mapping[int, float]
    validation: Scores


@dataclass(frozen=True)
class SpreadSummary:
    mean: float
    std: float
    min: float


@dataclass(frozen=True)
class StabilitySummary:
    splits: int
    macro_f1: SpreadSummary
    false_acceptance_mean: float


@dataclass(frozen=True)
class RunSummary:
    model: str  # "k-NN", so a results.json can be told apart from the decision tree's
    validation_report: ValidationReport
    split_counts: Mapping[str, Mapping[Acceptability, int]]
    knn: ModelSummary
    test_scores: Scores
    importances: Mapping[str, float]
    prediction_cost: PredictionCost
    stability: StabilitySummary | None
    misclassified: tuple[Misclassification, ...]
    explanations: tuple[Explanation, ...]


def summarize(
    report: ValidationReport, result: PipelineResult, explanations: tuple[Explanation, ...]
) -> RunSummary:
    stability = None
    if result.stability is not None:
        s = result.stability
        stability = StabilitySummary(
            splits=len(s.seeds),
            macro_f1=SpreadSummary(
                float(s.macro_f1.mean()), float(s.macro_f1.std()), float(s.macro_f1.min())
            ),
            false_acceptance_mean=float(s.false_acceptance.mean()),
        )
    search = result.search
    return RunSummary(
        model="k-NN",
        validation_report=report,
        split_counts=result.splits.class_counts(),
        knn=ModelSummary(
            spec=search.spec,
            cv_macro_f1_mean=search.cv_macro_f1_mean,
            cv_macro_f1_std=search.cv_macro_f1_std,
            best_per_metric=search.best_per_metric,
            k_scores=search.k_scores,
            validation=result.validation_scores,
        ),
        test_scores=result.test_scores,
        importances=result.importances,
        prediction_cost=result.cost,
        stability=stability,
        misclassified=result.misclassified,
        explanations=explanations,
    )


def to_json(value: Any, as_type: Any = None) -> str:
    """Strict JSON (NaN becomes null) of any value the converter can unstructure."""
    data = CONVERTER.unstructure(value, unstructure_as=as_type)
    return json.dumps(_finite(data), indent=2, allow_nan=False) + "\n"


def _finite(data: Any) -> Any:
    match data:
        case float() if not math.isfinite(data):
            return None
        case dict():
            return {_key(key): _finite(value) for key, value in data.items()}
        case list() | tuple():
            return [_finite(value) for value in data]
        case _:
            return data


def _key(key: Any) -> Any:
    """JSON object keys must be strings; mapping keys such as K values (ints) become text."""
    return key if isinstance(key, str) else str(key)


def write_json(path: Path, value: Any, as_type: Any = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(value, as_type), encoding="utf-8")
    logger.debug("wrote %s", path)
    return path


def load_recipe(path: Path) -> ModelRecipe:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ArtifactError(
            f"cannot read model recipe {path}: {error.strerror}. Run `knn-eval run` first."
        ) from error
    except json.JSONDecodeError as error:
        raise ArtifactError(f"model recipe {path} is not valid JSON: {error}") from error
    try:
        return CONVERTER.structure(document, ModelRecipe)
    except Exception as error:
        raise ArtifactError(
            f"model recipe {path} is invalid: " + "; ".join(describe_errors(error))
        ) from error
