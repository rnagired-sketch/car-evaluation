"""Files a run leaves behind: the model recipe and a machine-readable results summary.

The trained tree is not pickled. `model.json` records how to rebuild it (data source,
seed, split fractions, feature set, hyperparameters); refitting is deterministic and takes
well under a second, and the recipe stays human-readable and safe to load.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from car_eval_core.benchmarks import AdequacyTargets, ComparisonRow
from car_eval_core.dataset import CarDataset, DataSplits, SplitFractions, ValidationReport
from car_eval_core.domain import Acceptability, CcpAlpha, Seed
from car_eval_core.evaluator import CarEvaluator, Explanation, Importances, TreePurity
from car_eval_core.features import FeatureSet
from car_eval_core.metrics import Scores
from car_eval_core.pipeline import Misclassification, PipelineResult
from car_eval_core.tree import TreeSpec
from car_eval_io.converters import CONVERTER, describe_errors
from car_eval_io.sources import DataSource

logger = logging.getLogger(__name__)


class ArtifactError(Exception):
    pass


@dataclass(frozen=True)
class ModelRecipe:
    data: DataSource
    rows: int  # rows kept after cleaning; guards against rebuilding from different data
    seed: Seed
    fractions: SplitFractions
    spec: TreeSpec

    def rebuild(self, dataset: CarDataset) -> CarEvaluator:
        if len(dataset) != self.rows:
            raise ArtifactError(
                f"the recipe was trained on {self.rows} cleaned rows, this dataset has {len(dataset)}"
            )
        splits = DataSplits.stratified(dataset, self.fractions, self.seed)
        return CarEvaluator.fit(self.spec, splits.train, self.seed)


@dataclass(frozen=True)
class CandidateSummary:
    features: FeatureSet
    spec: TreeSpec
    cv_macro_f1_mean: float
    cv_macro_f1_std: float
    chosen_alpha: CcpAlpha
    validation: Scores


@dataclass(frozen=True)
class AdequacySummary:
    passes: bool
    validation: Scores
    targets: AdequacyTargets


@dataclass(frozen=True)
class SpreadSummary:
    mean: float
    std: float
    min: float


@dataclass(frozen=True)
class StabilitySummary:
    splits: int
    rev0_macro_f1: SpreadSummary
    rev1_macro_f1: SpreadSummary
    rev1_wins: int
    rev1_false_acceptance_mean: float
    rev1_leaves_mean: float
    root_features: Mapping[str, int]


@dataclass(frozen=True)
class RunSummary:
    validation_report: ValidationReport
    split_counts: Mapping[str, Mapping[Acceptability, int]]
    candidates: tuple[CandidateSummary, ...]
    chosen: TreeSpec
    comparison: tuple[ComparisonRow, ...]
    adequacy: AdequacySummary
    test_scores: Scores
    purity: TreePurity
    importances: Importances
    stability: StabilitySummary | None
    misclassified: tuple[Misclassification, ...]
    explanations: tuple[Explanation, ...]


def _spread(values: Any) -> SpreadSummary:
    return SpreadSummary(float(values.mean()), float(values.std()), float(values.min()))


def summarize(
    report: ValidationReport, result: PipelineResult, explanations: tuple[Explanation, ...]
) -> RunSummary:
    stability = None
    if result.stability is not None:
        s = result.stability
        stability = StabilitySummary(
            splits=len(s.seeds),
            rev0_macro_f1=_spread(s.rev0_macro_f1),
            rev1_macro_f1=_spread(s.rev1_macro_f1),
            rev1_wins=s.rev1_wins,
            rev1_false_acceptance_mean=float(s.rev1_false_acceptance.mean()),
            rev1_leaves_mean=float(s.rev1_leaves.mean()),
            root_features=s.root_features,
        )
    return RunSummary(
        validation_report=report,
        split_counts=result.splits.class_counts(),
        candidates=tuple(
            CandidateSummary(
                features,
                d.spec,
                d.search.cv_macro_f1_mean,
                d.search.cv_macro_f1_std,
                d.pruning.chosen_alpha,
                d.validation,
            )
            for features, d in result.candidates.items()
        ),
        chosen=result.chosen.spec,
        comparison=result.comparison,
        adequacy=AdequacySummary(result.adequacy.passes, result.adequacy.validation, result.adequacy.targets),
        test_scores=result.test_scores,
        purity=result.purity,
        importances=result.importances,
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
            return {key: _finite(value) for key, value in data.items()}
        case list() | tuple():
            return [_finite(value) for value in data]
        case _:
            return data


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
            f"cannot read model recipe {path}: {error.strerror}. Run `car-eval run` first."
        ) from error
    except json.JSONDecodeError as error:
        raise ArtifactError(f"model recipe {path} is not valid JSON: {error}") from error
    try:
        return CONVERTER.structure(document, ModelRecipe)
    except Exception as error:
        raise ArtifactError(
            f"model recipe {path} is invalid: " + "; ".join(describe_errors(error))
        ) from error
