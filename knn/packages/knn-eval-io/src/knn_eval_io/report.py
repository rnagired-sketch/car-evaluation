"""Plain-text rendering of k-NN results for the console and `report.txt`."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping

import pandas as pd
from sklearn.metrics import classification_report

from knn_eval_core.dataset import DataSplits, ValidationReport
from knn_eval_core.diagnostics import PredictionCost
from knn_eval_core.domain import CLASS_CODES, CLASS_LABELS, Acceptability, Car
from knn_eval_core.evaluator import Explanation
from knn_eval_core.features import FEATURES
from knn_eval_core.neighbors import SearchResult
from knn_eval_core.pipeline import PipelineResult
from knn_eval_core.stability import StabilityStudy

RULE = "=" * 78


def section(title: str) -> str:
    return f"\n{RULE}\n{title}\n{RULE}"


def describe_car(car: Car) -> str:
    return ", ".join(f"{f.name}={getattr(car, f.name)}" for f in FEATURES)


def render_explanation(explanation: Explanation) -> str:
    votes = ", ".join(f"{a.label}: {share:.2f}" for a, share in explanation.votes.items() if share > 0)
    lines = [
        f"  Car: {describe_car(explanation.car)}",
        f"    Rating: {explanation.rating.label.upper()}  (confidence {explanation.confidence:.2f})",
        f"    Vote shares: {{{votes}}}",
        f"    Nearest {len(explanation.neighbors)} training cars:",
    ]
    lines += [
        f"      {i}. {n.rating.label:<5}  distance {n.distance:.3f}  weight {n.weight:.2f}  "
        f"{describe_car(n.car)}"
        for i, n in enumerate(explanation.neighbors, start=1)
    ]
    return "\n".join(lines)


def render_validation_report(report: ValidationReport) -> str:
    return "\n".join(
        f"  {name.replace('_', ' ')}: {value}" for name, value in dataclasses.asdict(report).items()
    )


def counts_table(counts: Mapping[str, Mapping[Acceptability, int]]) -> pd.DataFrame:
    table = pd.DataFrame({name: {a.label: n for a, n in c.items()} for name, c in counts.items()})
    table.loc["total"] = table.sum()
    return table


def metric_table(search: SearchResult) -> pd.DataFrame:
    """The best setting found for each distance metric (Lecture 3B), best first."""
    records = [
        {
            "distance": metric.value,
            "best k": score.params.k,
            "voting": score.params.voting.value,
            "CV macro-F1": score.macro_f1_mean,
            "+/-": score.macro_f1_std,
            "CV accuracy": score.accuracy_mean,
        }
        for metric, score in search.best_per_metric.items()
    ]
    return pd.DataFrame(records).set_index("distance")


def render_search(search: SearchResult, validation_macro_f1: float) -> str:
    k_line = ", ".join(f"k={k}: {score:.3f}" for k, score in search.k_scores.items())
    return (
        f"inputs: {search.spec.inputs.description}\n"
        f"  chosen: {search.spec.params.describe()}\n"
        f"  CV macro-F1 {search.cv_macro_f1_mean:.3f} +/- {search.cv_macro_f1_std:.3f}\n"
        f"  validation macro-F1: {validation_macro_f1:.3f}\n\n"
        f"Best setting per distance metric (cross-validated on the training split):\n"
        f"{metric_table(search).to_string(float_format='{:.3f}'.format)}\n\n"
        f"Choosing K (chosen distance, voting and scaling; CV macro-F1):\n  {k_line}"
    )


def render_cost(cost: PredictionCost) -> str:
    return (
        f"  prediction cost: {cost.seconds_per_car * 1e6:.1f} microseconds per car, "
        f"comparing each car with {cost.stored_cars} stored training cars"
    )


def render_stability(study: StabilityStudy) -> str:
    f1 = study.macro_f1
    return (
        f"  over {len(study.seeds)} random splits (test set each time):\n"
        f"  test macro-F1 {f1.mean():.3f} +/- {f1.std():.3f} (min {f1.min():.3f}); "
        f"false acceptance {study.false_acceptance.mean():.3f} +/- {study.false_acceptance.std():.3f}"
    )


def importance_table(importances: Mapping[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"macro-F1 drop when shuffled (test)": pd.Series(importances)}).sort_values(
        "macro-F1 drop when shuffled (test)", ascending=False
    )


def render_test_report(splits: DataSplits, result: PipelineResult) -> str:
    return str(
        classification_report(
            splits.test.y,
            result.test_predictions,
            labels=list(CLASS_CODES),
            target_names=list(CLASS_LABELS),
            digits=3,
            zero_division=0,  # pyright: ignore[reportArgumentType]  (sklearn is unannotated)
        )
    )


def render_run(
    profile: str,
    source: str,
    validation: ValidationReport,
    result: PipelineResult,
    explanations: Iterable[Explanation],
    max_errors_shown: int = 10,
) -> str:
    parts = [
        section("1. Acquire, validate, clean and encode"),
        "model: k-nearest neighbors",
        f"profile: {profile}",
        f"source: {source}",
        render_validation_report(validation),
        section("2. Stratified train / validation / test split"),
        counts_table(result.splits.class_counts()).to_string(),
        section("3. Choosing the distance metric, K, voting rule and scaling (Lecture 3B)"),
        render_search(result.search, result.validation_scores.macro_f1),
        section("4. The chosen k-NN on the untouched test split"),
        render_test_report(result.splits, result),
        pd.Series(dataclasses.asdict(result.test_scores)).to_string(),
        render_cost(result.cost),
        importance_table(result.importances).to_string(float_format="{:.3f}".format),
    ]
    if result.stability is not None:
        parts += [
            section("5. Robustness: scores across random splits"),
            render_stability(result.stability),
        ]
    parts += [section("6. Requested cars: rating, vote shares and nearest neighbors")]
    parts += [render_explanation(e) for e in explanations] or [
        "  (no cars listed under `cars` in the config)"
    ]
    shown = result.misclassified[:max_errors_shown]
    parts += [
        section("7. Error analysis: misclassified test cars"),
        f"{len(result.misclassified)} of {len(result.splits.test)} test cars misclassified "
        f"(showing {len(shown)})",
    ]
    for item in shown:
        parts += [f"  true rating: {item.truth.label.upper()}", render_explanation(item.explanation)]
    return "\n".join(parts)
