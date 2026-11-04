"""Plain-text rendering of results for the console and `report.txt`."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import assert_never

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.tree import export_text

from car_eval_core.dataset import DataSplits, ValidationReport
from car_eval_core.domain import CLASS_CODES, CLASS_LABELS, Acceptability, Car
from car_eval_core.evaluator import CarEvaluator, Explanation, Importances, TreePurity
from car_eval_core.features import (
    FEATURES,
    MODEL_COLUMNS,
    MODEL_INPUTS,
    Condition,
    LevelCondition,
    ScoreCondition,
)
from car_eval_core.pipeline import PipelineResult
from car_eval_core.stability import StabilityStudy
from car_eval_core.tree import DevelopedTree

RULE = "=" * 78


def section(title: str) -> str:
    return f"\n{RULE}\n{title}\n{RULE}"


def describe_condition(condition: Condition) -> str:
    match condition:
        case LevelCondition(feature):
            return f"{feature.name} in {{{', '.join(condition.levels)}}}"
        case ScoreCondition(concept, low, high):
            span = str(low) if low == high else f"{low}..{high}"
            return (
                f"{concept.name} score ({' + '.join(concept.sources)}) in {span} "
                f"of 0..{concept.max_score} ({concept.meaning})"
            )
        case _:
            assert_never(condition)


def describe_car(car: Car) -> str:
    return ", ".join(f"{f.name}={getattr(car, f.name)}" for f in FEATURES)


def render_explanation(explanation: Explanation) -> str:
    support = ", ".join(f"{a.label}: {n}" for a, n in explanation.leaf_counts.items() if n)
    steps = (
        "\n".join(f"      - {describe_condition(c)}" for c in explanation.path) or "      - (root is a leaf)"
    )
    return (
        f"  Car: {describe_car(explanation.car)}\n"
        f"    Rating: {explanation.rating.label.upper()}  (confidence {explanation.confidence:.2f})\n"
        f"    Leaf support: {explanation.leaf_support} training cars -> {{{support}}}\n"
        f"    Because:\n{steps}"
    )


def render_validation_report(report: ValidationReport) -> str:
    return "\n".join(
        f"  {name.replace('_', ' ')}: {value}" for name, value in dataclasses.asdict(report).items()
    )


def counts_table(counts: Mapping[str, Mapping[Acceptability, int]]) -> pd.DataFrame:
    table = pd.DataFrame({name: {a.label: n for a, n in c.items()} for name, c in counts.items()})
    table.loc["total"] = table.sum()
    return table


def render_tree(developed: DevelopedTree) -> str:
    search = developed.search
    return (
        f"inputs: {MODEL_INPUTS}\n"
        f"  grid-search best: {search.spec.params}\n"
        f"  CV macro-F1 {search.cv_macro_f1_mean:.3f} +/- {search.cv_macro_f1_std:.3f}\n"
        f"  pruning: ccp_alpha = {developed.pruning.chosen_alpha:.5f}\n"
        f"  validation macro-F1 after pruning: {developed.validation.macro_f1:.3f}"
    )


def render_purity(purity: TreePurity) -> str:
    return (
        f"  Gini at root: {purity.root_gini:.3f}  ->  "
        f"weighted Gini over leaves: {purity.weighted_leaf_gini:.4f}\n"
        f"  pure leaves: {purity.pure_leaf_share:.0%} of {purity.n_leaves}, depth {purity.depth}"
    )


def importance_table(importances: Importances) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "impurity decrease": pd.Series(importances.impurity),
            "permutation (raw attrs, test)": pd.Series(importances.permutation),
        }
    ).sort_values("impurity decrease", ascending=False)


def render_stability(study: StabilityStudy) -> str:
    roots = ", ".join(f"{name} x{n}" for name, n in study.root_features.items())
    n = len(study.seeds)
    f1 = study.macro_f1
    return (
        f"  over {n} random splits (test set each time):\n"
        f"  test macro-F1 {f1.mean():.3f} +/- {f1.std():.3f} (min {f1.min():.3f})\n"
        f"  false acceptance {study.false_acceptance.mean():.3f} +/- {study.false_acceptance.std():.3f}; "
        f"leaves {study.leaves.mean():.0f} +/- {study.leaves.std():.0f}; root split: {roots}"
    )


def render_rules(evaluator: CarEvaluator) -> str:
    return export_text(
        evaluator.tree,
        feature_names=list(MODEL_COLUMNS),
        class_names=[Acceptability(int(c)).label for c in evaluator.tree.classes_],
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
        f"profile: {profile}",
        f"source: {source}",
        render_validation_report(validation),
        section("2. Stratified train / validation / test split"),
        counts_table(result.splits.class_counts()).to_string(),
        section("3. Tree building (CART grid search) and pruning (cost complexity on validation)"),
        render_tree(result.developed),
        section("4. The tree on the untouched test split"),
        render_test_report(result.splits, result),
        pd.Series(dataclasses.asdict(result.test_scores)).to_string(),
        render_purity(result.purity),
        importance_table(result.importances).to_string(float_format="{:.3f}".format),
    ]
    if result.stability is not None:
        parts += [
            section("5. Robustness: instability across random splits"),
            render_stability(result.stability),
        ]
    parts += [section("6. Requested cars: rating, confidence and decision path")]
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
    parts += [section("8. Decision rules"), render_rules(result.evaluator)]
    return "\n".join(parts)
