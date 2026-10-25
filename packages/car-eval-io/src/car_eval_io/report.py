"""Plain-text rendering of results for the console and `report.txt`."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import assert_never

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.tree import export_text

from car_eval_core.benchmarks import AdequacyCheck, ComparisonRow
from car_eval_core.dataset import DataSplits, ValidationReport
from car_eval_core.domain import CLASS_CODES, CLASS_LABELS, Acceptability, Car
from car_eval_core.evaluator import CarEvaluator, Explanation, Importances, TreePurity
from car_eval_core.features import FEATURES, Condition, LevelCondition, ScoreCondition
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


def render_candidate(developed: DevelopedTree) -> str:
    search = developed.search
    return (
        f"[{developed.spec.features.description}]\n"
        f"  grid-search best: {search.spec.params}\n"
        f"  CV macro-F1 {search.cv_macro_f1_mean:.3f} +/- {search.cv_macro_f1_std:.3f}\n"
        f"  pruning: ccp_alpha = {developed.pruning.chosen_alpha:.5f}\n"
        f"  validation macro-F1 after pruning: {developed.validation.macro_f1:.3f}"
    )


def comparison_table(rows: Iterable[ComparisonRow]) -> pd.DataFrame:
    records = [{"split": r.split, "model": r.model, **dataclasses.asdict(r.scores)} for r in rows]
    return pd.DataFrame(records).set_index(["split", "model"]).sort_index()


def render_adequacy(check: AdequacyCheck) -> str:
    verdict = "ADEQUATE - keep the decision tree" if check.passes else "INADEQUATE - run the k-NN trade study"
    return (
        f"  validation macro-F1 {check.validation.macro_f1:.3f} (target >= {check.targets.min_macro_f1}), "
        f"false acceptance {check.validation.false_acceptance:.3f} "
        f"(target <= {check.targets.max_false_acceptance})\n"
        f"  -> {verdict}"
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
    rev0, rev1 = study.rev0_macro_f1, study.rev1_macro_f1
    return (
        f"  over {n} random splits (test set each time):\n"
        f"  rev0 macro-F1 {rev0.mean():.3f} +/- {rev0.std():.3f} (min {rev0.min():.3f})\n"
        f"  rev1 macro-F1 {rev1.mean():.3f} +/- {rev1.std():.3f} (min {rev1.min():.3f}); "
        f"rev1 >= rev0 on {study.rev1_wins}/{n} splits\n"
        f"  rev1 false acceptance {study.rev1_false_acceptance.mean():.3f} "
        f"+/- {study.rev1_false_acceptance.std():.3f}; "
        f"leaves {study.rev1_leaves.mean():.0f} +/- {study.rev1_leaves.std():.0f}; root split: {roots}"
    )


def render_rules(evaluator: CarEvaluator) -> str:
    return export_text(
        evaluator.tree,
        feature_names=list(evaluator.spec.features.columns),
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
        section("3-4. Tree building (CART grid search) and pruning (cost complexity on validation)"),
        "\n\n".join(render_candidate(d) for d in result.candidates.values()),
        f"\nselected on validation: {result.chosen.spec.features.description}",
    ]
    if result.comparison:
        parts += [
            section("5. Model comparison (all fitted on the training split)"),
            comparison_table(result.comparison).to_string(float_format="{:.3f}".format),
        ]
    parts += [
        render_adequacy(result.adequacy),
        section("6. Selected tree on the untouched test split"),
        render_test_report(result.splits, result),
        pd.Series(dataclasses.asdict(result.test_scores)).to_string(),
        render_purity(result.purity),
        importance_table(result.importances).to_string(float_format="{:.3f}".format),
    ]
    if result.stability is not None:
        parts += [
            section("7. Robustness: instability across random splits"),
            render_stability(result.stability),
        ]
    parts += [section("8. Requested cars: rating, confidence and decision path")]
    parts += [render_explanation(e) for e in explanations] or [
        "  (no cars listed under `cars` in the config)"
    ]
    shown = result.misclassified[:max_errors_shown]
    parts += [
        section("9. Error analysis: misclassified test cars"),
        f"{len(result.misclassified)} of {len(result.splits.test)} test cars misclassified "
        f"(showing {len(shown)})",
    ]
    for item in shown:
        parts += [f"  true rating: {item.truth.label.upper()}", render_explanation(item.explanation)]
    parts += [section("10. Decision rules"), render_rules(result.evaluator)]
    return "\n".join(parts)
