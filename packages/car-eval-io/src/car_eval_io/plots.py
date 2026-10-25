"""Figures for the report, drawn from the core package's result types."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree

from car_eval_core.diagnostics import Curve
from car_eval_core.domain import CLASS_LABELS, Acceptability
from car_eval_core.evaluator import CarEvaluator, Importances
from car_eval_core.pipeline import PipelineResult
from car_eval_core.tree import PruningResult


@dataclass
class FigureSink:
    """Where finished figures go: saved as PNG to `directory`, shown inline, or both."""

    directory: Path | None
    show: bool
    saved: list[Path] = field(default_factory=list)

    @contextlib.contextmanager
    def figure(self, name: str, size: tuple[float, float] = (7, 5)) -> Iterator[Figure]:
        """Own one figure: on a clean exit lay it out, save and/or show it; always free it."""
        fig = plt.figure(figsize=size)
        try:
            yield fig
            fig.tight_layout()
            if self.directory is not None:
                self.directory.mkdir(parents=True, exist_ok=True)
                path = self.directory / f"{name}.png"
                fig.savefig(path, dpi=150)
                self.saved.append(path)
            if self.show:
                plt.show()
        finally:
            plt.close(fig)


def _plot_curve(ax: Axes, curve: Curve) -> None:
    for mean, std, label in (
        (curve.train_mean, curve.train_std, "training"),
        (curve.cv_mean, curve.cv_std, "cross-validation"),
    ):
        ax.plot(curve.x, mean, marker="o", label=label)
        ax.fill_between(curve.x, mean - std, mean + std, alpha=0.2)
    ax.legend()


def class_balance(sink: FigureSink, counts: pd.Series) -> None:
    with sink.figure("01_class_balance", (6, 4)) as fig:
        ax = fig.subplots()
        bars = ax.bar(counts.index, counts.to_numpy(), color="tab:blue")
        ax.bar_label(bars, labels=[f"{n} ({n / counts.sum():.0%})" for n in counts])
        ax.set(title="Class balance (full dataset)", xlabel="acceptability", ylabel="cars")


def depth_curve(sink: FigureSink, curve: Curve) -> None:
    with sink.figure("02_depth_validation_curve") as fig:
        ax = fig.subplots()
        _plot_curve(ax, curve)
        ax.set(title="Under- vs overfitting: tree depth", xlabel="max_depth", ylabel="macro-F1")


def pruning(sink: FigureSink, result: PruningResult) -> None:
    with sink.figure("03_cost_complexity_pruning") as fig:
        ax = fig.subplots()
        ax.plot(
            result.alphas,
            result.validation_macro_f1,
            marker="o",
            color="tab:blue",
            label="validation macro-F1",
        )
        ax.axvline(
            result.chosen_alpha,
            color="tab:red",
            linestyle="--",
            label=f"chosen alpha = {result.chosen_alpha:.4f}",
        )
        ax.set(
            title="Pruning: validation score vs tree size", xlabel="ccp_alpha", ylabel="validation macro-F1"
        )
        ax.set_xscale("symlog", linthresh=1e-4)
        leaves_ax = ax.twinx()
        leaves_ax.plot(result.alphas, result.n_leaves, color="tab:gray", alpha=0.6, label="leaves")
        leaves_ax.set_ylabel("leaves", color="tab:gray")
        ax.legend(loc="lower left")


def learning_curve(sink: FigureSink, curve: Curve) -> None:
    with sink.figure("04_learning_curve") as fig:
        ax = fig.subplots()
        _plot_curve(ax, curve)
        ax.set(title="Learning curve: is more data needed?", xlabel="training cars", ylabel="macro-F1")


def confusion(sink: FigureSink, matrix: np.ndarray) -> None:
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, row_totals, out=np.zeros(matrix.shape), where=row_totals > 0)
    with sink.figure("05_confusion_matrix", (12, 5)) as fig:
        panels = ((matrix, "d", "counts"), (normalised, ".2f", "row-normalised (recall)"))
        for ax, (values, fmt, title) in zip(fig.subplots(1, 2), panels, strict=False):
            ConfusionMatrixDisplay(values, display_labels=list(CLASS_LABELS)).plot(
                ax=ax, cmap="Blues", colorbar=False, values_format=fmt
            )
            ax.set_title(f"Test set confusion matrix - {title}")


def importances(sink: FigureSink, values: Importances) -> None:
    with sink.figure("06_feature_importance", (12, 4)) as fig:
        panels = (
            (values.impurity, "impurity decrease (model inputs)"),
            (values.permutation, "test macro-F1 drop when shuffled (raw attributes)"),
        )
        for ax, (mapping, title) in zip(fig.subplots(1, 2), panels, strict=False):
            ordered = pd.Series(mapping).sort_values()
            ax.barh(ordered.index, ordered.to_numpy(), color="tab:blue")
            ax.set_title(title)


def decision_tree(sink: FigureSink, evaluator: CarEvaluator) -> None:
    tree = evaluator.tree
    with sink.figure("07_decision_tree", (28, 14)) as fig:
        plot_tree(
            tree,
            ax=fig.subplots(),
            feature_names=list(evaluator.spec.features.columns),
            class_names=[Acceptability(int(c)).label for c in tree.classes_],
            filled=True,
            rounded=True,
            fontsize=7,
        )


def all_figures(sink: FigureSink, class_counts: pd.Series, result: PipelineResult) -> list[Path]:
    class_balance(sink, class_counts)
    depth_curve(sink, result.depth_curve)
    pruning(sink, result.chosen.pruning)
    learning_curve(sink, result.learning_curve)
    confusion(sink, result.confusion)
    importances(sink, result.importances)
    decision_tree(sink, result.evaluator)
    return sink.saved
