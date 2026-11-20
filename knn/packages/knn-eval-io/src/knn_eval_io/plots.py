"""Figures for the k-NN report, drawn from the core package's result types."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay

from knn_eval_core.diagnostics import Curve
from knn_eval_core.domain import CLASS_LABELS
from knn_eval_core.neighbors import SearchResult
from knn_eval_core.pipeline import PipelineResult

logger = logging.getLogger(__name__)


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
                logger.debug("saved figure %s", path)
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


def distance_metrics(sink: FigureSink, search: SearchResult) -> None:
    """Best cross-validated macro-F1 per distance metric (Lecture 3B's list of metrics)."""
    items = sorted(search.best_per_metric.items(), key=lambda kv: kv[1].macro_f1_mean)
    with sink.figure("02_distance_metrics", (7, 4)) as fig:
        ax = fig.subplots()
        names = [m.value for m, _ in items]
        means = np.array([s.macro_f1_mean for _, s in items])
        stds = np.array([s.macro_f1_std for _, s in items])
        colors = ["tab:blue" if m == search.spec.params.metric else "tab:gray" for m, _ in items]
        ax.barh(names, means, xerr=stds, color=colors, capsize=3)
        for y, (_, score) in enumerate(items):
            ax.text(0.01, y, f"k={score.params.k}, {score.params.voting.value}", va="center", color="white")
        ax.set(title="Best CV macro-F1 per distance metric (chosen in blue)", xlabel="macro-F1", xlim=(0, 1))


def k_curve(sink: FigureSink, curve: Curve, chosen_k: int) -> None:
    with sink.figure("03_choosing_k") as fig:
        ax = fig.subplots()
        _plot_curve(ax, curve)
        ax.axvline(chosen_k, color="tab:red", linestyle="--", label=f"chosen k = {chosen_k}")
        ax.legend()
        ax.set(
            title="Choosing K: small K follows noise, large K blurs classes", xlabel="k", ylabel="macro-F1"
        )


def learning_curve(sink: FigureSink, curve: Curve) -> None:
    with sink.figure("04_learning_curve") as fig:
        ax = fig.subplots()
        _plot_curve(ax, curve)
        ax.set(
            title="Learning curve: is more stored data helping?",
            xlabel="stored training cars",
            ylabel="macro-F1",
        )


def confusion(sink: FigureSink, matrix: np.ndarray) -> None:
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, row_totals, out=np.zeros(matrix.shape), where=row_totals > 0)
    with sink.figure("05_confusion_matrix", (12, 5)) as fig:
        panels = ((matrix, "d", "counts"), (normalised, ".2f", "row-normalised (recall)"))
        for ax, (values, fmt, title) in zip(fig.subplots(1, 2), panels, strict=True):
            ConfusionMatrixDisplay(values, display_labels=list(CLASS_LABELS)).plot(
                ax=ax, cmap="Blues", colorbar=False, values_format=fmt
            )
            ax.set_title(f"Test set confusion matrix - {title}")


def importances(sink: FigureSink, values: Mapping[str, float]) -> None:
    with sink.figure("06_feature_importance", (7, 4)) as fig:
        ax = fig.subplots()
        ordered = pd.Series(values).sort_values()
        ax.barh(ordered.index, ordered.to_numpy(), color="tab:blue")
        ax.set(title="Test macro-F1 drop when one attribute is shuffled", xlabel="macro-F1 drop")


def all_figures(sink: FigureSink, class_counts: pd.Series, result: PipelineResult) -> list[Path]:
    class_balance(sink, class_counts)
    distance_metrics(sink, result.search)
    k_curve(sink, result.k_curve, result.search.spec.params.k)
    learning_curve(sink, result.learning_curve)
    confusion(sink, result.confusion)
    importances(sink, result.importances)
    return sink.saved
