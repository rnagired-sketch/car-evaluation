"""Validation, cleaning, encoding and stratified splitting.

Typestate: a raw string table only becomes a `CarDataset` through `CarDataset.from_frame`,
so every later stage can rely on clean, encoded data.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, cast

import pandas as pd
from sklearn.model_selection import train_test_split

from car_eval_core.domain import CLASS_LABELS, Acceptability, Seed
from car_eval_core.features import FEATURE_NAMES, FEATURES

logger = logging.getLogger(__name__)

TARGET: Final = "class"
COLUMN_NAMES: Final[tuple[str, ...]] = (*FEATURE_NAMES, TARGET)


@dataclass(frozen=True)
class ValidationReport:
    rows_read: int
    rows_with_missing_values: int
    rows_with_unknown_levels: int
    exact_duplicates_removed: int
    conflicting_rows_removed: int
    rows_kept: int


@dataclass(frozen=True)
class Partition:
    """Encoded raw attributes (ordinal ranks) and integer class codes."""

    X: pd.DataFrame
    y: pd.Series

    def __post_init__(self) -> None:
        if len(self.X) != len(self.y):
            raise ValueError(f"X has {len(self.X)} rows but y has {len(self.y)}")
        if tuple(self.X.columns) != FEATURE_NAMES:
            raise ValueError(f"expected columns {FEATURE_NAMES}, got {tuple(self.X.columns)}")

    def __len__(self) -> int:
        return len(self.y)

    def class_counts(self) -> dict[Acceptability, int]:
        return {a: int((self.y == a).sum()) for a in Acceptability}

    @staticmethod
    def concat(first: Partition, second: Partition) -> Partition:
        return Partition(pd.concat([first.X, second.X]), pd.concat([first.y, second.y]))


@dataclass(frozen=True)
class CarDataset:
    """Validated, cleaned and encoded data."""

    data: Partition

    @staticmethod
    def from_frame(raw: pd.DataFrame) -> tuple[CarDataset, ValidationReport]:
        """Validate a table of raw strings with columns `COLUMN_NAMES`.

        Levels are compared after trimming whitespace and lower-casing. Rows with missing
        values or unknown levels are dropped, exact duplicates are collapsed, and cars that
        appear with two different labels are dropped entirely because the data contradicts
        itself about them.
        """
        if tuple(raw.columns) != COLUMN_NAMES:
            raise ValueError(f"expected columns {COLUMN_NAMES}, got {tuple(raw.columns)}")

        text = raw.apply(lambda col: col.astype("string").str.strip().str.lower())
        missing = text.isna().any(axis=1)
        known = {f.name: f.levels for f in FEATURES} | {TARGET: CLASS_LABELS}
        unknown = pd.Series(False, index=text.index)
        for column, levels in known.items():
            unknown |= text[column].notna() & ~text[column].isin(levels)

        valid = text[~missing & ~unknown]
        exact_duplicates = valid.duplicated()
        valid = valid[~exact_duplicates]
        conflicting = valid.duplicated(subset=list(FEATURE_NAMES), keep=False)
        valid = valid[~conflicting]
        report = ValidationReport(
            rows_read=len(raw),
            rows_with_missing_values=int(missing.sum()),
            rows_with_unknown_levels=int((unknown & ~missing).sum()),
            exact_duplicates_removed=int(exact_duplicates.sum()),
            conflicting_rows_removed=int(conflicting.sum()),
            rows_kept=len(valid),
        )
        _log_report(report)
        if valid.empty:
            raise ValueError("no valid rows left after cleaning")

        X = pd.DataFrame({f.name: valid[f.name].map(f.ranks).astype(int) for f in FEATURES})
        y = valid[TARGET].map(lambda label: int(Acceptability.from_label(str(label)))).astype(int)
        data = Partition(X.reset_index(drop=True), y.rename(TARGET).reset_index(drop=True))
        return CarDataset(data), report

    def __len__(self) -> int:
        return len(self.data)


def _log_report(report: ValidationReport) -> None:
    dropped = report.rows_read - report.rows_kept
    if dropped:
        logger.warning(
            "dropped %d of %d rows: %d with missing values, %d with unknown levels, "
            "%d exact duplicates, %d with conflicting labels",
            dropped,
            report.rows_read,
            report.rows_with_missing_values,
            report.rows_with_unknown_levels,
            report.exact_duplicates_removed,
            report.conflicting_rows_removed,
        )
    logger.info("validated %d rows, kept %d", report.rows_read, report.rows_kept)


@dataclass(frozen=True)
class SplitFractions:
    train: float = 0.50
    validation: float = 0.20
    test: float = 0.30

    def __post_init__(self) -> None:
        parts = (self.train, self.validation, self.test)
        if any(p <= 0 for p in parts) or not math.isclose(sum(parts), 1.0):
            raise ValueError(f"fractions must be positive and sum to 1, got {parts}")


@dataclass(frozen=True)
class DataSplits:
    """Train builds trees, validation chooses among them, test is scored once at the end."""

    train: Partition
    validation: Partition
    test: Partition

    @staticmethod
    def stratified(dataset: CarDataset, fractions: SplitFractions, seed: Seed) -> DataSplits:
        """Stratify on the class so every split keeps the same (imbalanced) class mix."""
        development, test = _stratified_split(dataset.data, fractions.test, seed)
        validation_share = fractions.validation / (fractions.train + fractions.validation)
        train, validation = _stratified_split(development, validation_share, seed)
        logger.debug(
            "seed %d split: %d train / %d validation / %d test", seed, len(train), len(validation), len(test)
        )
        return DataSplits(train, validation, test)

    @property
    def development(self) -> Partition:
        """Train + validation, for cross-validated diagnostics that never touch the test set."""
        return Partition.concat(self.train, self.validation)

    def class_counts(self) -> Mapping[str, dict[Acceptability, int]]:
        return {
            "train": self.train.class_counts(),
            "validation": self.validation.class_counts(),
            "test": self.test.class_counts(),
        }


def _stratified_split(data: Partition, share: float, seed: Seed) -> tuple[Partition, Partition]:
    """Split off `share` of the rows, keeping the class mix; returns (rest, share)."""
    X_rest, X_part, y_rest, y_part = train_test_split(
        data.X, data.y, test_size=share, stratify=data.y, random_state=seed
    )
    # train_test_split preserves pandas types at runtime; its annotations are looser.
    return (
        Partition(cast(pd.DataFrame, X_rest), cast(pd.Series, y_rest)),
        Partition(cast(pd.DataFrame, X_part), cast(pd.Series, y_part)),
    )
