"""Domain vocabulary: identifier types, the target classes and a single car."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Final, NewType

import pandas as pd

from knn_eval_core.features import FEATURES

Seed = NewType("Seed", int)

DEFAULT_SEED: Final = Seed(42)


class Acceptability(enum.IntEnum):
    """Target classes, ordered from worst to best."""

    UNACC = 0
    ACC = 1
    GOOD = 2
    VGOOD = 3

    @property
    def label(self) -> str:
        return self.name.lower()

    @property
    def is_acceptable(self) -> bool:
        return self >= Acceptability.ACC

    @classmethod
    def from_label(cls, label: str) -> Acceptability:
        try:
            return cls[label.strip().upper()]
        except KeyError:
            raise ValueError(f"unknown class label {label!r}; expected one of {CLASS_LABELS}") from None


CLASS_LABELS: Final[tuple[str, ...]] = tuple(a.label for a in Acceptability)
CLASS_CODES: Final[tuple[int, ...]] = tuple(int(a) for a in Acceptability)


@dataclass(frozen=True)
class Car:
    """One car described by its raw attribute levels. Invalid levels cannot be constructed."""

    buying: str
    maint: str
    doors: str
    persons: str
    lug_boot: str
    safety: str

    def __post_init__(self) -> None:
        for feature in FEATURES:
            value = getattr(self, feature.name)
            if value not in feature.ranks:
                raise ValueError(f"{feature.name}={value!r}; expected one of {feature.levels}")

    @staticmethod
    def from_ranks(ranks: pd.Series) -> Car:
        """Inverse of `encoded`: build a car from one row of ordinal ranks."""
        return Car(**{f.name: f.levels[int(ranks[f.name])] for f in FEATURES})

    def encoded(self) -> pd.DataFrame:
        """One-row frame of ordinal ranks, in model column order."""
        return pd.DataFrame([{f.name: f.ranks[getattr(self, f.name)] for f in FEATURES}])
