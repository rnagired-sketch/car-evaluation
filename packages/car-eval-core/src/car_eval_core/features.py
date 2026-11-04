"""Ordinal attributes, expert-model concept scores, and the conditions a decision path imposes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import pandas as pd


@dataclass(frozen=True)
class OrdinalFeature:
    """A categorical attribute whose levels have a natural order (lowest first)."""

    name: str
    levels: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.levels) < 2:
            raise ValueError(f"{self.name}: an ordinal feature needs at least two levels")
        if len(set(self.levels)) != len(self.levels):
            raise ValueError(f"{self.name}: duplicate levels in {self.levels}")

    @property
    def ranks(self) -> dict[str, int]:
        return {level: rank for rank, level in enumerate(self.levels)}

    @property
    def max_rank(self) -> int:
        return len(self.levels) - 1

    def condition(self, threshold: float, goes_left: bool) -> LevelCondition:
        """Turn a CART rule `rank <= threshold` (or its negation) into the levels it admits."""
        admitted = {rank for rank in range(len(self.levels)) if (rank <= threshold) == goes_left}
        return LevelCondition(self, frozenset(admitted))


FEATURES: Final[tuple[OrdinalFeature, ...]] = (
    OrdinalFeature("buying", ("low", "med", "high", "vhigh")),
    OrdinalFeature("maint", ("low", "med", "high", "vhigh")),
    OrdinalFeature("doors", ("2", "3", "4", "5more")),
    OrdinalFeature("persons", ("2", "4", "more")),
    OrdinalFeature("lug_boot", ("small", "med", "big")),
    OrdinalFeature("safety", ("low", "med", "high")),
)
FEATURE_NAMES: Final[tuple[str, ...]] = tuple(f.name for f in FEATURES)
FEATURE_BY_NAME: Final[dict[str, OrdinalFeature]] = {f.name: f for f in FEATURES}


@dataclass(frozen=True)
class ConceptFeature:
    """Intermediate concept of the expert model (Bohanec & Rajkovic, 1988).

    Its score is the sum of the ordinal ranks of its source attributes, so a CART tree
    can split on the combination directly instead of rebuilding it from many leaves.
    """

    name: str
    sources: tuple[str, ...]
    meaning: str

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError(f"{self.name}: a concept needs at least one source attribute")
        unknown = set(self.sources) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"{self.name}: unknown source attributes {sorted(unknown)}")

    @property
    def max_score(self) -> int:
        return sum(FEATURE_BY_NAME[s].max_rank for s in self.sources)

    def score(self, X: pd.DataFrame) -> pd.Series:
        return X[list(self.sources)].sum(axis=1).rename(self.name)

    def condition(self, threshold: float, goes_left: bool) -> ScoreCondition:
        cut = math.floor(threshold)  # scores are integers, CART thresholds sit halfway between
        if goes_left:
            return ScoreCondition(self, 0, cut)
        return ScoreCondition(self, cut + 1, self.max_score)


CONCEPTS: Final[tuple[ConceptFeature, ...]] = (
    ConceptFeature("price", ("buying", "maint"), "0 = cheapest"),
    ConceptFeature("comfort", ("doors", "persons", "lug_boot"), "0 = least roomy"),
)
CONCEPT_BY_NAME: Final[dict[str, ConceptFeature]] = {c.name: c for c in CONCEPTS}


@dataclass(frozen=True)
class LevelCondition:
    """'attribute is one of these levels': all splits on one raw attribute along a path."""

    feature: OrdinalFeature
    admitted: frozenset[int]

    @property
    def levels(self) -> tuple[str, ...]:
        return tuple(lvl for rank, lvl in enumerate(self.feature.levels) if rank in self.admitted)

    def holds_for(self, rank: int) -> bool:
        return rank in self.admitted


@dataclass(frozen=True)
class ScoreCondition:
    """'concept score lies in [low, high]': all splits on one concept along a path."""

    concept: ConceptFeature
    low: int
    high: int

    def holds_for(self, score: int) -> bool:
        return self.low <= score <= self.high


Condition = LevelCondition | ScoreCondition


def tighten(old: Condition, new: Condition) -> Condition:
    """Combine two conditions on the same feature met along one decision path."""
    match old, new:
        case LevelCondition(feature, a), LevelCondition(other, b) if feature == other:
            return LevelCondition(feature, a & b)
        case ScoreCondition(concept, low1, high1), ScoreCondition(other, low2, high2) if concept == other:
            return ScoreCondition(concept, max(low1, low2), min(high1, high2))
        case _:
            raise TypeError(f"cannot combine {old!r} with {new!r}")


MODEL_COLUMNS: Final[tuple[str, ...]] = (*FEATURE_NAMES, *CONCEPT_BY_NAME)
MODEL_INPUTS: Final = "6 raw attributes + price & comfort scores"


def model_inputs(X: pd.DataFrame) -> pd.DataFrame:
    """The tree's inputs: the encoded raw attributes plus the two concept scores."""
    return X.assign(**{c.name: c.score(X) for c in CONCEPTS})


def path_condition(column: str, threshold: float, goes_left: bool) -> Condition:
    """The condition a CART split on `column` imposes on the side a car goes to."""
    if column in FEATURE_BY_NAME:
        return FEATURE_BY_NAME[column].condition(threshold, goes_left)
    if column in CONCEPT_BY_NAME:
        return CONCEPT_BY_NAME[column].condition(threshold, goes_left)
    raise KeyError(f"{column!r} is not a model input; expected one of {MODEL_COLUMNS}")
