"""Ordinal car attributes and the expert-model concept scores used as k-NN inputs."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Final, assert_never

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
    """Intermediate concept of the expert model (Bohanec & Rajkovic, 1988): a sum of ranks."""

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


CONCEPTS: Final[tuple[ConceptFeature, ...]] = (
    ConceptFeature("price", ("buying", "maint"), "0 = cheapest"),
    ConceptFeature("comfort", ("doors", "persons", "lug_boot"), "0 = least roomy"),
)


class InputSet(enum.Enum):
    """Which columns the distance is measured on."""

    RAW = "raw"  # the six encoded attributes
    WITH_CONCEPTS = "with_concepts"  # plus the price and comfort scores, as the decision tree uses

    @property
    def description(self) -> str:
        match self:
            case InputSet.RAW:
                return "6 raw attributes"
            case InputSet.WITH_CONCEPTS:
                return "6 raw attributes + price & comfort scores"
            case _:
                assert_never(self)

    @property
    def columns(self) -> tuple[str, ...]:
        match self:
            case InputSet.RAW:
                return FEATURE_NAMES
            case InputSet.WITH_CONCEPTS:
                return (*FEATURE_NAMES, *(c.name for c in CONCEPTS))
            case _:
                assert_never(self)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Model inputs from encoded raw attributes; never mutates `X`."""
        match self:
            case InputSet.RAW:
                return X.copy()
            case InputSet.WITH_CONCEPTS:
                return X.assign(**{c.name: c.score(X) for c in CONCEPTS})
            case _:
                assert_never(self)
