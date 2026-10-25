"""Shared fixtures. Expensive objects are session-scoped and must not be mutated by tests."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import pytest

from car_eval_core.dataset import COLUMN_NAMES, CarDataset, DataSplits, SplitFractions
from car_eval_core.domain import Car, Seed
from car_eval_core.evaluator import CarEvaluator
from car_eval_core.features import FEATURES, FeatureSet
from car_eval_core.metrics import CvSettings
from car_eval_core.tree import SearchGrid, TreeHyperparameters, TreeSpec

REPO_ROOT = Path(__file__).resolve().parents[3]
CAR_DATA = REPO_ROOT / "data" / "car.data"
SEED = Seed(42)


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    return pd.read_csv(CAR_DATA, header=None, names=list(COLUMN_NAMES), dtype=str)


@pytest.fixture(scope="session")
def dataset(raw_frame: pd.DataFrame) -> CarDataset:
    return CarDataset.from_frame(raw_frame)[0]


@pytest.fixture(scope="session")
def splits(dataset: CarDataset) -> DataSplits:
    return DataSplits.stratified(dataset, SplitFractions(), SEED)


@pytest.fixture(scope="session")
def tiny_grid() -> SearchGrid:
    return SearchGrid(
        criterion=("entropy",),
        max_depth=(None,),
        min_samples_split=(2,),
        min_samples_leaf=(1,),
        class_weight=(None,),
    )


@pytest.fixture(scope="session")
def small_grid() -> SearchGrid:
    return SearchGrid(
        criterion=("gini", "entropy"),
        max_depth=(4, None),
        min_samples_split=(2,),
        min_samples_leaf=(1, 4),
        class_weight=(None,),
    )


@pytest.fixture(scope="session")
def fast_cv() -> CvSettings:
    return CvSettings(n_splits=3, n_repeats=1)


@pytest.fixture(scope="session")
def full_tree_params() -> TreeHyperparameters:
    return TreeHyperparameters("entropy", None, 2, 1, None)


@pytest.fixture(scope="session")
def concept_spec(full_tree_params: TreeHyperparameters) -> TreeSpec:
    return TreeSpec(FeatureSet.WITH_CONCEPTS, full_tree_params)


@pytest.fixture(scope="session")
def evaluator(concept_spec: TreeSpec, splits: DataSplits) -> CarEvaluator:
    return CarEvaluator.fit(concept_spec, splits.train, SEED)


@pytest.fixture(scope="session")
def all_cars() -> list[Car]:
    """Every one of the 4*4*4*3*3*3 = 1728 attribute combinations."""
    return [Car(*levels) for levels in itertools.product(*(f.levels for f in FEATURES))]


RawRow = Sequence[str | None]


@pytest.fixture
def make_raw() -> Callable[[Sequence[RawRow]], pd.DataFrame]:
    def build(rows: Sequence[RawRow]) -> pd.DataFrame:
        return pd.DataFrame([list(r) for r in rows], columns=list(COLUMN_NAMES), dtype=object)

    return build
