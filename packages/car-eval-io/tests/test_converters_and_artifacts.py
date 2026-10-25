from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from car_eval_core.dataset import SplitFractions
from car_eval_core.domain import Acceptability, Car, CcpAlpha, Seed
from car_eval_core.features import (
    CONCEPT_BY_NAME,
    FEATURE_BY_NAME,
    FeatureSet,
    LevelCondition,
    ScoreCondition,
)
from car_eval_core.tree import TreeHyperparameters, TreeSpec
from car_eval_io import app
from car_eval_io.artifacts import ArtifactError, ModelRecipe, load_recipe, to_json, write_json
from car_eval_io.config import AppConfig, FigureSettings
from car_eval_io.converters import CONVERTER
from car_eval_io.paths import RunPaths
from car_eval_io.sources import DataSource, LocalFile, UciDownload


class TestConverter:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("vgood", Acceptability.VGOOD), ("ACC", Acceptability.ACC), (0, Acceptability.UNACC)],
    )
    def test_structures_class_labels_and_codes(self, value: object, expected: Acceptability) -> None:
        assert CONVERTER.structure(value, Acceptability) is expected

    def test_unstructures_class_labels_including_mapping_keys(self) -> None:
        counts = {Acceptability.UNACC: 3, Acceptability.VGOOD: 1}
        assert CONVERTER.unstructure(counts, Mapping[Acceptability, int]) == {"unacc": 3, "vgood": 1}

    def test_conditions_become_readable_records(self) -> None:
        level = LevelCondition(FEATURE_BY_NAME["safety"], frozenset({1, 2}))
        score = ScoreCondition(CONCEPT_BY_NAME["price"], 0, 2)
        assert CONVERTER.unstructure(level) == {"feature": "safety", "levels": ["med", "high"]}
        assert CONVERTER.unstructure(score) == {
            "concept": "price",
            "sources": ["buying", "maint"],
            "low": 0,
            "high": 2,
            "max": 6,
        }

    def test_arrays_become_lists(self) -> None:
        assert CONVERTER.unstructure(np.array([1.5, 2.0])) == [1.5, 2.0]

    @pytest.mark.parametrize(
        "source", [LocalFile(Path("/d/car.data")), UciDownload(), UciDownload(timeout_seconds=3.0)]
    )
    def test_data_source_round_trip_is_tagged(self, source: DataSource) -> None:
        data = CONVERTER.unstructure(source, DataSource)
        assert data["kind"] in {"local", "uci"}
        assert CONVERTER.structure(data, DataSource) == source  # pyright: ignore[reportArgumentType]  (cattrs accepts unions)

    def test_whole_config_round_trips(self) -> None:
        config = AppConfig(
            profile="quick",
            figures=FigureSettings(save=False, show=True),
            cars=(Car("low", "med", "5more", "more", "small", "high"),),
        )
        assert CONVERTER.structure(CONVERTER.unstructure(config), AppConfig) == config


def recipe(rows: int = 1728, data: Path = Path("/data/car.data")) -> ModelRecipe:
    return ModelRecipe(
        data=LocalFile(data),
        rows=rows,
        seed=Seed(42),
        fractions=SplitFractions(),
        spec=TreeSpec(
            FeatureSet.WITH_CONCEPTS, TreeHyperparameters("entropy", None, 2, 1, None, CcpAlpha(0.001))
        ),
    )


class TestJson:
    def test_nan_becomes_null(self) -> None:
        assert json.loads(to_json({"a": float("nan"), "b": [1.0, float("inf")]})) == {
            "a": None,
            "b": [1.0, None],
        }

    def test_write_json_creates_folders(self, tmp_path: Path) -> None:
        path = write_json(tmp_path / "a" / "b.json", {"x": 1})
        assert json.loads(path.read_text()) == {"x": 1}


class TestRecipe:
    def test_round_trip_through_a_file(self, tmp_path: Path) -> None:
        path = write_json(tmp_path / "model.json", recipe())
        assert load_recipe(path) == recipe()

    def test_file_is_human_readable(self, tmp_path: Path) -> None:
        data = json.loads(write_json(tmp_path / "model.json", recipe()).read_text())
        assert data["spec"]["features"] == "with_concepts"
        assert data["data"]["kind"] == "local"
        assert data["spec"]["params"]["max_depth"] is None

    def test_rebuild_reproduces_the_trained_model(
        self, finished_run: tuple[AppConfig, RunPaths, app.RunOutcome, str]
    ) -> None:
        _, paths, outcome, _ = finished_run
        recipe = load_recipe(paths.model)
        rebuilt = recipe.rebuild(app.load_dataset(recipe.data)[0])
        test = outcome.result.splits.test
        np.testing.assert_array_equal(rebuilt.predict(test.X), outcome.result.test_predictions)

    def test_rebuild_refuses_different_data(self, car_data: Path) -> None:
        dataset, _ = app.load_dataset(LocalFile(car_data))
        with pytest.raises(ArtifactError, match="1000 cleaned rows"):
            recipe(rows=1000).rebuild(dataset)

    def test_missing_file_points_to_run(self, tmp_path: Path) -> None:
        with pytest.raises(ArtifactError, match="car-eval run"):
            load_recipe(tmp_path / "model.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "model.json"
        path.write_text("{", encoding="utf-8")
        with pytest.raises(ArtifactError, match="not valid JSON"):
            load_recipe(path)

    def test_schema_violations(self, tmp_path: Path) -> None:
        path = tmp_path / "model.json"
        data = json.loads(to_json(recipe()))
        data["spec"]["features"] = "everything"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ArtifactError, match=r"\$\.spec\.features"):
            load_recipe(path)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"kind": "ftp"}, "unknown kind 'ftp' (one of: local, uci) @ $.data"),
        ({"path": "/d/car.data"}, "missing 'kind' (one of: local, uci) @ $.data"),
    ],
)
def test_recipe_with_bad_data_source_names_the_problem(
    tmp_path: Path, data: dict[str, str], message: str
) -> None:
    document = json.loads(to_json(recipe()))
    document["data"] = data
    path = tmp_path / "model.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ArtifactError, match=re.escape(message)):
        load_recipe(path)
