from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from car_eval_core.domain import Car, Seed
from car_eval_core.features import FeatureSet
from car_eval_core.metrics import CvSettings
from car_eval_core.pipeline import PipelineSettings
from car_eval_io.config import ConfigError, FigureSettings, load_config, merge, parse_config
from car_eval_io.logging_setup import LogLevel, LogSettings

HERE = Path("/project/configs/run.yaml")


def problems(text: str) -> list[str]:
    with pytest.raises(ConfigError) as caught:
        parse_config(text, HERE)
    return caught.value.problems


class TestShippedConfig:
    @pytest.mark.parametrize("profile", ["default", "quick", "colab"])
    def test_every_profile_parses(self, profile: str, configs_dir: Path) -> None:
        config = load_config(configs_dir / "config.yaml", profile)
        assert config.profile == profile
        assert len(config.cars) == 5

    def test_selects_the_default_profile(self, configs_dir: Path) -> None:
        assert load_config(configs_dir / "config.yaml").profile == "default"

    def test_default_profile_matches_pipeline_defaults(self, configs_dir: Path) -> None:
        config = load_config(configs_dir / "config.yaml")
        assert config.pipeline == PipelineSettings()
        assert config.figures == FigureSettings(save=True, show=False)

    def test_quick_profile_only_shrinks_the_work(self, configs_dir: Path) -> None:
        base = load_config(configs_dir / "config.yaml")
        quick = load_config(configs_dir / "config.yaml", "quick")
        assert quick.pipeline.grid.size == 2
        assert quick.pipeline.cv == CvSettings(n_splits=5, n_repeats=1)
        assert quick.pipeline.stability_seeds == 3
        assert quick.pipeline.fractions == base.pipeline.fractions
        assert quick.cars == base.cars
        assert quick.figures == FigureSettings(save=False, show=False)

    def test_colab_profile_shows_figures_inline(self, configs_dir: Path) -> None:
        base = load_config(configs_dir / "config.yaml")
        colab = load_config(configs_dir / "config.yaml", "colab")
        assert colab.figures == FigureSettings(save=True, show=True)
        assert colab.pipeline == base.pipeline

    def test_holds_no_paths(self, configs_dir: Path) -> None:
        import yaml

        document = yaml.safe_load((configs_dir / "config.yaml").read_text())
        sections = [document, *document["profiles"].values()]
        for section in sections:
            assert not {"data", "output", "path", "directory"} & set(section)


BASE = """
figures: {save: true}
pipeline:
  seed: 1
  grid: {max_depth: [4, 8], criterion: [gini]}
cars: [{buying: low, maint: low, doors: 4, persons: 4, lug_boot: big, safety: high}]
"""


class TestProfiles:
    def test_no_profiles_section_means_base_settings(self) -> None:
        config = parse_config(BASE, HERE)
        assert config.profile == "default"
        assert config.pipeline.seed == Seed(1)

    def test_nested_sections_merge_key_by_key(self) -> None:
        text = BASE + "profile: p\nprofiles:\n  p: {pipeline: {grid: {criterion: [entropy]}}}"
        grid = parse_config(text, HERE).pipeline.grid
        assert grid.criterion == ("entropy",)
        assert grid.max_depth == (4, 8)

    def test_lists_are_replaced_not_appended(self) -> None:
        text = BASE + "profile: p\nprofiles:\n  p: {pipeline: {grid: {max_depth: [null]}}}"
        assert parse_config(text, HERE).pipeline.grid.max_depth == (None,)

    def test_profiles_can_add_sections_missing_from_the_base(self) -> None:
        text = "profile: p\nprofiles:\n  p: {pipeline: {stability_seeds: 0}}"
        assert parse_config(text, HERE).pipeline.stability_seeds == 0

    def test_profile_overrides_figure_flags(self) -> None:
        text = BASE + "profile: p\nprofiles:\n  p: {figures: {show: true}}"
        assert parse_config(text, HERE).figures == FigureSettings(save=True, show=True)

    def test_argument_overrides_the_file(self) -> None:
        text = BASE + "profile: a\nprofiles:\n  a: {pipeline: {seed: 2}}\n  b: {pipeline: {seed: 3}}"
        assert parse_config(text, HERE).pipeline.seed == Seed(2)
        assert parse_config(text, HERE, "b").pipeline.seed == Seed(3)
        assert parse_config(text, HERE, "default").pipeline.seed == Seed(1)

    def test_empty_profile_entry_is_allowed(self) -> None:
        assert parse_config(BASE + "profile: p\nprofiles:\n  p:", HERE).profile == "p"

    def test_profile_name_reaches_the_config(self) -> None:
        text = BASE + "profile: fast\nprofiles:\n  fast: {}"
        assert parse_config(text, HERE).profile == "fast"

    def test_unknown_profile_lists_the_choices(self) -> None:
        text = BASE + "profile: fsat\nprofiles:\n  fast: {}\n  colab: {}"
        assert problems(text) == ["unknown profile 'fsat' (available: colab, default, fast)"]

    def test_profile_must_be_a_name(self) -> None:
        assert problems("profile: [a]") == ["`profile` must be a profile name, got ['a']"]

    @pytest.mark.parametrize("profiles", ["[a, b]", "{a: 1}"])
    def test_profiles_must_map_names_to_settings(self, profiles: str) -> None:
        assert problems(f"profiles: {profiles}") == ["`profiles` must map profile names to settings"]

    @pytest.mark.parametrize("key", ["profile", "profiles"])
    def test_profiles_cannot_nest(self, key: str) -> None:
        text = f"profile: p\nprofiles:\n  p: {{{key}: x}}"
        assert problems(text) == [f"`{key}` cannot be set inside a profile"]

    def test_errors_inside_a_profile_name_it(self) -> None:
        text = BASE + "profile: p\nprofiles:\n  p: {pipeline: {sed: 1}}"
        with pytest.raises(ConfigError) as caught:
            parse_config(text, HERE)
        assert caught.value.profile == "p"
        assert str(caught.value).startswith(f"invalid configuration {HERE} (profile 'p'):")
        assert caught.value.problems == ["unknown key(s): sed @ $.pipeline"]


class TestMerge:
    def test_recursive(self) -> None:
        assert merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}

    def test_replaces_scalars_and_lists(self) -> None:
        assert merge({"a": [1, 2], "b": 1}, {"a": [3], "b": None}) == {"a": [3], "b": None}

    def test_adds_new_keys(self) -> None:
        assert merge({"a": 1}, {"b": {"c": 2}}) == {"a": 1, "b": {"c": 2}}

    def test_does_not_mutate_inputs(self) -> None:
        base, override = {"a": {"b": 1}}, {"a": {"b": 2}}
        merge(base, override)
        assert base == {"a": {"b": 1}}
        assert override == {"a": {"b": 2}}


class TestDefaults:
    def test_empty_file_gives_all_defaults(self) -> None:
        config = parse_config("", HERE)
        assert config.profile == "default"
        assert config.figures == FigureSettings()
        assert config.pipeline == PipelineSettings()
        assert config.cars == ()

    def test_partial_sections_keep_other_defaults(self) -> None:
        config = parse_config("pipeline: {seed: 7}\nfigures: {show: true}", HERE)
        assert config.pipeline.seed == Seed(7)
        assert config.pipeline.fractions == PipelineSettings().fractions
        assert config.figures == FigureSettings(save=True, show=True)


class TestStructure:
    def test_yaml_null_means_unlimited_depth(self) -> None:
        config = parse_config(
            "pipeline: {grid: {max_depth: [4, null], class_weight: [null, balanced]}}", HERE
        )
        assert config.pipeline.grid.max_depth == (4, None)
        assert config.pipeline.grid.class_weight == (None, "balanced")

    def test_feature_sets_by_name(self) -> None:
        config = parse_config("pipeline: {feature_sets: [with_concepts]}", HERE)
        assert config.pipeline.feature_sets == (FeatureSet.WITH_CONCEPTS,)

    def test_numeric_yaml_levels_become_strings(self) -> None:
        config = parse_config(
            "cars: [{buying: low, maint: low, doors: 2, persons: 4, lug_boot: big, safety: high}]", HERE
        )
        assert config.cars == (Car("low", "low", "2", "4", "big", "high"),)


class TestLoggingSettings:
    def test_default(self) -> None:
        assert parse_config("", HERE).logging == LogSettings(LogLevel.INFO, save=True)

    def test_from_yaml(self) -> None:
        config = parse_config("logging: {level: debug, save: false}", HERE)
        assert config.logging == LogSettings(LogLevel.DEBUG, save=False)

    def test_shipped_config_logs_info_to_a_file(self, configs_dir: Path) -> None:
        assert load_config(configs_dir / "config.yaml").logging == LogSettings(LogLevel.INFO, save=True)

    def test_unknown_level_is_reported(self) -> None:
        (message,) = problems("logging: {level: verbose}")
        assert "verbose" in message
        assert "$.logging.level" in message


class TestErrors:
    def test_unknown_keys_are_rejected_with_location(self) -> None:
        assert problems("pipeline: {sed: 1}") == ["unknown key(s): sed @ $.pipeline"]

    @pytest.mark.parametrize("section", ["data: {path: car.data}", "output: {directory: out}"])
    def test_paths_are_not_config_settings(self, section: str) -> None:
        (message,) = problems(section)
        assert message.startswith("unknown key(s): ")

    def test_core_validation_messages_are_passed_through(self) -> None:
        (message,) = problems("pipeline: {fractions: {train: 0.9}}")
        assert "sum to 1" in message
        assert "$.pipeline.fractions" in message

    def test_invalid_car_names_the_bad_level(self) -> None:
        (message,) = problems(
            "cars: [{buying: cheap, maint: low, doors: 4, persons: 4, lug_boot: big, safety: high}]"
        )
        assert "buying='cheap'" in message
        assert "$.cars[0]" in message

    def test_every_missing_car_field_is_listed(self) -> None:
        assert len(problems("cars: [{buying: low}]")) == 5

    def test_invalid_literal(self) -> None:
        (message,) = problems("pipeline: {grid: {criterion: [log_loss]}}")
        assert "log_loss" in message
        assert "$.pipeline.grid.criterion[0]" in message

    def test_wrong_scalar_type(self) -> None:
        (message,) = problems("pipeline: {stability_seeds: many}")
        assert "expected int" in message

    def test_several_problems_are_reported_together(self) -> None:
        assert len(problems("pipeline: {seed: x, stability_seeds: y}")) == 2

    @pytest.mark.parametrize("text", ["- a\n- b", "just a string", "42"])
    def test_top_level_must_be_a_mapping(self, text: str) -> None:
        assert problems(text) == ["the top level must be a mapping of section names to settings"]

    def test_invalid_yaml(self) -> None:
        (message,) = problems("pipeline: [")
        assert message.startswith("not valid YAML")

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="cannot read file"):
            load_config(tmp_path / "absent.yaml")

    def test_error_message_lists_problems(self, write_config: Callable[[str], Path]) -> None:
        path = write_config("pipeline: {sed: 1}")
        with pytest.raises(ConfigError) as caught:
            load_config(path)
        assert str(caught.value).splitlines() == [
            f"invalid configuration {path.resolve()}:",
            "  - unknown key(s): sed @ $.pipeline",
        ]
