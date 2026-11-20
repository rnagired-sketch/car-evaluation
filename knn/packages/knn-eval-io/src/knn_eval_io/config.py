"""The YAML configuration file: what a run does.

The file holds every modelling input: pipeline settings, figure and logging options and the
cars to explain. Where the data comes from and where results go are not part of it; those paths
come from the command line (see `knn_eval_io.paths`).

One file holds the base settings plus named `profiles`. The top-level `profile` key picks
one, and its entries are merged over the base settings before anything is constructed:
mappings merge key by key, while lists and scalars are replaced. The profile `default`
needs no entry and means the base settings unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from knn_eval_core.domain import Car
from knn_eval_core.pipeline import PipelineSettings
from knn_eval_io.converters import CONVERTER, describe_errors
from knn_eval_io.logging_setup import LogSettings

DEFAULT_PROFILE: Final = "default"
PROFILE_KEY: Final = "profile"
PROFILES_KEY: Final = "profiles"


class ConfigError(Exception):
    """The configuration file is missing, is not valid YAML, or does not match the schema."""

    def __init__(self, path: Path, problems: list[str], profile: str | None = None):
        self.path = path
        self.problems = problems
        self.profile = profile
        where = f"{path} (profile {profile!r})" if profile else f"{path}"
        super().__init__(f"invalid configuration {where}:\n" + "\n".join(f"  - {p}" for p in problems))


@dataclass(frozen=True)
class FigureSettings:
    save: bool = True  # write PNGs to <output>/figures
    show: bool = False  # display inline (notebooks)


@dataclass(frozen=True)
class AppConfig:
    profile: str = DEFAULT_PROFILE  # the profile these settings were resolved for
    figures: FigureSettings = FigureSettings()
    logging: LogSettings = LogSettings()
    pipeline: PipelineSettings = PipelineSettings()
    cars: tuple[Car, ...] = ()  # cars to rate and explain, by `run` and by `explain`


def merge(base: Any, override: Any) -> Any:
    """`override` laid over `base`: mappings merge recursively, everything else is replaced."""
    if not (isinstance(base, dict) and isinstance(override, dict)):
        return override
    merged = dict(base)
    for key, value in override.items():
        merged[key] = merge(base[key], value) if key in base else value
    return merged


def select_profile(document: dict[str, Any], source: Path, profile: str | None = None) -> dict[str, Any]:
    """Apply the chosen profile's overrides and drop the profile bookkeeping keys.

    `profile` (from code, e.g. a notebook) takes precedence over the file's `profile` key.
    """
    base = dict(document)
    profiles = base.pop(PROFILES_KEY, None)
    name = profile if profile is not None else base.pop(PROFILE_KEY, DEFAULT_PROFILE)
    base.pop(PROFILE_KEY, None)
    if not isinstance(name, str):
        raise ConfigError(source, [f"`{PROFILE_KEY}` must be a profile name, got {name!r}"])
    if profiles is None:
        profiles = {}
    if not isinstance(profiles, dict) or not all(isinstance(v, dict | None) for v in profiles.values()):
        raise ConfigError(source, [f"`{PROFILES_KEY}` must map profile names to settings"])
    if name not in profiles and name != DEFAULT_PROFILE:
        available = ", ".join(sorted({DEFAULT_PROFILE, *profiles}))
        raise ConfigError(source, [f"unknown profile {name!r} (available: {available})"])
    overrides = profiles.get(name) or {}
    for forbidden in (PROFILE_KEY, PROFILES_KEY):
        if forbidden in overrides:
            raise ConfigError(source, [f"`{forbidden}` cannot be set inside a profile"], name)
    return merge(base, overrides) | {PROFILE_KEY: name}


def parse_config(text: str, source: Path, profile: str | None = None) -> AppConfig:
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigError(source, [f"not valid YAML: {error}"]) from error
    if document is None:
        document = {}
    if not isinstance(document, dict):
        raise ConfigError(source, ["the top level must be a mapping of section names to settings"])
    selected = select_profile(document, source, profile)
    try:
        return CONVERTER.structure(selected, AppConfig)
    except Exception as error:
        name = selected[PROFILE_KEY]
        shown = None if name == DEFAULT_PROFILE else name
        raise ConfigError(source, describe_errors(error), shown) from error


def load_config(path: Path, profile: str | None = None) -> AppConfig:
    """Read a config file; `profile` overrides the file's own `profile` key."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(path, [f"cannot read file: {error.strerror}"]) from error
    return parse_config(text, path.resolve(), profile)
