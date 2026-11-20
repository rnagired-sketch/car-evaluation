"""cattrs converter: builds typed objects from YAML/JSON data and turns results back into data.

Construction goes through each class's own `__init__`/`__post_init__`, so the validation
rules of the core package apply to everything read from a file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import cattrs
import numpy as np
from cattrs.errors import CattrsError
from cattrs.preconf.pyyaml import PyyamlConverter, make_converter
from cattrs.strategies import configure_tagged_union

from knn_eval_core.domain import Acceptability
from knn_eval_io.sources import DataSource, LocalFile, UciDownload

SOURCE_TAGS: Final[dict[type, str]] = {LocalFile: "local", UciDownload: "uci"}


def build_converter() -> PyyamlConverter:
    converter = make_converter(forbid_extra_keys=True)

    converter.register_structure_hook(Path, lambda value, _: Path(value))
    converter.register_unstructure_hook(Path, str)

    # Class labels are written as words ('vgood'), never as their integer codes.
    converter.register_structure_hook(Acceptability, _structure_acceptability)
    converter.register_unstructure_hook(Acceptability, lambda a: a.label)

    converter.register_unstructure_hook(np.ndarray, lambda array: array.tolist())
    # Distances and weights are plain numbers in JSON.
    converter.register_unstructure_hook(np.floating, float)

    # The data source recorded in model.json: `{kind: local, path: ...}` or `{kind: uci, ...}`
    configure_tagged_union(DataSource, converter, tag_name="kind", tag_generator=SOURCE_TAGS.__getitem__)
    return converter


def _structure_acceptability(value: Any, _: type) -> Acceptability:
    if isinstance(value, str):
        return Acceptability.from_label(value)
    return Acceptability(int(value))


CONVERTER: Final = build_converter()


def describe_errors(error: Exception) -> list[str]:
    """One readable line per problem, each with the path into the document where it occurred."""
    return cattrs.transform_error(error, format_exception=_format_exception)


def _format_exception(exc: BaseException, type_: Any) -> str:
    if isinstance(exc, cattrs.ForbiddenExtraKeysError):
        return f"unknown key(s): {', '.join(sorted(exc.extra_fields))}"
    if isinstance(exc, KeyError) and type_ == DataSource:
        kinds = ", ".join(SOURCE_TAGS.values())
        if exc.args == ("kind",):
            return f"missing 'kind' (one of: {kinds})"
        return f"unknown kind {exc.args[0]!r} (one of: {kinds})"
    if isinstance(exc, KeyError):
        return "required key is missing"
    if isinstance(exc, ValueError | TypeError):
        expected = f" (expected {type_.__name__})" if isinstance(type_, type) else ""
        return f"{exc}{expected}"
    if isinstance(exc, CattrsError):
        return str(exc)
    return cattrs.v.format_exception(exc, type_)
