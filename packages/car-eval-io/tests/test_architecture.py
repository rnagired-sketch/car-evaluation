"""Layering rules between the two uv packages, checked from source so CI catches violations.

car-eval-core (backend) holds the logic and knows nothing about files, consoles or formats.
car-eval-io (application layer) imports the backend's classes and handles everything else.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

PACKAGES = Path(__file__).resolve().parents[2]
CORE = PACKAGES / "car-eval-core"
IO = PACKAGES / "car-eval-io"
CORE_SOURCES = sorted((CORE / "src" / "car_eval_core").glob("*.py"))
IO_SOURCES = sorted((IO / "src" / "car_eval_io").glob("*.py"))

# Modules of the application layer that turn backend objects into, or out of, files and text.
IO_MODULES_USING_BACKEND = {"app", "artifacts", "config", "converters", "plots", "report", "sources"}
PRESENTATION_LIBRARIES = {"car_eval_io", "cattrs", "matplotlib", "yaml"}


def imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def called_names(path: Path) -> set[str]:
    return {
        node.func.id
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def dependencies(project: Path) -> list[str]:
    with (project / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["dependencies"]


def test_both_packages_have_sources() -> None:
    assert len(CORE_SOURCES) >= 10
    assert len(IO_SOURCES) >= 9


@pytest.mark.parametrize("path", CORE_SOURCES, ids=lambda p: p.name)
def test_backend_never_imports_the_application_layer(path: Path) -> None:
    roots = {module.split(".")[0] for module in imported_modules(path)}
    assert not roots & PRESENTATION_LIBRARIES


@pytest.mark.parametrize("path", CORE_SOURCES, ids=lambda p: p.name)
def test_backend_does_no_console_or_file_io(path: Path) -> None:
    assert not called_names(path) & {"print", "open", "input"}


@pytest.mark.parametrize("module", sorted(IO_MODULES_USING_BACKEND))
def test_application_layer_builds_on_backend_classes(module: str) -> None:
    imports = imported_modules(IO / "src" / "car_eval_io" / f"{module}.py")
    assert any(name.startswith("car_eval_core.") for name in imports)


def test_entry_point_reaches_the_backend_only_through_the_application_layer() -> None:
    imports = imported_modules(IO / "src" / "car_eval_io" / "__main__.py")
    assert not any(name.startswith("car_eval_core") for name in imports)
    assert {"car_eval_io", "car_eval_io.config", "car_eval_io.paths", "car_eval_io.sources"} <= imports


def test_declared_dependencies_point_one_way() -> None:
    core = dependencies(CORE)
    application = dependencies(IO)
    assert "car-eval-core" in application
    assert not any(dep.startswith(("car-eval-io", "cattrs", "matplotlib", "pyyaml")) for dep in core)
