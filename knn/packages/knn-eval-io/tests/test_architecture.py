"""Layering and isolation rules, checked from source so CI catches violations.

The k-NN application must stay independent of the decision tree application: neither imports
the other, and the k-NN workspace does not depend on the tree's packages.

knn-eval-core (backend) holds the logic and knows nothing about files, consoles or formats.
knn-eval-io (application layer) imports the backend's classes and handles everything else.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

PACKAGES = Path(__file__).resolve().parents[2]
CORE = PACKAGES / "knn-eval-core"
IO = PACKAGES / "knn-eval-io"
CORE_SOURCES = sorted((CORE / "src" / "knn_eval_core").glob("*.py"))
IO_SOURCES = sorted((IO / "src" / "knn_eval_io").glob("*.py"))
KNN_ROOT = PACKAGES.parent
TREE_SOURCES = sorted((KNN_ROOT.parent / "packages").glob("car-eval-*/src/car_eval_*/*.py"))

# Modules of the application layer that turn backend objects into, or out of, files and text.
IO_MODULES_USING_BACKEND = {"app", "artifacts", "config", "converters", "plots", "report", "sources"}
PRESENTATION_LIBRARIES = {"knn_eval_io", "cattrs", "matplotlib", "yaml"}


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


@pytest.mark.parametrize("path", CORE_SOURCES, ids=lambda p: p.name)
def test_backend_logs_but_never_configures_logging(path: Path) -> None:
    calls = {
        node.func.attr
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not calls & {"basicConfig", "addHandler", "setLevel", "captureWarnings", "dictConfig"}


@pytest.mark.parametrize("module", sorted(IO_MODULES_USING_BACKEND))
def test_application_layer_builds_on_backend_classes(module: str) -> None:
    imports = imported_modules(IO / "src" / "knn_eval_io" / f"{module}.py")
    assert any(name.startswith("knn_eval_core.") for name in imports)


def test_entry_point_reaches_the_backend_only_through_the_application_layer() -> None:
    imports = imported_modules(IO / "src" / "knn_eval_io" / "__main__.py")
    assert not any(name.startswith("knn_eval_core") for name in imports)
    assert {"knn_eval_io", "knn_eval_io.config", "knn_eval_io.paths", "knn_eval_io.sources"} <= imports
    assert "knn_eval_io.logging_setup" in imports


def test_declared_dependencies_point_one_way() -> None:
    core = dependencies(CORE)
    application = dependencies(IO)
    assert "knn-eval-core" in application
    assert not any(dep.startswith(("knn-eval-io", "cattrs", "matplotlib", "pyyaml")) for dep in core)


@pytest.mark.parametrize("path", CORE_SOURCES + IO_SOURCES, ids=lambda p: p.name)
def test_knn_never_imports_the_decision_tree(path: Path) -> None:
    assert not any(
        module.split(".")[0] in {"car_eval_core", "car_eval_io"} for module in imported_modules(path)
    )


def test_decision_tree_never_imports_knn() -> None:
    assert TREE_SOURCES, "decision tree sources not found next to the knn folder"
    for path in TREE_SOURCES:
        assert not any(module.startswith("knn_eval") for module in imported_modules(path)), path


def test_knn_workspace_is_separate_from_the_tree_workspace() -> None:
    with (KNN_ROOT / "pyproject.toml").open("rb") as handle:
        knn = tomllib.load(handle)
    with (KNN_ROOT.parent / "pyproject.toml").open("rb") as handle:
        tree = tomllib.load(handle)
    assert knn["tool"]["uv"]["workspace"]["members"] == ["packages/knn-eval-core", "packages/knn-eval-io"]
    assert not any("knn" in member for member in tree["tool"]["uv"]["workspace"]["members"])
    every_dependency = dependencies(CORE) + dependencies(IO) + knn["project"]["dependencies"]
    assert not any(dep.startswith("car-eval") for dep in every_dependency)
