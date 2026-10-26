"""Output locations for a command, built from the command line (never from the YAML config)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    """Every file a command writes lives under one output folder.

    The folder is stored as an absolute path (a relative one is taken from the current working
    directory), so log messages always say exactly where the results are.
    """

    output: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", self.output.expanduser().resolve())
        if self.output.exists() and not self.output.is_dir():
            raise ValueError(f"output path exists and is not a folder: {self.output}")

    def create(self) -> Path:
        """Make the output folder (and its parents) if needed; returns it."""
        self.output.mkdir(parents=True, exist_ok=True)
        return self.output

    @property
    def report(self) -> Path:
        return self.output / "report.txt"

    @property
    def results(self) -> Path:
        return self.output / "results.json"

    @property
    def model(self) -> Path:
        return self.output / "model.json"

    @property
    def explanations(self) -> Path:
        return self.output / "explanations.json"

    @property
    def log(self) -> Path:
        return self.output / "car-eval.log"

    @property
    def figures(self) -> Path:
        return self.output / "figures"
