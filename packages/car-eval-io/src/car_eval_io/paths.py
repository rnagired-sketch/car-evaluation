"""Output locations for a command, built from the command line (never from the YAML config)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    """Every file a command writes lives under one output folder."""

    output: Path

    def __post_init__(self) -> None:
        if self.output.exists() and not self.output.is_dir():
            raise ValueError(f"output path exists and is not a folder: {self.output}")

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
