"""Progress logging for the backend.

The backend only emits log records on loggers named `knn_eval_core.<module>`; it never adds
handlers or sets levels. The application layer decides where records go and how much is shown.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    """One numbered step of a longer run, e.g. `[2/6] tuning k-NN on 6 raw attributes`."""

    number: int
    total: int
    title: str

    def __post_init__(self) -> None:
        if not 1 <= self.number <= self.total:
            raise ValueError(f"stage {self.number} is outside 1..{self.total}")

    @property
    def label(self) -> str:
        return f"[{self.number}/{self.total}] {self.title}"


@contextlib.contextmanager
def logged_stage(logger: logging.Logger, stage: Stage) -> Iterator[None]:
    """Log the start of a stage and, when it completes, how long it took."""
    logger.info("%s ...", stage.label)
    started = time.perf_counter()
    yield
    logger.info("%s done in %.1f s", stage.label, time.perf_counter() - started)


def log_every(index: int, total: int, steps: int = 10) -> bool:
    """True for about `steps` evenly spaced items of a loop (and always for the last one).

    Keeps progress lines for long loops readable: a 20-split study logs every 2nd split.
    """
    if total <= 0 or index < 1:
        return False
    stride = max(1, round(total / steps))
    return index == total or index % stride == 0
