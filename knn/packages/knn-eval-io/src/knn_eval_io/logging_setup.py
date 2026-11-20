"""Where log records go and how much is shown.

Both packages log under their own names (`knn_eval_core.*`, `knn_eval_io.*`). This module
installs the handlers once per command: a console handler on standard error and, when a
command has an output folder, a log file there. Standard output stays reserved for the
report itself, so `knn-eval run ... > report.txt` captures the report without log lines.

Third-party libraries (scikit-learn, matplotlib) show WARNING and above only; Python
warnings they raise are routed through logging as well.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final, TextIO

PACKAGE_LOGGERS: Final[Sequence[str]] = ("knn_eval_core", "knn_eval_io")
LOG_FORMAT: Final = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT: Final = "%H:%M:%S"
_INSTALLED: Final = "_knn_eval_handler"  # marks handlers this module owns


class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    @property
    def number(self) -> int:
        return logging.getLevelNamesMapping()[self.name]


@dataclass(frozen=True)
class LogSettings:
    level: LogLevel = LogLevel.INFO  # how much of the project's own logging to show
    save: bool = True  # also write <output>/knn-eval.log when the command has an output folder


def configure_logging(
    level: LogLevel = LogLevel.INFO, log_file: Path | None = None, stream: TextIO | None = None
) -> list[logging.Handler]:
    """Install (or replace) this project's handlers; safe to call more than once per process.

    The log file is opened in append mode so `run` and later `explain` calls share one history.
    """
    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, _INSTALLED, False)]:
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    handlers: list[logging.Handler] = [logging.StreamHandler(stream if stream is not None else sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        setattr(handler, _INSTALLED, True)
        root.addHandler(handler)

    root.setLevel(logging.WARNING)  # third-party libraries: warnings and errors only
    for name in PACKAGE_LOGGERS:
        logging.getLogger(name).setLevel(level.number)
    logging.captureWarnings(True)
    return handlers
