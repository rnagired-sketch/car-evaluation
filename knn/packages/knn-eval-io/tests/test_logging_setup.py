from __future__ import annotations

import io
import logging
import re
import warnings
from pathlib import Path

import pytest

from knn_eval_io.logging_setup import LogLevel, configure_logging

LINE = re.compile(r"^\d{2}:\d{2}:\d{2} (INFO|WARNING|ERROR|DEBUG) +[\w.]+: .+$")


def ours() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if getattr(h, "_knn_eval_handler", False)]


class TestConfigureLogging:
    def test_console_lines_are_timestamped_with_level_and_logger(self) -> None:
        stream = io.StringIO()
        configure_logging(LogLevel.INFO, stream=stream)
        logging.getLogger("knn_eval_core.pipeline").info("[1/7] splitting data ...")
        (line,) = stream.getvalue().splitlines()
        assert LINE.match(line)
        assert line.endswith("INFO    knn_eval_core.pipeline: [1/7] splitting data ...")

    @pytest.mark.parametrize(
        ("level", "shown"),
        [
            (LogLevel.DEBUG, ["d", "i", "w", "e"]),
            (LogLevel.INFO, ["i", "w", "e"]),
            (LogLevel.WARNING, ["w", "e"]),
            (LogLevel.ERROR, ["e"]),
        ],
    )
    def test_level_filters_the_project_loggers(self, level: LogLevel, shown: list[str]) -> None:
        stream = io.StringIO()
        configure_logging(level, stream=stream)
        logger = logging.getLogger("knn_eval_io.app")
        logger.debug("d")
        logger.info("i")
        logger.warning("w")
        logger.error("e")
        assert [line.rsplit(": ", 1)[1] for line in stream.getvalue().splitlines()] == shown

    def test_third_party_loggers_show_warnings_only(self) -> None:
        stream = io.StringIO()
        configure_logging(LogLevel.DEBUG, stream=stream)
        logging.getLogger("matplotlib.font_manager").info("font cache")
        logging.getLogger("sklearn").warning("convergence")
        assert stream.getvalue().count("\n") == 1
        assert "convergence" in stream.getvalue()

    def test_python_warnings_become_log_records(self) -> None:
        stream = io.StringIO()
        configure_logging(LogLevel.INFO, stream=stream)
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn("y_pred contains classes not in y_true", UserWarning, stacklevel=1)
        assert "WARNING py.warnings:" in stream.getvalue()

    def test_reconfiguring_replaces_instead_of_duplicating(self) -> None:
        first, second = io.StringIO(), io.StringIO()
        configure_logging(LogLevel.INFO, stream=first)
        configure_logging(LogLevel.INFO, stream=second)
        assert len(ours()) == 1
        logging.getLogger("knn_eval_io").info("once")
        assert first.getvalue() == ""
        assert second.getvalue().count("once") == 1

    def test_leaves_other_handlers_alone(self) -> None:
        foreign = logging.NullHandler()
        logging.getLogger().addHandler(foreign)
        try:
            configure_logging(LogLevel.INFO, stream=io.StringIO())
            configure_logging(LogLevel.INFO, stream=io.StringIO())
            assert foreign in logging.getLogger().handlers
        finally:
            logging.getLogger().removeHandler(foreign)


class TestLogFile:
    def test_file_receives_the_same_lines_and_folders_are_created(self, tmp_path: Path) -> None:
        log = tmp_path / "new" / "knn-eval.log"
        stream = io.StringIO()
        configure_logging(LogLevel.INFO, log, stream=stream)
        logging.getLogger("knn_eval_io").warning("disk is slow")
        for handler in ours():
            handler.flush()
        assert log.read_text().splitlines() == stream.getvalue().splitlines()

    def test_file_is_appended_across_commands(self, tmp_path: Path) -> None:
        log = tmp_path / "knn-eval.log"
        for word in ("run", "explain"):
            configure_logging(LogLevel.INFO, log, stream=io.StringIO())
            logging.getLogger("knn_eval_io").info(word)
        configure_logging(LogLevel.INFO, stream=io.StringIO())  # closes the file handler
        assert [line.rsplit(": ", 1)[1] for line in log.read_text().splitlines()] == ["run", "explain"]


def test_level_names_match_logging() -> None:
    assert [level.number for level in LogLevel] == [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
    ]
