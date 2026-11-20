from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

from knn_eval_core.dataset import CarDataset
from knn_eval_core.logs import Stage, log_every, logged_stage

LOGGER = logging.getLogger("knn_eval_core.test")


class TestStage:
    def test_label(self) -> None:
        assert Stage(3, 7, "pruning").label == "[3/7] pruning"

    @pytest.mark.parametrize(("number", "total"), [(0, 3), (4, 3), (1, 0)])
    def test_number_must_lie_in_range(self, number: int, total: int) -> None:
        with pytest.raises(ValueError, match="outside"):
            Stage(number, total, "x")


class TestLoggedStage:
    def test_logs_start_and_duration(self, record_logs: Callable[..., Any]) -> None:
        with record_logs() as logs, logged_stage(LOGGER, Stage(2, 5, "fitting")):
            LOGGER.info("inside")
        messages = logs.messages()
        assert messages[:2] == ["[2/5] fitting ...", "inside"]
        assert messages[2].startswith("[2/5] fitting done in ")
        assert messages[2].endswith(" s")

    def test_no_done_line_when_the_stage_fails(self, record_logs: Callable[..., Any]) -> None:
        with record_logs() as logs, pytest.raises(RuntimeError), logged_stage(LOGGER, Stage(1, 1, "x")):
            raise RuntimeError("boom")
        assert logs.messages() == ["[1/1] x ..."]


class TestLogEvery:
    def test_twenty_items_log_ten_times_including_the_last(self) -> None:
        logged = [i for i in range(1, 21) if log_every(i, 20)]
        assert logged == [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    def test_short_loops_log_every_item(self) -> None:
        assert [i for i in range(1, 4) if log_every(i, 3)] == [1, 2, 3]

    def test_last_item_always_logged(self) -> None:
        assert log_every(23, 23)

    @pytest.mark.parametrize(("index", "total"), [(0, 5), (1, 0), (-1, 5)])
    def test_out_of_range_is_never_logged(self, index: int, total: int) -> None:
        assert not log_every(index, total)


class TestDatasetLogging:
    def test_clean_data_logs_info_only(
        self, raw_frame: pd.DataFrame, record_logs: Callable[..., Any]
    ) -> None:
        with record_logs(logging.DEBUG) as logs:
            CarDataset.from_frame(raw_frame)
        assert [(r.levelname, r.getMessage()) for r in logs.records] == [
            ("INFO", "validated 1728 rows, kept 1728")
        ]

    def test_dropped_rows_are_a_warning(
        self, raw_frame: pd.DataFrame, record_logs: Callable[..., Any]
    ) -> None:
        dirty = pd.concat([raw_frame.head(3), raw_frame.head(1)], ignore_index=True)
        dirty.iloc[2, 0] = "cheap"
        with record_logs() as logs:
            CarDataset.from_frame(dirty)
        warning = next(r for r in logs.records if r.levelno == logging.WARNING)
        assert warning.getMessage() == (
            "dropped 2 of 4 rows: 0 with missing values, 1 with unknown levels, "
            "1 exact duplicates, 0 with conflicting labels"
        )

    def test_report_is_logged_before_rejecting_empty_data(
        self, raw_frame: pd.DataFrame, record_logs: Callable[..., Any]
    ) -> None:
        empty = raw_frame.head(2).assign(safety="unknown")
        with record_logs() as logs, pytest.raises(ValueError, match="no valid rows"):
            CarDataset.from_frame(empty)
        assert any("2 with unknown levels" in m for m in logs.messages())
