from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from car_eval_core.dataset import COLUMN_NAMES
from car_eval_io.sources import LocalFile, UciDownload, describe, read_raw


class FakeResponse(io.BytesIO):
    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def zipped(member: str, text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(member, text)
    return buffer.getvalue()


class TestLocalFile:
    def test_reads_raw_strings_with_named_columns(self, car_data: Path) -> None:
        frame = read_raw(LocalFile(car_data))
        assert tuple(frame.columns) == COLUMN_NAMES
        assert len(frame) == 1728
        assert frame.iloc[0].tolist() == ["vhigh", "vhigh", "2", "2", "small", "low", "unacc"]
        assert all(pd.api.types.is_string_dtype(dtype) for dtype in frame.dtypes)

    def test_keeps_numeric_looking_levels_as_text(self, tmp_path: Path) -> None:
        path = tmp_path / "car.data"
        path.write_text("low, low, 2, 4, big, high, acc\n", encoding="utf-8")
        assert read_raw(LocalFile(path)).iloc[0].tolist() == ["low", "low", "2", "4", "big", "high", "acc"]

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="data file not found"):
            read_raw(LocalFile(tmp_path / "nope.data"))


class TestUciDownload:
    def test_reads_the_member_from_the_archive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        requested: dict[str, Any] = {}

        def fake_urlopen(url: str, timeout: float) -> FakeResponse:
            requested.update(url=url, timeout=timeout)
            return FakeResponse(zipped("car.data", "low,low,2,2,small,low,unacc\n"))

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        frame = read_raw(UciDownload(url="https://example.test/car.zip", timeout_seconds=3.0))
        assert requested == {"url": "https://example.test/car.zip", "timeout": 3.0}
        assert frame.iloc[0, -1] == "unacc"

    def test_missing_member(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: FakeResponse(zipped("other", "")))
        with pytest.raises(KeyError):
            read_raw(UciDownload())

    def test_defaults_point_at_uci(self) -> None:
        assert UciDownload().url.startswith("https://archive.ics.uci.edu/")
        assert UciDownload().member == "car.data"


def test_describe() -> None:
    assert describe(LocalFile(Path("/d/car.data"))) == "local file /d/car.data"
    assert describe(UciDownload(url="u")) == "UCI download u"
