"""Where the raw dataset comes from (a tagged union) and how it is read."""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, assert_never

import pandas as pd

from car_eval_core.dataset import COLUMN_NAMES

logger = logging.getLogger(__name__)

UCI_ZIP_URL: Final = "https://archive.ics.uci.edu/static/public/19/car+evaluation.zip"


@dataclass(frozen=True)
class LocalFile:
    """A `car.data` file on disk (or on a mounted Google Drive)."""

    path: Path


@dataclass(frozen=True)
class UciDownload:
    """The official zip archive from the UCI Machine Learning Repository."""

    url: str = UCI_ZIP_URL
    member: str = "car.data"
    timeout_seconds: float = 60.0


DataSource = LocalFile | UciDownload

_READ_OPTIONS: Final[dict[str, Any]] = {
    "header": None,
    "names": list(COLUMN_NAMES),
    "dtype": str,
    "skipinitialspace": True,
}


def read_raw(source: DataSource) -> pd.DataFrame:
    """Raw string table with columns `COLUMN_NAMES`; validation happens in the core package."""
    match source:
        case LocalFile(path):
            if not path.is_file():
                raise FileNotFoundError(f"data file not found: {path}")
            return pd.read_csv(path, **_READ_OPTIONS)
        case UciDownload(url, member, timeout):
            logger.info("downloading %s (timeout %.0f s)", url, timeout)
            with urllib.request.urlopen(url, timeout=timeout) as response:
                archive = zipfile.ZipFile(io.BytesIO(response.read()))
            with archive.open(member) as handle:
                return pd.read_csv(handle, **_READ_OPTIONS)
        case _:
            assert_never(source)


def describe(source: DataSource) -> str:
    match source:
        case LocalFile(path):
            return f"local file {path}"
        case UciDownload(url):
            return f"UCI download {url}"
        case _:
            assert_never(source)
