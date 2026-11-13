"""CLI wiring of ``etfcorr returns`` (step 2.1; needs `typer`, skipped where missing).

Returns logic is tested in ``test_returns.py``; here only the command line: that the command is
listed, exit codes, and that a missing clean panel surfaces the same ``ConfigError`` message as
calling ``command_returns`` directly.
"""

import importlib.util
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]

# A full run reads the real Parquet clean panel (`download.py`'s default reader/writer, reused
# by `command_returns`); the planning sandbox has neither `pyarrow` nor `fastparquet` installed
# (PyPI unreachable there). NOT EXECUTED OFFLINE, confirmed at Gate 2 where pyarrow is a real
# dependency (pyproject.toml). Only the missing-cache and bad-config paths avoid the Parquet
# boundary entirely and are not skipped.
needs_parquet = pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None and importlib.util.find_spec("fastparquet") is None,
    reason="no Parquet engine installed (offline sandbox)",
)

typer_testing = pytest.importorskip("typer.testing")

from etfcorr import cli  # noqa: E402  (importing the CLI needs typer, hence after the guard)

UNIVERSE = {
    "version": 1,
    "start_date": "2014-01-01",
    "min_adv_usd": 10_000_000,
    "tickers": [
        {"ticker": "AAA", "sector": "ai_semis", "panel": "core", "name": "Alpha"},
        {"ticker": "SPY", "sector": "control", "panel": "control", "name": "S&P"},
    ],
}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    shutil.copytree(ROOT / "config", tmp_path / "config")
    (tmp_path / "config" / "universe.yaml").write_text(yaml.safe_dump(UNIVERSE))
    return tmp_path


def invoke(*args: str) -> Any:
    return typer_testing.CliRunner().invoke(cli.app, list(args))


def test_returns_command_is_listed() -> None:
    assert "returns" in invoke("--help").output


def test_returns_needs_a_valid_config(tmp_path: Path) -> None:
    result = invoke("returns", "--root", str(tmp_path))
    assert result.exit_code == 1
    assert "CONFIG ERROR" in result.output


def test_returns_missing_clean_panel_exits_1(repo: Path) -> None:
    result = invoke("returns", "--root", str(repo))
    assert result.exit_code == 1
    assert "CONFIG ERROR" in result.output
    assert "data build" in result.output


@needs_parquet
def test_returns_runs_and_writes_a_file_per_horizon(repo: Path) -> None:
    idx = pd.bdate_range("2016-01-04", "2016-06-30")
    n = len(idx)
    prices = 100.0 + np.arange(n, dtype=float)
    frames = {
        t: pd.DataFrame(
            {"close": prices, "volume": 1_000_000.0, "adj_close": prices},
            index=idx,
        )
        for t in ("AAA", "SPY")
    }
    panel = pd.concat(frames, axis=1)
    panel.columns = panel.columns.set_names(["ticker", "field"])
    (repo / "data" / "interim").mkdir(parents=True, exist_ok=True)
    panel.to_parquet(repo / "data" / "interim" / "prices_clean.parquet")

    result = invoke("returns", "--root", str(repo))
    assert result.exit_code == 0, result.output
    for horizon in ("D", "W", "M", "Q"):
        assert (repo / "data" / "processed" / f"returns_{horizon}.parquet").is_file()
