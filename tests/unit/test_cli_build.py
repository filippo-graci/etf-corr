"""CLI wiring of ``etfcorr data build`` (step 1.3; needs `typer`, skipped where missing).

Validation/build logic is tested in ``test_validate.py``/``test_build_panel.py``; here only the
command line: option parsing, exit codes, and that a validation failure surfaces as exit 1 with
the full report, not just a one-line exception message.
"""

import importlib.util
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]

# A full run reads the real Parquet raw cache (`download.py`'s default) and, for any ticker
# present in the panel, calls the real `pandas_market_calendars`-backed `nyse_sessions` inside
# `validate_panel`'s staleness/missing-pct checks -- the CLI never overrides `sessions_fn` with a
# fixture calendar, unlike the unit tests. The planning sandbox has neither a Parquet engine nor
# `pandas_market_calendars` (PyPI unreachable there). NOT EXECUTED OFFLINE, confirmed at Gate 2
# where both are real dependencies (pyproject.toml). Only the bad-config/date and
# missing-raw-cache paths avoid both and are not skipped: they fail before either is touched.
needs_full_stack = pytest.mark.skipif(
    (
        importlib.util.find_spec("pyarrow") is None
        and importlib.util.find_spec("fastparquet") is None
    )
    or importlib.util.find_spec("pandas_market_calendars") is None,
    reason="no Parquet engine and/or pandas_market_calendars installed (offline sandbox)",
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
ASOF = date(2024, 1, 10)
DAYS = pd.bdate_range("2024-01-02", ASOF)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    shutil.copytree(ROOT / "config", tmp_path / "config")
    (tmp_path / "config" / "universe.yaml").write_text(yaml.safe_dump(UNIVERSE))
    return tmp_path


def _panel(tickers_close: dict[str, float]) -> pd.DataFrame:
    frames = {
        t: pd.DataFrame({"close": c, "volume": 1_000_000.0, "adj_close": c}, index=DAYS)
        for t, c in tickers_close.items()
    }
    panel = pd.concat(frames, axis=1)
    panel.columns = panel.columns.set_names(["ticker", "field"])
    return panel


def invoke(*args: str) -> Any:
    return typer_testing.CliRunner().invoke(cli.app, list(args))


def test_build_group_is_listed_and_has_the_command() -> None:
    assert "build" in invoke("data", "--help").output


@needs_full_stack
def test_build_runs_and_writes_the_clean_panel_and_coverage(repo: Path) -> None:
    (repo / "data" / "raw").mkdir(parents=True, exist_ok=True)
    _panel({"AAA": 100.0, "SPY": 400.0}).to_parquet(repo / "data" / "raw" / "prices_yf.parquet")
    result = invoke("data", "build", "--root", str(repo), "--asof", "2024-01-10")
    assert result.exit_code == 0, result.output
    assert "0 error(s)" in result.output
    assert (repo / "data" / "interim" / "prices_clean.parquet").is_file()
    assert (repo / "data" / "processed" / "coverage.csv").is_file()


def test_build_needs_a_valid_config(tmp_path: Path) -> None:
    result = invoke("data", "build", "--root", str(tmp_path), "--asof", "2024-01-10")
    assert result.exit_code == 1
    assert "CONFIG ERROR" in result.output


def test_build_bad_date_exits_1(repo: Path) -> None:
    result = invoke("data", "build", "--root", str(repo), "--asof", "10/01/2024")
    assert result.exit_code == 1
    assert "CONFIG ERROR" in result.output
    assert "not an ISO date" in result.output


def test_build_missing_raw_cache_exits_1(repo: Path) -> None:
    result = invoke("data", "build", "--root", str(repo), "--asof", "2024-01-10")
    assert result.exit_code == 1
    assert "CONFIG ERROR" in result.output
    assert "data download" in result.output


@needs_full_stack
def test_build_validation_failure_exits_1_with_the_full_report(repo: Path) -> None:
    (repo / "data" / "raw").mkdir(parents=True, exist_ok=True)
    _panel({"AAA": 100.0}).to_parquet(repo / "data" / "raw" / "prices_yf.parquet")  # SPY missing
    result = invoke("data", "build", "--root", str(repo), "--asof", "2024-01-10")
    assert result.exit_code == 1
    assert "DATA QUALITY ERROR" in result.output
    assert "SPY" in result.output
    assert "absent from the panel" in result.output
    assert not (repo / "data" / "interim" / "prices_clean.parquet").exists()
