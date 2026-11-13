"""CLI wiring of ``etfcorr data download`` (step 1.2; needs `typer`, skipped where missing).

Download/caching logic is tested in ``test_download.py``; here only the command line: option
parsing, exit codes, that the fetcher and calendar are injected (never touching the real vendor
or a real calendar), and that the default ``--asof`` resolves through the calendar rather than
falling back to ``date.today()`` (the mistake step 1.1's Gate 2 made for ``check-universe``).
"""

import importlib.util
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from etfcorr.data.universe_check import FetchResult

ROOT = Path(__file__).resolve().parents[2]

# A full run writes a real Parquet cache (etfcorr.data.download.download_universe's own
# default), which needs pyarrow/fastparquet; the planning sandbox has neither (PyPI
# unreachable there). The command-line wiring these tests check is independent of the storage
# engine, so only the tests that reach a successful write are skipped offline; NOT EXECUTED
# OFFLINE, confirmed at Gate 2 where pyarrow is a real dependency (pyproject.toml).
needs_pyarrow = pytest.mark.skipif(
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


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, date, date]]:
    recorded: list[tuple[str, date, date]] = []

    def fake_fetch(ticker: str, start: date, end: date) -> FetchResult:
        recorded.append((ticker, start, end))
        index = pd.bdate_range("2014-01-02", end)
        frame = pd.DataFrame({"close": 100.0, "volume": 200_000.0, "adj_close": 99.0}, index=index)
        return FetchResult(frame, "V", "ETF")

    monkeypatch.setattr(cli, "fetch_yfinance", fake_fetch)
    return recorded


def invoke(*args: str) -> Any:
    return typer_testing.CliRunner().invoke(cli.app, list(args))


def test_download_group_is_listed_and_has_the_command() -> None:
    assert "download" in invoke("data", "--help").output


@needs_pyarrow
def test_download_runs_writes_the_cache_and_uses_the_given_asof(
    repo: Path, calls: list[tuple[str, date, date]]
) -> None:
    result = invoke("data", "download", "--root", str(repo), "--asof", "2024-01-10")
    assert result.exit_code == 0, result.output
    assert "download as of 2024-01-10" in result.output
    assert "panel:" in result.output and "log:" in result.output
    assert (repo / "data" / "raw" / "prices_yf.parquet").is_file()
    assert (repo / "data" / "raw" / "prices_yf.json").is_file()
    assert [c[0] for c in calls] == ["AAA", "SPY"]
    assert all(c[2] == date(2024, 1, 10) for c in calls)


@needs_pyarrow
def test_download_default_asof_goes_through_the_calendar_not_todays_date(
    repo: Path, calls: list[tuple[str, date, date]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``--asof``: the CLI must resolve via ``last_session``, not use an unfinished session."""
    seen: list[date] = []

    def fake_last_session(asof: date) -> date:
        seen.append(asof)
        return date(2024, 1, 5)  # deliberately different from "today" so the wiring is visible

    monkeypatch.setattr(cli, "last_session", fake_last_session)
    result = invoke("data", "download", "--root", str(repo))
    assert result.exit_code == 0, result.output
    assert seen, "last_session was never called: the CLI is not using the calendar default"
    assert all(c[2] == date(2024, 1, 5) for c in calls)


@needs_pyarrow
def test_download_force_flag_is_passed_through(
    repo: Path, calls: list[tuple[str, date, date]]
) -> None:
    invoke("data", "download", "--root", str(repo), "--asof", "2024-01-10")
    calls.clear()
    result = invoke("data", "download", "--root", str(repo), "--asof", "2024-01-10", "--force")
    assert result.exit_code == 0, result.output
    assert sorted(c[0] for c in calls) == ["AAA", "SPY"]  # re-fetched, not skipped as cached


def test_download_needs_a_valid_config(tmp_path: Path, calls: list[tuple[str, date, date]]) -> None:
    result = invoke("data", "download", "--root", str(tmp_path), "--asof", "2024-01-10")
    assert result.exit_code == 1
    assert "CONFIG ERROR" in result.output
    assert calls == []


def test_download_bad_date_exits_1_before_any_download(
    repo: Path, calls: list[tuple[str, date, date]]
) -> None:
    result = invoke("data", "download", "--root", str(repo), "--asof", "10/01/2024")
    assert result.exit_code == 1
    assert "CONFIG ERROR" in result.output
    assert "not an ISO date" in result.output
    assert calls == []


def test_download_asof_before_universe_start_exits_1(
    repo: Path, calls: list[tuple[str, date, date]]
) -> None:
    result = invoke("data", "download", "--root", str(repo), "--asof", "2013-01-01")
    assert result.exit_code == 1
    assert "CONFIG ERROR" in result.output
    assert "start_date" in result.output
    assert calls == []
