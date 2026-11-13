"""CLI wiring of ``etfcorr data check-universe`` (needs `typer`; skipped where it is missing).

The audit logic is tested in ``test_universe_check.py``; here only the command line is: option
parsing, exit codes, that the vendor fetcher is injected (never called for real), and that a bad
argument fails before any download.
"""

import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from etfcorr.data.universe_check import FetchResult

ROOT = Path(__file__).resolve().parents[2]

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
        return FetchResult(
            pd.DataFrame({"close": 100.0, "volume": 200_000.0}, index=index), "V", "ETF"
        )

    monkeypatch.setattr(cli, "fetch_yfinance", fake_fetch)
    return recorded


def invoke(*args: str) -> Any:
    return typer_testing.CliRunner().invoke(cli.app, list(args))


def test_data_group_is_listed_and_has_the_command() -> None:
    assert "data" in invoke("--help").output
    assert "check-universe" in invoke("data", "--help").output


def test_check_universe_runs_and_writes_the_report(
    repo: Path, calls: list[tuple[str, date, date]]
) -> None:
    result = invoke("data", "check-universe", "--root", str(repo), "--asof", "2026-09-18")
    assert result.exit_code == 0, result.output
    assert "universe check as of 2026-09-18" in result.output
    assert "written:" in result.output
    assert (repo / "data" / "interim" / "universe_check.csv").is_file()
    assert [c[0] for c in calls] == ["AAA", "SPY"]
    assert all(c[2] == date(2026, 9, 18) for c in calls)


def test_check_universe_output_option(repo: Path, calls: list[tuple[str, date, date]]) -> None:
    result = invoke(
        "data", "check-universe", "--root", str(repo), "--asof", "2026-09-18",
        "--out", "reports/tables/u.csv",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert (repo / "reports" / "tables" / "u.csv").is_file()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--asof", "18/09/2026"], "not an ISO date"),
        (["--asof", "2013-01-01"], "invalid audit parameters"),
        (["--asof", "2026-09-18", "--out", "config/universe.yaml"], "refusing to write"),
        (["--asof", "2026-09-18", "--out", "../x.csv"], "refusing to write"),
    ],
)
def test_bad_arguments_exit_1_before_any_download(
    repo: Path, calls: list[tuple[str, date, date]], extra: list[str], message: str
) -> None:
    result = invoke("data", "check-universe", "--root", str(repo), *extra)
    assert result.exit_code == 1
    assert "CONFIG ERROR" in result.output
    assert message in result.output
    assert calls == []


def test_check_universe_needs_a_valid_config(
    tmp_path: Path, calls: list[tuple[str, date, date]]
) -> None:
    result = invoke("data", "check-universe", "--root", str(tmp_path), "--asof", "2026-09-18")
    assert result.exit_code == 1
    assert "CONFIG ERROR" in result.output
    assert calls == []
