"""CLI wiring of ``etfcorr data holdings`` (step 2.4; needs `typer`, skipped where missing).

Fetch/parse/overlap logic is tested in ``test_holdings.py``; here only the command line: option
parsing, exit codes, that ``build_series_map``/``fetch_sec_nport_holdings`` are injected (never
touching the real SEC EDGAR endpoints), that ``--year`` reaches ``build_series_map``, and that the
default ``--asof`` resolves through the calendar rather than falling back to ``date.today()`` (the
mistake step 1.1's Gate 2 made for ``check-universe``; ``data download`` already guards against it
the same way in ``test_cli_download.py``).

``command_holdings`` does not take an injectable fetcher parameter the way ``download_universe``
does: it builds ``fetch_sec_nport_holdings`` internally from ``build_series_map``'s result. So the
network boundary is stubbed by monkeypatching those two names on the ``etfcorr.data.holdings``
module itself (the module ``command_holdings`` looks them up in at call time), not on ``cli``.
"""

import importlib.util
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from etfcorr.data import holdings as H

ROOT = Path(__file__).resolve().parents[2]

# overlap_matrix.parquet needs a real Parquet engine (pyarrow/fastparquet); the planning sandbox
# has neither (PyPI unreachable there). The command-line wiring these tests check is independent
# of the storage engine, so only the tests that reach a successful two-ticker write are skipped
# offline; NOT EXECUTED OFFLINE, confirmed at Gate 2 where pyarrow is a real dependency
# (pyproject.toml), same convention as test_cli_download.py's ``needs_pyarrow``.
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
        {"ticker": "AAA", "sector": "ai_semis", "panel": "core", "name": "Alpha ETF"},
        {"ticker": "BBB", "sector": "ai_semis", "panel": "extended", "name": "Beta ETF"},
        {"ticker": "SPY", "sector": "control", "panel": "control", "name": "S&P"},
    ],
}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    shutil.copytree(ROOT / "config", tmp_path / "config")
    (tmp_path / "config" / "universe.yaml").write_text(yaml.safe_dump(UNIVERSE))
    return tmp_path


def _snap(ticker: str, asof: date, filing_date: date) -> H.HoldingsSnapshot:
    holding = H.Holding(constituent="111111111", name="X Corp", weight=1.0)
    return H.HoldingsSnapshot(
        ticker=ticker, asof=asof, filing_date=filing_date, source="sec_nport", holdings=(holding,)
    )


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Stand-ins for the two network-touching functions ``command_holdings`` calls by name."""
    recorded: dict[str, list[Any]] = {"series_map": [], "fetch": []}

    def fake_build_series_map(
        cache_path: Path, *, year: int, force: bool = False, get: Any = None, user_agent: Any = None
    ) -> dict[str, H.SeriesRef]:
        recorded["series_map"].append(year)
        return {}

    def fake_fetch_sec_nport_holdings(
        ticker: str, asof: date, *, series_map: Any, **kwargs: Any
    ) -> H.HoldingsSnapshot | None:
        recorded["fetch"].append((ticker, asof))
        return _snap(ticker, date(2024, 3, 31), date(2024, 5, 15))

    monkeypatch.setattr(H, "build_series_map", fake_build_series_map)
    monkeypatch.setattr(H, "fetch_sec_nport_holdings", fake_fetch_sec_nport_holdings)
    return recorded


def invoke(*args: str) -> Any:
    return typer_testing.CliRunner().invoke(cli.app, list(args))


def test_holdings_group_is_listed_and_has_the_command() -> None:
    assert "holdings" in invoke("data", "--help").output


@needs_pyarrow
def test_holdings_runs_writes_snapshots_and_uses_given_asof(
    repo: Path, calls: dict[str, list[Any]]
) -> None:
    result = invoke("data", "holdings", "--root", str(repo), "--asof", "2024-06-01")
    assert result.exit_code == 0, result.output
    assert "holdings as of 2024-06-01" in result.output
    assert "snapshots:" in result.output and "overlap matrix:" in result.output
    assert sorted(t for t, _ in calls["fetch"]) == ["AAA", "BBB"]  # SPY (control) excluded
    assert all(a == date(2024, 6, 1) for _, a in calls["fetch"])
    out_dir = repo / "data" / "external" / "holdings"
    assert (out_dir / "AAA_2024-03-31.csv").is_file()
    assert (out_dir / "BBB_2024-03-31.csv").is_file()
    assert (repo / "data" / "external" / "overlap_matrix.parquet").is_file()


def test_holdings_default_asof_goes_through_the_calendar_not_todays_date(
    repo: Path, calls: dict[str, list[Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``--asof``: the CLI must resolve via ``last_session``, not use an unfinished session."""
    seen: list[date] = []

    def fake_last_session(asof: date) -> date:
        seen.append(asof)
        return date(2024, 1, 5)  # deliberately different from "today" so the wiring is visible

    monkeypatch.setattr(cli, "last_session", fake_last_session)
    result = invoke("data", "holdings", "--root", str(repo))
    assert result.exit_code == 0, result.output
    assert seen, "last_session was never called: the CLI is not using the calendar default"
    assert all(a == date(2024, 1, 5) for _, a in calls["fetch"])


def test_holdings_year_option_reaches_build_series_map(
    repo: Path, calls: dict[str, list[Any]]
) -> None:
    result = invoke(
        "data", "holdings", "--root", str(repo), "--asof", "2024-06-01", "--year", "2019"
    )
    assert result.exit_code == 0, result.output
    assert calls["series_map"] == [2019]


def test_holdings_year_defaults_to_the_current_year(
    repo: Path, calls: dict[str, list[Any]]
) -> None:
    result = invoke("data", "holdings", "--root", str(repo), "--asof", "2024-06-01")
    assert result.exit_code == 0, result.output
    assert calls["series_map"] == [date.today().year]


def test_holdings_force_flag_is_passed_through(repo: Path, calls: dict[str, list[Any]]) -> None:
    invoke("data", "holdings", "--root", str(repo), "--asof", "2024-06-01")
    calls["fetch"].clear()
    result = invoke("data", "holdings", "--root", str(repo), "--asof", "2024-06-01", "--force")
    assert result.exit_code == 0, result.output
    # re-fetched, not skipped as cached, even though the first run already wrote a cached snapshot
    assert sorted(t for t, _ in calls["fetch"]) == ["AAA", "BBB"]


def test_holdings_needs_a_valid_config(tmp_path: Path, calls: dict[str, list[Any]]) -> None:
    result = invoke("data", "holdings", "--root", str(tmp_path), "--asof", "2024-06-01")
    assert result.exit_code == 1
    assert "CONFIG ERROR" in result.output
    assert calls["fetch"] == []


def test_holdings_bad_date_exits_1_before_any_fetch(
    repo: Path, calls: dict[str, list[Any]]
) -> None:
    result = invoke("data", "holdings", "--root", str(repo), "--asof", "01/06/2024")
    assert result.exit_code == 1
    assert "CONFIG ERROR" in result.output
    assert "not an ISO date" in result.output
    assert calls["fetch"] == []


def test_holdings_max_candidates_option_reaches_the_fetcher(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--max-candidates`` (2.4, Gate 2 round 3: the 300-default fix) reaches
    ``fetch_sec_nport_holdings`` as an explicit override."""
    recorded: list[dict[str, Any]] = []

    def fake_build_series_map(
        cache_path: Path, *, year: int, force: bool = False, get: Any = None, user_agent: Any = None
    ) -> dict[str, H.SeriesRef]:
        return {}

    def fake_fetch(ticker: str, asof: date, *, series_map: Any, **kwargs: Any) -> Any:
        recorded.append(kwargs)
        return _snap(ticker, date(2024, 3, 31), date(2024, 5, 15))

    monkeypatch.setattr(H, "build_series_map", fake_build_series_map)
    monkeypatch.setattr(H, "fetch_sec_nport_holdings", fake_fetch)

    result = invoke(
        "data", "holdings", "--root", str(repo), "--asof", "2024-06-01", "--max-candidates", "50"
    )
    assert result.exit_code == 0, result.output
    assert recorded, "fetch_sec_nport_holdings was never called"
    assert all(kwargs.get("max_candidates") == 50 for kwargs in recorded)


def test_holdings_omitted_max_candidates_does_not_override_the_fetchers_default(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``--max-candidates``, ``command_holdings`` must not force a ``max_candidates``
    kwarg at all, so ``fetch_sec_nport_holdings`` keeps its own (300) default."""
    recorded: list[dict[str, Any]] = []

    def fake_build_series_map(
        cache_path: Path, *, year: int, force: bool = False, get: Any = None, user_agent: Any = None
    ) -> dict[str, H.SeriesRef]:
        return {}

    def fake_fetch(ticker: str, asof: date, *, series_map: Any, **kwargs: Any) -> Any:
        recorded.append(kwargs)
        return _snap(ticker, date(2024, 3, 31), date(2024, 5, 15))

    monkeypatch.setattr(H, "build_series_map", fake_build_series_map)
    monkeypatch.setattr(H, "fetch_sec_nport_holdings", fake_fetch)

    result = invoke("data", "holdings", "--root", str(repo), "--asof", "2024-06-01")
    assert result.exit_code == 0, result.output
    assert recorded, "fetch_sec_nport_holdings was never called"
    assert all("max_candidates" not in kwargs for kwargs in recorded)


def test_holdings_reports_a_holdings_error_distinctly_from_a_config_error(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``HoldingsError`` (e.g. unset SEC_USER_AGENT) is not mislabelled as a config problem."""

    def raising_build_series_map(*args: Any, **kwargs: Any) -> dict[str, H.SeriesRef]:
        raise H.HoldingsError("SEC_USER_AGENT is not set")

    monkeypatch.setattr(H, "build_series_map", raising_build_series_map)
    result = invoke("data", "holdings", "--root", str(repo), "--asof", "2024-06-01")
    assert result.exit_code == 1
    assert "HOLDINGS ERROR" in result.output
    assert "SEC_USER_AGENT" in result.output
    assert "CONFIG ERROR" not in result.output
