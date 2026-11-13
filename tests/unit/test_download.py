"""``etfcorr.data.download`` (step 1.2). No network: ``fetch`` is always an injected stand-in.

Uses a small custom ``universe.yaml`` (2-3 tickers) copied next to the real, committed
``experiment.yaml``/``hypotheses.lock`` (same pattern as ``test_cli_data.py``'s ``repo`` fixture),
so ``load_project`` validates normally without dragging in all 56 real tickers.
"""

import json
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml

from etfcorr import config as C
from etfcorr.data import download as D
from etfcorr.data.universe_check import FetchResult

ROOT = Path(__file__).resolve().parents[2]
ASOF = date(2024, 1, 10)


def _read_pickle(path: Path) -> pd.DataFrame:
    return pd.read_pickle(path)


def _write_pickle(panel: pd.DataFrame, path: Path) -> None:
    panel.to_pickle(path)


def download(project: C.Project, fetch: D.Fetcher, **kw: object) -> D.DownloadReport:
    """``download_universe`` with a pickle-based panel reader/writer (no ``pyarrow`` offline)."""
    kw.setdefault("read_panel", _read_pickle)
    kw.setdefault("write_panel", _write_pickle)
    return D.download_universe(project, fetch, **kw)  # type: ignore[arg-type]


def frame(start: str, n: int, *, close0: float = 100.0) -> pd.DataFrame:
    index = pd.bdate_range(start, periods=n)
    close = pd.Series([close0 + i for i in range(n)], index=index, dtype=float)
    return pd.DataFrame(
        {"close": close, "volume": pd.Series(1_000.0, index=index), "adj_close": close - 0.5}
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    shutil.copytree(ROOT / "config", tmp_path / "config")
    universe = {
        "version": 1,
        "start_date": "2014-01-01",
        "min_adv_usd": 10_000_000,
        "tickers": [
            {"ticker": "AAA", "sector": "ai_semis", "panel": "core", "name": "Alpha"},
            {"ticker": "BBB", "sector": "ai_semis", "panel": "core", "name": "Beta"},
            {"ticker": "SPY", "sector": "control", "panel": "control", "name": "S&P"},
        ],
    }
    (tmp_path / "config" / "universe.yaml").write_text(yaml.safe_dump(universe))
    return tmp_path


@pytest.fixture
def project(repo: Path) -> C.Project:
    return C.load_project(repo)


def clock() -> datetime:
    return datetime(2026, 9, 27, 12, 0, tzinfo=UTC)


def fetcher(histories: dict[str, pd.DataFrame]) -> D.Fetcher:
    calls: list[str] = []

    def _fetch(ticker: str, start: date, end: date) -> FetchResult:
        calls.append(ticker)
        if ticker not in histories:
            return FetchResult(None)
        return FetchResult(histories[ticker], f"{ticker} Inc", "ETF")

    _fetch.calls = calls  # type: ignore[attr-defined]
    return _fetch


def read_log(project: C.Project) -> dict:
    return json.loads((project.paths.data_raw / D.LOG_NAME).read_text())


def read_panel(project: C.Project) -> pd.DataFrame:
    return _read_pickle(project.paths.data_raw / D.CACHE_NAME)


# --------------------------------------------------------------------------------------
# first download
# --------------------------------------------------------------------------------------


def test_first_download_writes_a_panel_and_a_log_for_every_ticker(project: C.Project) -> None:
    histories = {
        "AAA": frame("2024-01-02", 7),
        "BBB": frame("2024-01-02", 7),
        "SPY": frame("2024-01-02", 7),
    }
    report = download(project, fetcher(histories), asof=ASOF, now=clock)

    assert report.fetched == ("AAA", "BBB", "SPY")
    assert report.cached == ()
    assert report.failed == ()
    assert report.fetched_at == "2026-09-27T12:00:00+00:00"

    panel = read_panel(project)
    assert isinstance(panel.columns, pd.MultiIndex)
    assert set(panel.columns.get_level_values("ticker")) == {"AAA", "BBB", "SPY"}
    assert list(panel["AAA"].columns) == list(D.FIELDS)
    assert len(panel) == 7

    log = read_log(project)
    assert log["asof"] == "2024-01-10"
    assert set(log["tickers"]) == {"AAA", "BBB", "SPY"}
    assert all(e["status"] == "fetched" and e["rows"] == 7 for e in log["tickers"].values())
    assert len(log["sha256"]) == 64


def test_a_missing_ticker_is_reported_failed_and_absent_from_the_panel(
    project: C.Project,
) -> None:
    report = download(project, fetcher({"AAA": frame("2024-01-02", 5)}), asof=ASOF, now=clock)
    assert report.fetched == ("AAA",)
    assert report.failed == ("BBB", "SPY")
    panel = read_panel(project)
    assert set(panel.columns.get_level_values("ticker")) == {"AAA"}
    log = read_log(project)
    assert log["tickers"]["BBB"]["status"] == "failed"
    assert log["tickers"]["BBB"]["error"] == "no_data"


def test_a_raising_fetcher_is_recorded_as_failed_and_does_not_stop_the_run(
    project: C.Project,
) -> None:
    def flaky(ticker: str, start: date, end: date) -> FetchResult:
        if ticker == "BBB":
            raise ConnectionError("429 Too Many Requests")
        return FetchResult(frame("2024-01-02", 5), ticker, "ETF")

    report = download(project, flaky, asof=ASOF, now=clock)
    assert set(report.fetched) == {"AAA", "SPY"}
    assert report.failed == ("BBB",)
    log = read_log(project)
    assert "429" in log["tickers"]["BBB"]["error"]


# --------------------------------------------------------------------------------------
# incremental caching (skip / force)
# --------------------------------------------------------------------------------------


def test_a_second_run_at_the_same_asof_skips_every_ticker(project: C.Project) -> None:
    histories = {t: frame("2024-01-02", 7) for t in ("AAA", "BBB", "SPY")}
    download(project, fetcher(histories), asof=ASOF, now=clock)
    fetch2 = fetcher(histories)
    report2 = download(project, fetch2, asof=ASOF, now=clock)
    assert report2.cached == ("AAA", "BBB", "SPY")
    assert report2.fetched == ()
    assert fetch2.calls == []  # type: ignore[attr-defined]  # never called: nothing needed fetching


def test_a_later_asof_refetches_every_ticker(project: C.Project) -> None:
    histories = {t: frame("2024-01-02", 7) for t in ("AAA", "BBB", "SPY")}
    download(project, fetcher(histories), asof=ASOF, now=clock)
    later = {t: frame("2024-01-02", 8) for t in ("AAA", "BBB", "SPY")}
    report2 = download(project, fetcher(later), asof=date(2024, 1, 11), now=clock)
    assert report2.fetched == ("AAA", "BBB", "SPY")
    assert len(read_panel(project)) == 8


def test_force_refetches_even_when_the_cache_is_fresh(project: C.Project) -> None:
    histories = {t: frame("2024-01-02", 7) for t in ("AAA", "BBB", "SPY")}
    download(project, fetcher(histories), asof=ASOF, now=clock)
    fetch2 = fetcher(histories)
    report2 = download(project, fetch2, asof=ASOF, force=True, now=clock)
    assert report2.fetched == ("AAA", "BBB", "SPY")
    assert sorted(fetch2.calls) == ["AAA", "BBB", "SPY"]  # type: ignore[attr-defined]


def test_a_failed_refresh_keeps_the_previous_good_data(project: C.Project) -> None:
    good = {t: frame("2024-01-02", 7) for t in ("AAA", "BBB", "SPY")}
    download(project, fetcher(good), asof=ASOF, now=clock)
    panel_before = read_panel(project)["BBB"].copy()

    def bbb_fails(ticker: str, start: date, end: date) -> FetchResult:
        if ticker == "BBB":
            raise ConnectionError("boom")
        return FetchResult(frame("2024-01-02", 8), ticker, "ETF")

    report2 = download(project, bbb_fails, asof=date(2024, 1, 11), force=True, now=clock)
    assert report2.failed == ("BBB",)
    panel_after = read_panel(project)
    assert len(panel_after["AAA"]) == 8  # the others did refresh
    # BBB's own 7 rows are untouched; the panel's shared date index grew by one day (from AAA/
    # SPY), so BBB gains a trailing NaN row there -- that is honest alignment, not silent loss.
    pd.testing.assert_frame_equal(panel_after["BBB"].iloc[:7], panel_before)
    assert panel_after["BBB"].iloc[7].isna().all()
    log = read_log(project)
    assert log["tickers"]["BBB"]["status"] == "failed"


def test_a_ticker_removed_from_the_universe_is_dropped_on_the_next_run(
    tmp_path: Path,
) -> None:
    shutil.copytree(ROOT / "config", tmp_path / "config")
    full = {
        "version": 1,
        "start_date": "2014-01-01",
        "min_adv_usd": 10_000_000,
        "tickers": [
            {"ticker": "AAA", "sector": "ai_semis", "panel": "core", "name": "Alpha"},
            {"ticker": "BBB", "sector": "ai_semis", "panel": "core", "name": "Beta"},
        ],
    }
    (tmp_path / "config" / "universe.yaml").write_text(yaml.safe_dump(full))
    project = C.load_project(tmp_path)
    histories = {"AAA": frame("2024-01-02", 5), "BBB": frame("2024-01-02", 5)}
    download(project, fetcher(histories), asof=ASOF, now=clock)

    shrunk = dict(full)
    shrunk["tickers"] = [full["tickers"][0]]
    (tmp_path / "config" / "universe.yaml").write_text(yaml.safe_dump(shrunk))
    project2 = C.load_project(tmp_path)
    report2 = download(project2, fetcher(histories), asof=ASOF, now=clock)

    assert report2.cached == ("AAA",)
    panel = read_panel(project2)
    assert set(panel.columns.get_level_values("ticker")) == {"AAA"}
    log = read_log(project2)
    assert set(log["tickers"]) == {"AAA"}


# --------------------------------------------------------------------------------------
# defensive no-lookahead clipping
# --------------------------------------------------------------------------------------


def test_rows_after_asof_are_clipped_before_hashing_and_writing(project: C.Project) -> None:
    """A fetcher bug that returns rows past asof must never reach the cache or the hash."""
    leaky = frame("2024-01-02", 10)  # runs to 2024-01-15, well past ASOF = 2024-01-10
    report = download(
        project, fetcher({"AAA": leaky, "BBB": leaky, "SPY": leaky}), asof=ASOF, now=clock
    )
    panel = read_panel(project)
    assert panel.index.max() <= pd.Timestamp(ASOF)
    assert (panel.index > pd.Timestamp(ASOF)).sum() == 0
    log = read_log(project)
    assert log["tickers"]["AAA"]["rows"] == len(panel)
    # the report's own outcome rows count must match what was actually clipped and written
    assert [o.rows for o in report.outcomes] == [len(panel)] * 3


# --------------------------------------------------------------------------------------
# validation and error paths
# --------------------------------------------------------------------------------------


def test_asof_before_universe_start_date_raises(project: C.Project) -> None:
    with pytest.raises(C.ConfigError, match="start_date"):
        download(project, fetcher({}), asof=date(2013, 12, 31), now=clock)


def test_an_unreadable_log_raises_config_error(project: C.Project) -> None:
    log_path = project.paths.data_raw / D.LOG_NAME
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("{not json")
    with pytest.raises(C.ConfigError, match="unreadable download log"):
        download(project, fetcher({}), asof=ASOF, now=clock)


# --------------------------------------------------------------------------------------
# summary_lines
# --------------------------------------------------------------------------------------


def test_summary_lines_reports_counts_and_paths(project: C.Project) -> None:
    histories = {"AAA": frame("2024-01-02", 5)}
    report = download(project, fetcher(histories), asof=ASOF, now=clock)
    lines = D.summary_lines(report)
    joined = "\n".join(lines)
    assert "fetched 1, cached 0, failed 2" in joined
    assert "failed: BBB, SPY" in joined
    assert str(report.panel_path) in joined
    assert str(report.log_path) in joined
