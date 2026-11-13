"""``etfcorr.data.build_panel`` (step 1.3b).

``build_panel`` itself is pure and does not re-validate (see its docstring): a ticker missing
from the raw panel is silently skipped, on the documented assumption that
``command_build_panel`` already refused to reach it via ``validate_panel``. Tests for that
refusal live here too (``command_build_panel``'s own wiring); the validation checks themselves
are ``test_validate.py``'s job.
"""

import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from etfcorr import config as C
from etfcorr.data import build_panel as B
from etfcorr.data.validate import DataQualityError

ROOT = Path(__file__).resolve().parents[2]

UNIVERSE = {
    "version": 1,
    "start_date": "2014-01-01",  # matches the real experiment.yaml's train-start warmup rule
    "min_adv_usd": 10_000_000,
    "tickers": [
        {"ticker": "AAA", "sector": "ai_semis", "panel": "core", "name": "Alpha"},
        {"ticker": "BBB", "sector": "ai_semis", "panel": "core", "name": "Beta"},
        {"ticker": "SPY", "sector": "control", "panel": "control", "name": "S&P"},
        {"ticker": "BTC-USD", "sector": "fintech_crypto", "panel": "extended", "name": "Bitcoin"},
        {"ticker": "^VIX", "sector": "control", "panel": "control", "name": "VIX"},
    ],
}
ASOF = date(2024, 1, 10)


@pytest.fixture
def project(tmp_path: Path) -> C.Project:
    shutil.copytree(ROOT / "config", tmp_path / "config")
    (tmp_path / "config" / "universe.yaml").write_text(yaml.safe_dump(UNIVERSE))
    return C.load_project(tmp_path)


def fixture_sessions(start: date, end: date) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.bdate_range(start, end))


def frame(index: pd.DatetimeIndex, close: float | list[float]) -> pd.DataFrame:
    values = [close] * len(index) if isinstance(close, int | float) else close
    return pd.DataFrame(
        {"close": values, "volume": [1_000_000.0] * len(index), "adj_close": values}, index=index
    )


def make_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    panel = pd.concat(frames, axis=1)
    panel.columns = panel.columns.set_names(["ticker", "field"])
    return panel.sort_index()


# The raw panel only ever contains the days a ticker was actually reported for (download.py
# never writes speculative rows); every ticker here is "listed" only in this recent window, far
# after the universe's configured start_date (2014-01-01). That lets one fixture exercise both
# forms of "missing, not filled": the multi-year pre-inception block every ticker gets once
# build_panel reindexes onto the full 2014-2024 calendar, and BBB's few-day-late start within
# its own recent window.
RECENT_DAYS = pd.bdate_range("2024-01-02", ASOF)  # 7 business days


def full_panel() -> pd.DataFrame:
    late_start = RECENT_DAYS[3:]  # BBB "listed" a few sessions after everyone else
    return make_panel(
        {
            "AAA": frame(RECENT_DAYS, 100.0),
            "BBB": frame(late_start, 50.0),
            "SPY": frame(RECENT_DAYS, 400.0),
            "BTC-USD": frame(RECENT_DAYS, 30_000.0),
            "^VIX": frame(RECENT_DAYS, 15.0),
        }
    )


# --------------------------------------------------------------------------------------
# build_panel (pure)
# --------------------------------------------------------------------------------------


def test_output_index_is_the_shared_nyse_calendar_for_every_ticker_including_24_7(
    project: C.Project,
) -> None:
    clean, _ = B.build_panel(full_panel(), project, ASOF, sessions_fn=fixture_sessions)
    expected_index = pd.DatetimeIndex(fixture_sessions(project.universe.start_date, ASOF))
    assert list(clean.index) == list(expected_index)
    # BTC-USD (24/7) is reindexed onto the SAME index, not its own calendar-free one.
    assert list(clean["BTC-USD"].index) == list(expected_index)


def test_pre_inception_rows_are_nan_not_forward_filled(project: C.Project) -> None:
    clean, _ = B.build_panel(full_panel(), project, ASOF, sessions_fn=fixture_sessions)
    aaa_close = clean[("AAA", "close")]
    # Everything before AAA's first reported session (2024-01-02) is pre-inception: NaN, over
    # roughly a decade of the aligned calendar, never filled in. (This much would hold even with
    # a forward-fill: there is nothing earlier to fill *from*. It is a real check on alignment,
    # just not, by itself, a mutation-sensitive one -- see the interior-gap test below for that.)
    assert aaa_close.loc[: pd.Timestamp("2024-01-01")].isna().all()
    assert aaa_close.loc[pd.Timestamp("2024-01-02") :].notna().all()

    bbb_close = clean[("BBB", "close")]
    tail = bbb_close.loc[pd.Timestamp("2024-01-02") : pd.Timestamp(ASOF)]
    assert tail.iloc[:3].isna().all()  # BBB's own few-day-late start within the recent window
    assert tail.iloc[3:].notna().all()


def test_an_interior_gap_is_not_forward_filled(project: C.Project) -> None:
    """The one scenario a `.ffill()` mutation actually changes: a gap *between* two valid rows.

    (Confirmed by mutation: temporarily changing `build_panel`'s `raw.reindex(aligned_index)` to
    `raw.reindex(aligned_index).ffill()` turns this test red, restored after.)
    """
    gappy_close = [100.0, 100.0, 100.0, None, 100.0, 100.0, 100.0]  # a hole on day 4 of 7
    panel = make_panel({"AAA": frame(RECENT_DAYS, gappy_close), "SPY": frame(RECENT_DAYS, 400.0)})
    clean, _ = B.build_panel(panel, project, ASOF, sessions_fn=fixture_sessions)
    aaa_close = clean[("AAA", "close")]
    assert aaa_close.loc[RECENT_DAYS[2]] == 100.0
    assert pd.isna(aaa_close.loc[RECENT_DAYS[3]])  # not filled from the previous day
    assert aaa_close.loc[RECENT_DAYS[4]] == 100.0


def test_a_ticker_missing_from_the_raw_panel_is_skipped_not_raised(project: C.Project) -> None:
    panel = full_panel().drop(columns="BBB", level="ticker")
    clean, coverage = B.build_panel(panel, project, ASOF, sessions_fn=fixture_sessions)
    assert "BBB" not in set(clean.columns.get_level_values("ticker"))
    assert "BBB" not in coverage.index


def test_coverage_table_fields(project: C.Project) -> None:
    _, coverage = B.build_panel(full_panel(), project, ASOF, sessions_fn=fixture_sessions)
    aaa = coverage.loc["AAA"]
    assert aaa["sector"] == "ai_semis"
    assert aaa["panel"] == "core"
    assert aaa["first_date"] == "2024-01-02"
    assert aaa["last_date"] == ASOF.isoformat()
    assert aaa["n_valid"] == aaa["expected_sessions"]
    assert aaa["missing_pct"] == 0.0

    bbb = coverage.loc["BBB"]
    assert bbb["first_date"] == str(RECENT_DAYS[3].date())
    assert bbb["missing_pct"] == 0.0  # no gaps *within* BBB's own active window


def test_no_sessions_in_the_window_raises_config_error(project: C.Project) -> None:
    def empty(start: date, end: date) -> pd.DatetimeIndex:
        return pd.DatetimeIndex([])

    with pytest.raises(C.ConfigError, match="no NYSE sessions"):
        B.build_panel(full_panel(), project, ASOF, sessions_fn=empty)


# --------------------------------------------------------------------------------------
# write_coverage_csv (real CSV, runs offline)
# --------------------------------------------------------------------------------------


def test_write_coverage_csv_round_trips(tmp_path: Path, project: C.Project) -> None:
    _, coverage = B.build_panel(full_panel(), project, ASOF, sessions_fn=fixture_sessions)
    path = tmp_path / "coverage.csv"
    B.write_coverage_csv(coverage, path)
    back = pd.read_csv(path, index_col="ticker")
    assert set(back.index) == set(coverage.index)
    assert back.loc["AAA", "sector"] == "ai_semis"


# --------------------------------------------------------------------------------------
# command_build_panel (orchestration: validate, then build, then write)
# --------------------------------------------------------------------------------------


def _read_pickle(path: Path) -> pd.DataFrame:
    return pd.read_pickle(path)


def _write_pickle(panel: pd.DataFrame, path: Path) -> None:
    panel.to_pickle(path)


def _write_raw_cache(project: C.Project, panel: pd.DataFrame) -> None:
    raw_path = project.paths.data_raw / "prices_yf.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pickle(panel, raw_path)


def test_command_raises_config_error_when_the_raw_cache_is_missing(project: C.Project) -> None:
    with pytest.raises(C.ConfigError, match="data download"):
        B.command_build_panel(
            project, asof=ASOF, sessions_fn=fixture_sessions, read_panel=_read_pickle
        )


def test_command_raises_data_quality_error_and_writes_nothing_on_a_validation_failure(
    project: C.Project,
) -> None:
    panel = full_panel().drop(columns="BBB", level="ticker")  # BBB missing -> hard error
    _write_raw_cache(project, panel)
    with pytest.raises(DataQualityError, match="BBB"):
        B.command_build_panel(
            project,
            asof=ASOF,
            sessions_fn=fixture_sessions,
            read_panel=_read_pickle,
            write_clean=_write_pickle,
        )
    assert not (project.paths.data_interim / B.CLEAN_NAME).exists()
    assert not (project.paths.data_processed / B.COVERAGE_NAME).exists()


def test_command_writes_both_outputs_on_success(project: C.Project) -> None:
    _write_raw_cache(project, full_panel())
    lines = B.command_build_panel(
        project,
        asof=ASOF,
        sessions_fn=fixture_sessions,
        read_panel=_read_pickle,
        write_clean=_write_pickle,
    )
    clean_path = project.paths.data_interim / B.CLEAN_NAME
    coverage_path = project.paths.data_processed / B.COVERAGE_NAME
    assert clean_path.is_file()
    assert coverage_path.is_file()
    assert any(str(clean_path) in line for line in lines)
    assert any(str(coverage_path) in line for line in lines)
    clean = _read_pickle(clean_path)
    assert set(clean.columns.get_level_values("ticker")) == {"AAA", "BBB", "SPY", "BTC-USD", "^VIX"}
