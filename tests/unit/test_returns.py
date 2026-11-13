"""``etfcorr.returns`` (step 2.1).

Numerical correctness of ``log_returns``/``horizon_returns`` (known-answer tests, NaN
propagation, the incomplete-final-period guard) and the ``command_returns`` orchestration
(reading ``prices_clean.parquet``, writing ``returns_<horizon>.parquet`` per configured
horizon). The leakage properties of the incomplete-period guard across D/W/M/Q and multiple cut
points are ``tests/leakage/test_returns_no_lookahead.py``'s job, not repeated here in detail.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from etfcorr import config as C
from etfcorr import returns as R

ROOT = Path(__file__).resolve().parents[2]

UNIVERSE = {
    "version": 1,
    "start_date": "2014-01-01",
    "min_adv_usd": 10_000_000,
    "tickers": [
        {"ticker": "AAA", "sector": "ai_semis", "panel": "core", "name": "Alpha"},
        {"ticker": "BBB", "sector": "ai_semis", "panel": "core", "name": "Beta"},
        {"ticker": "SPY", "sector": "control", "panel": "control", "name": "S&P"},
    ],
}


@pytest.fixture
def project(tmp_path: Path) -> C.Project:
    shutil.copytree(ROOT / "config", tmp_path / "config")
    (tmp_path / "config" / "universe.yaml").write_text(yaml.safe_dump(UNIVERSE))
    return C.load_project(tmp_path)


def panel_frame(index: pd.DatetimeIndex, fields: dict[str, list[float | None]]) -> pd.DataFrame:
    return pd.DataFrame(fields, index=index)


def wide_panel(tickers: dict[str, pd.DataFrame]) -> pd.DataFrame:
    panel = pd.concat(tickers, axis=1)
    panel.columns = panel.columns.set_names(["ticker", "field"])
    return panel.sort_index()


# --------------------------------------------------------------------------------------
# select_field / adjusted_close
# --------------------------------------------------------------------------------------


def test_adjusted_close_selects_the_adj_close_field() -> None:
    idx = pd.bdate_range("2024-01-02", periods=3)
    fields = {"close": [100.0, 101.0, 102.0], "adj_close": [10.0, 11.0, 12.0]}
    panel = wide_panel({"AAA": panel_frame(idx, fields)})
    out = R.adjusted_close(panel)
    assert list(out.columns) == ["AAA"]
    assert out["AAA"].tolist() == [10.0, 11.0, 12.0]


def test_select_field_rejects_a_non_multiindex_panel() -> None:
    flat = pd.DataFrame({"AAA": [1.0, 2.0]})
    with pytest.raises(C.ConfigError, match="MultiIndex"):
        R.select_field(flat, "adj_close")


def test_select_field_rejects_a_missing_field() -> None:
    idx = pd.bdate_range("2024-01-02", periods=2)
    panel = wide_panel({"AAA": panel_frame(idx, {"close": [100.0, 101.0]})})
    with pytest.raises(C.ConfigError, match="no 'adj_close' field"):
        R.select_field(panel, "adj_close")


# --------------------------------------------------------------------------------------
# log_returns: known values, NaN propagation, positivity guard
# --------------------------------------------------------------------------------------


def test_log_returns_known_values() -> None:
    idx = pd.bdate_range("2024-01-02", periods=3)
    prices = pd.DataFrame({"AAA": [100.0, 110.0, 121.0]}, index=idx)
    out = R.log_returns(prices)
    assert list(out.index) == list(idx[1:])
    np.testing.assert_allclose(out["AAA"].to_numpy(), [np.log(1.1), np.log(1.1)])


def test_log_returns_drops_only_the_first_row() -> None:
    idx = pd.bdate_range("2024-01-02", periods=5)
    prices = pd.DataFrame({"AAA": np.linspace(100.0, 104.0, 5)}, index=idx)
    assert len(R.log_returns(prices)) == 4


def test_log_returns_pre_inception_gap_stays_nan_not_filled() -> None:
    idx = pd.bdate_range("2024-01-02", periods=5)
    # AAA "listed" only from the 3rd session; the first two are pre-inception, not a data error.
    prices = pd.DataFrame({"AAA": [None, None, 100.0, 101.0, 102.0]}, index=idx)
    out = R.log_returns(prices)
    assert out["AAA"].iloc[0:2].isna().all()  # rows for idx[1], idx[2]: no valid t-1 price yet
    np.testing.assert_allclose(out["AAA"].iloc[2:].to_numpy(), np.log([101 / 100, 102 / 101]))


def test_log_returns_interior_gap_produces_two_nan_rows_not_one() -> None:
    """A return needs *both* p_t and p_{t-1}: a single missing day poisons two output rows."""
    idx = pd.bdate_range("2024-01-02", periods=5)
    prices = pd.DataFrame({"AAA": [100.0, 101.0, None, 103.0, 104.0]}, index=idx)
    out = R.log_returns(prices)
    assert out["AAA"].tolist()[1:3] == [None, None] or out["AAA"].iloc[1:3].isna().all()
    assert out["AAA"].iloc[0] == pytest.approx(np.log(101 / 100))
    assert out["AAA"].iloc[3] == pytest.approx(np.log(104 / 103))


def test_log_returns_rejects_a_non_positive_price() -> None:
    idx = pd.bdate_range("2024-01-02", periods=3)
    prices = pd.DataFrame({"AAA": [100.0, 0.0, 102.0]}, index=idx)
    with pytest.raises(R.ReturnsError, match="positive"):
        R.log_returns(prices)


def test_log_returns_nan_prices_do_not_trip_the_positivity_guard() -> None:
    idx = pd.bdate_range("2024-01-02", periods=3)
    prices = pd.DataFrame({"AAA": [None, 101.0, 102.0]}, index=idx)
    R.log_returns(prices)  # must not raise


# --------------------------------------------------------------------------------------
# resample_prices / horizon_returns: horizon mapping and the incomplete-period guard
# --------------------------------------------------------------------------------------


def test_resample_prices_daily_is_the_identity() -> None:
    idx = pd.bdate_range("2024-01-02", periods=4)
    prices = pd.DataFrame({"AAA": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    assert R.resample_prices(prices, "D") is prices


def test_resample_prices_rejects_an_unknown_horizon() -> None:
    idx = pd.bdate_range("2024-01-02", periods=2)
    prices = pd.DataFrame({"AAA": [1.0, 2.0]}, index=idx)
    with pytest.raises(R.ReturnsError, match="unknown horizon"):
        R.resample_prices(prices, "Y")  # type: ignore[arg-type]  # deliberately invalid at runtime


def test_horizon_returns_weekly_drops_the_incomplete_final_week_known_values() -> None:
    """3 complete Mon-Fri weeks (2024-01-01..19) + 2 days into a 4th (Jan 22-23, incomplete).

    Prices are 100 + day-index, so each complete week's log return is known exactly by hand:
    Friday closes are 104 (week 1), 109 (week 2), 114 (week 3); week 4 has no Friday close and
    must be dropped, leaving exactly 2 returns (week1->week2, week2->week3).
    """
    idx = pd.bdate_range("2024-01-01", "2024-01-23")  # Mon 1 Jan .. Tue 23 Jan, 17 sessions
    assert len(idx) == 17
    prices = pd.DataFrame({"AAA": 100.0 + np.arange(17)}, index=idx)
    out = R.horizon_returns(prices, "W")
    assert list(out.index) == [pd.Timestamp("2024-01-12"), pd.Timestamp("2024-01-19")]
    np.testing.assert_allclose(out["AAA"].to_numpy(), [np.log(109 / 104), np.log(114 / 109)])


def test_horizon_returns_weekly_keeps_the_final_week_when_it_reaches_friday() -> None:
    idx = pd.bdate_range("2024-01-01", "2024-01-19")  # ends exactly on a Friday: 3 full weeks
    prices = pd.DataFrame({"AAA": 100.0 + np.arange(len(idx))}, index=idx)
    out = R.horizon_returns(prices, "W")
    assert list(out.index) == [pd.Timestamp("2024-01-12"), pd.Timestamp("2024-01-19")]


def test_horizon_returns_drop_incomplete_false_keeps_the_partial_period() -> None:
    """The escape hatch used only by the leakage test to build a deliberately leaky twin."""
    idx = pd.bdate_range("2024-01-01", "2024-01-23")
    prices = pd.DataFrame({"AAA": 100.0 + np.arange(len(idx))}, index=idx)
    with_guard = R.horizon_returns(prices, "W", drop_incomplete=True)
    without_guard = R.horizon_returns(prices, "W", drop_incomplete=False)
    assert len(without_guard) == len(with_guard) + 1


def test_horizon_counts_match_the_verified_project_plan_result() -> None:
    """Regression check against project_plan.md section 6.2's own reported synthetic-data test:
    'D = 2,794 / W = 558 / M = 127 / Q = 41 observations' over 2016-01-04..2026-09-18 business
    days -- recomputed here (not assumed) against this module's actual functions."""
    idx = pd.bdate_range("2016-01-04", "2026-09-18")
    prices = pd.DataFrame({"AAA": 100.0 + np.arange(len(idx), dtype=float)}, index=idx)
    assert len(R.horizon_returns(prices, "D")) == 2794
    assert len(R.horizon_returns(prices, "W")) == 558
    assert len(R.horizon_returns(prices, "M")) == 127
    assert len(R.horizon_returns(prices, "Q")) == 41


def test_returns_by_horizon_covers_every_requested_horizon_in_order() -> None:
    idx = pd.bdate_range("2016-01-04", "2018-01-01")
    prices = pd.DataFrame({"AAA": 100.0 + np.arange(len(idx), dtype=float)}, index=idx)
    out = R.returns_by_horizon(prices, ("D", "Q", "W"))
    assert list(out.keys()) == ["D", "Q", "W"]
    assert all(isinstance(df, pd.DataFrame) for df in out.values())


# --------------------------------------------------------------------------------------
# command_returns (orchestration: read prices_clean.parquet, write per-horizon files)
# --------------------------------------------------------------------------------------


def _clean_panel(project: C.Project) -> pd.DataFrame:
    idx = pd.bdate_range("2016-01-04", "2016-06-30")
    n = len(idx)
    frames = {
        t: panel_frame(
            idx,
            {
                "close": (100.0 + np.arange(n)).tolist(),
                "volume": [1_000_000.0] * n,
                "adj_close": (100.0 + np.arange(n)).tolist(),
            },
        )
        for t in ("AAA", "BBB", "SPY")
    }
    return wide_panel(frames)


def _read_pickle(path: Path) -> pd.DataFrame:
    return pd.read_pickle(path)


def _write_pickle(panel: pd.DataFrame, path: Path) -> None:
    panel.to_pickle(path)


def test_command_returns_raises_config_error_when_the_clean_panel_is_missing(
    project: C.Project,
) -> None:
    with pytest.raises(C.ConfigError, match="data build"):
        R.command_returns(project, read_panel=_read_pickle)


def test_command_returns_writes_one_file_per_configured_horizon(project: C.Project) -> None:
    clean_path = project.paths.data_interim / "prices_clean.parquet"
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pickle(_clean_panel(project), clean_path)

    lines = R.command_returns(project, read_panel=_read_pickle, write_returns=_write_pickle)

    for horizon in project.experiment.horizons:
        out_path = project.paths.data_processed / f"returns_{horizon}.parquet"
        assert out_path.is_file(), horizon
        assert any(str(out_path) in line for line in lines)
        written = _read_pickle(out_path)
        assert set(written.columns) == {"AAA", "BBB", "SPY"}
        assert len(written) > 0


def test_returns_path_matches_what_command_returns_writes(project: C.Project) -> None:
    clean_path = project.paths.data_interim / "prices_clean.parquet"
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pickle(_clean_panel(project), clean_path)
    R.command_returns(project, read_panel=_read_pickle, write_returns=_write_pickle)
    for horizon in project.experiment.horizons:
        assert R.returns_path(project.paths.data_processed, horizon).is_file()
