"""``etfcorr.data.validate`` (step 1.3a). Pure: a synthetic panel, a synthetic calendar, no I/O.

Ticker fixture used throughout: AAA/BBB (core, ai_semis, ETF-like, liquidity-gated), SPY
(control, ETF-like, not liquidity-gated), BTC-USD (extended, fintech_crypto, 24/7, never touches
the calendar), ^VIX (control, index-like, zero volume by design, not liquidity-gated).
"""

import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from etfcorr import config as C
from etfcorr.data import validate as V

ROOT = Path(__file__).resolve().parents[2]
ASOF = date(2024, 1, 10)

UNIVERSE = {
    "version": 1,
    "start_date": "2014-01-01",
    "min_adv_usd": 10_000_000,
    "tickers": [
        {"ticker": "AAA", "sector": "ai_semis", "panel": "core", "name": "Alpha"},
        {"ticker": "BBB", "sector": "ai_semis", "panel": "core", "name": "Beta"},
        {"ticker": "SPY", "sector": "control", "panel": "control", "name": "S&P"},
        {"ticker": "BTC-USD", "sector": "fintech_crypto", "panel": "extended", "name": "Bitcoin"},
        {"ticker": "^VIX", "sector": "control", "panel": "control", "name": "VIX"},
    ],
}


@pytest.fixture
def project(tmp_path: Path) -> C.Project:
    shutil.copytree(ROOT / "config", tmp_path / "config")
    (tmp_path / "config" / "universe.yaml").write_text(yaml.safe_dump(UNIVERSE))
    return C.load_project(tmp_path)


def fixture_sessions(start: date, end: date) -> pd.DatetimeIndex:
    """Plain business days, 2024-01-02..2024-01-10 inclusive of both fixture endpoints used here."""
    return pd.DatetimeIndex([d for d in pd.bdate_range(start, end)])


def frame(
    index: pd.DatetimeIndex, close: list[float], volume: list[float] | None = None
) -> pd.DataFrame:
    vol = [1_000_000.0] * len(close) if volume is None else volume
    return pd.DataFrame({"close": close, "volume": vol, "adj_close": close}, index=index)


def make_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # sort=False: tickers here can have genuinely different indices (BTC-USD's weekend-inclusive
    # one vs. everyone else's), and the explicit .sort_index() below is what actually orders the
    # result -- matches the pattern pandas will require by default from v4 (see the same call in
    # etfcorr.data.download._wide_panel, unchanged there: a LOCAL-PASSED file, not touched here).
    panel = pd.concat(frames, axis=1, sort=False)
    panel.columns = panel.columns.set_names(["ticker", "field"])
    return panel.sort_index()


DAYS = pd.bdate_range("2024-01-02", "2024-01-10")  # Tue..Wed, 7 business days
CALENDAR_DAYS = pd.date_range(DAYS[0], DAYS[-1])  # every calendar day, weekends included: BTC-USD


def good_panel() -> pd.DataFrame:
    # Flat prices, not a ramp: with only 6 daily returns, even a smooth arithmetic ramp's mildly
    # decelerating log-returns can look "extreme" relative to their own tiny-sample std (the
    # module's own documented limitation of a self-referential sigma on short samples) -- flat
    # keeps this fixture quiet on every soft dimension except where a test deliberately breaks it.
    # BTC-USD gets its own weekend-inclusive index (real crypto data has weekend rows; the other,
    # NYSE-calendar tickers correctly do not), so a "clean" panel does not spuriously show BTC-USD
    # missing the two weekend days every other ticker never had in the first place.
    aaa = frame(DAYS, [100.0] * len(DAYS))
    bbb = frame(DAYS, [50.0] * len(DAYS))
    spy = frame(DAYS, [400.0] * len(DAYS))
    btc = frame(CALENDAR_DAYS, [30_000.0] * len(CALENDAR_DAYS))
    vix = frame(DAYS, [15.0] * len(DAYS), volume=[0.0] * len(DAYS))
    return make_panel({"AAA": aaa, "BBB": bbb, "SPY": spy, "BTC-USD": btc, "^VIX": vix})


# --------------------------------------------------------------------------------------
# structural (hard) errors
# --------------------------------------------------------------------------------------


def test_a_clean_panel_has_no_errors(project: C.Project) -> None:
    report = V.validate_panel(good_panel(), project, ASOF, sessions_fn=fixture_sessions)
    assert report.ok
    assert report.missing_tickers == ()
    assert set(report.tickers) == {"AAA", "BBB", "SPY", "BTC-USD", "^VIX"}


def test_a_ticker_missing_from_the_panel_is_a_hard_error(project: C.Project) -> None:
    panel = good_panel().drop(columns="BBB", level="ticker")
    report = V.validate_panel(panel, project, ASOF, sessions_fn=fixture_sessions)
    assert not report.ok
    assert report.missing_tickers == ("BBB",)
    assert any("BBB" in e and "absent" in e for e in report.errors)
    assert "BBB" not in report.tickers


def test_non_positive_price_is_a_hard_error(project: C.Project) -> None:
    panel = good_panel()
    panel.loc[panel.index[3], ("AAA", "close")] = 0.0
    report = V.validate_panel(panel, project, ASOF, sessions_fn=fixture_sessions)
    assert not report.ok
    assert any("AAA" in e and "non-positive" in e for e in report.errors)


def test_duplicate_index_is_a_hard_error(project: C.Project) -> None:
    panel = good_panel()
    dup = pd.concat([panel, panel.iloc[[0]]]).sort_index()
    report = V.validate_panel(dup, project, ASOF, sessions_fn=fixture_sessions)
    assert not report.ok
    assert any("duplicate" in e for e in report.errors)


def test_non_monotonic_index_is_a_hard_error(project: C.Project) -> None:
    panel = good_panel().iloc[::-1]  # reversed, still no duplicates
    report = V.validate_panel(panel, project, ASOF, sessions_fn=fixture_sessions)
    assert not report.ok
    assert any("not sorted" in e for e in report.errors)


def test_a_ticker_with_no_valid_close_is_a_hard_error(project: C.Project) -> None:
    panel = good_panel()
    panel[("AAA", "close")] = np.nan
    report = V.validate_panel(panel, project, ASOF, sessions_fn=fixture_sessions)
    assert not report.ok
    assert any("AAA" in e and "no valid close" in e for e in report.errors)
    assert report.tickers["AAA"].is_stale


def test_staleness_is_a_hard_error_for_a_calendar_ticker(project: C.Project) -> None:
    panel = good_panel()
    # Drop AAA's last two rows: its last valid close now lags the fixture calendar's last session.
    panel.loc[panel.index[-2:], ("AAA", "close")] = np.nan
    report = V.validate_panel(panel, project, ASOF, sessions_fn=fixture_sessions)
    assert not report.ok
    assert any("AAA" in e and "before the expected last session" in e for e in report.errors)
    assert report.tickers["AAA"].is_stale


def test_a_24_7_ticker_is_stale_only_against_asof_itself_not_the_calendar(
    project: C.Project,
) -> None:
    panel = good_panel()
    panel.loc[panel.index[-1], ("BTC-USD", "close")] = np.nan  # last row missing, but not asof
    report = V.validate_panel(panel, project, ASOF, sessions_fn=fixture_sessions)
    # BTC-USD's last valid date is now 2024-01-09, one day short of ASOF (2024-01-10): stale.
    assert report.tickers["BTC-USD"].is_stale
    assert any("BTC-USD" in e for e in report.errors)


# --------------------------------------------------------------------------------------
# soft warnings
# --------------------------------------------------------------------------------------


def test_missing_pct_reflects_an_interior_gap(project: C.Project) -> None:
    panel = good_panel()
    panel.loc[panel.index[3], ("AAA", "close")] = np.nan  # one interior gap, not the tail
    report = V.validate_panel(panel, project, ASOF, sessions_fn=fixture_sessions)
    assert report.ok  # a soft issue, not an error
    aaa = report.tickers["AAA"]
    assert aaa.missing_pct is not None and aaa.missing_pct > 0
    assert any("AAA" in w and "missing" in w for w in report.warnings)


def test_extreme_return_day_is_flagged_as_a_warning(project: C.Project) -> None:
    # A single outlier amid only 7 days would dominate its own reference std (self-masking); use
    # a longer, otherwise-quiet series so the +-5-sigma threshold is computed from genuine noise,
    # not from the two return observations the spike itself creates.
    long_days = pd.bdate_range("2024-01-02", periods=250)
    rng = np.random.default_rng(0)
    quiet_returns = rng.normal(0.0, 0.001, size=len(long_days) - 1)
    closes = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(quiet_returns)]))
    closes[120] *= 1.5  # one clear, isolated 50% jump amid ~0.1%-a-day noise
    panel = make_panel(
        {
            "AAA": frame(long_days, list(closes)),
            "BBB": frame(long_days, [50.0] * len(long_days)),
            "SPY": frame(long_days, [400.0] * len(long_days)),
            "BTC-USD": frame(long_days, [30_000.0] * len(long_days)),
            "^VIX": frame(long_days, [15.0] * len(long_days), volume=[0.0] * len(long_days)),
        }
    )
    report = V.validate_panel(panel, project, long_days[-1].date(), sessions_fn=fixture_sessions)
    aaa = report.tickers["AAA"]
    assert len(aaa.extreme_return_dates) >= 1
    assert any("AAA" in w and "sigma" in w for w in report.warnings)


def test_zero_volume_days_warn_for_an_etf_like_ticker_but_not_for_the_index(
    project: C.Project,
) -> None:
    panel = good_panel()
    panel[("AAA", "volume")] = 0.0  # every day zero volume
    report = V.validate_panel(panel, project, ASOF, sessions_fn=fixture_sessions)
    assert any("AAA" in w and "zero-volume" in w for w in report.warnings)
    # ^VIX is already all-zero-volume by construction (index, no volume) and must NOT warn.
    assert not any("^VIX" in w and "zero-volume" in w for w in report.warnings)


def test_low_liquidity_warns_only_for_gated_core_extended_etf_like_tickers(
    project: C.Project,
) -> None:
    panel = good_panel()
    # Drive AAA's (core) and SPY's (control) dollar volume below the 10M threshold identically.
    panel[("AAA", "volume")] = 1.0
    panel[("SPY", "volume")] = 1.0
    report = V.validate_panel(panel, project, ASOF, sessions_fn=fixture_sessions)
    assert any("AAA" in w and "median dollar volume" in w for w in report.warnings)
    assert not any("SPY" in w and "median dollar volume" in w for w in report.warnings)


def test_min_adv_usd_override_is_honoured(project: C.Project) -> None:
    report = V.validate_panel(
        good_panel(), project, ASOF, sessions_fn=fixture_sessions, min_adv_usd=1e12
    )
    assert any("AAA" in w and "median dollar volume" in w for w in report.warnings)


# --------------------------------------------------------------------------------------
# assert_valid / summary_lines
# --------------------------------------------------------------------------------------


def test_assert_valid_raises_joining_every_error(project: C.Project) -> None:
    panel = good_panel().drop(columns="BBB", level="ticker")
    report = V.validate_panel(panel, project, ASOF, sessions_fn=fixture_sessions)
    with pytest.raises(V.DataQualityError, match="BBB"):
        V.assert_valid(report)


def test_assert_valid_does_not_raise_when_ok(project: C.Project) -> None:
    report = V.validate_panel(good_panel(), project, ASOF, sessions_fn=fixture_sessions)
    V.assert_valid(report)  # does not raise


def test_summary_lines_reports_counts_and_details(project: C.Project) -> None:
    panel = good_panel().drop(columns="BBB", level="ticker")
    report = V.validate_panel(panel, project, ASOF, sessions_fn=fixture_sessions)
    lines = "\n".join(V.summary_lines(report))
    assert "1 error(s)" in lines
    assert "missing from panel: BBB" in lines
    assert "ERROR:" in lines
