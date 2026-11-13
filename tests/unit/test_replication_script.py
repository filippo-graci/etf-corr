"""Tests for scripts/replication_pre2014.py (E5-E8), on synthetic data.

The script is loaded by path; nothing here reads the project's data or the replication root.
"""

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from etfcorr.config import load_project
from etfcorr.synthetic import make_factor_returns

ROOT = Path(__file__).resolve().parents[2]


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = load("replication_pre2014")


@pytest.fixture(scope="module")
def panel() -> tuple[pd.DataFrame, dict[str, str], pd.Series, pd.Series]:
    """Daily returns from 2006-06-20 to past 2013-12-31, sectors, SPY and a VIX-like level."""
    synth = make_factor_returns(1980, (2, 2, 2), sector_loading=0.4, start="2006-06-20", seed=3)
    daily = synth.returns.loc[:"2013-12-31"]
    sectors = dict(zip(daily.columns, synth.sectors, strict=True))
    spy = daily.mean(axis=1).rename("SPY")
    vix = (15 + 400 * spy.abs().rolling(10, min_periods=1).mean()).rename("^VIX")
    return daily, sectors, spy, vix


def test_panel_rule_drops_holdrs_late_funds_and_non_core() -> None:
    coverage = pd.DataFrame(
        {
            "ticker": ["SMH", "OIH", "SOXX", "XME", "AMLP", "SPY", "IHF"],
            "panel": ["core", "core", "core", "core", "core", "control", "core"],
            "first_date": [
                "2000-06-05",
                "2001-02-26",
                "2006-01-03",
                "2006-07-10",
                "2010-08-25",
                "2006-01-03",
                None,
            ],
        }
    )
    assert R.panel_rule(coverage) == ["SOXX", "XME"]


def test_phase_codes_cover_the_declared_dates_and_nothing_else() -> None:
    index = pd.DatetimeIndex(
        [
            "2006-12-29",
            "2007-01-02",
            "2007-07-31",
            "2007-08-01",
            "2008-09-12",
            "2008-09-15",
            "2009-03-09",
            "2009-03-10",
            "2011-12-30",
            "2012-01-03",
            "2013-12-31",
        ]
    )
    codes = R.phase_codes(index)
    assert np.isnan(codes.iloc[0])
    assert codes.iloc[1:].tolist() == [0, 0, 1, 1, 2, 2, 3, 4, 5, 5]


def test_within_phase_draws_never_cross_a_phase() -> None:
    codes = np.repeat([0, 1, 2], [50, 30, 80])
    rng = np.random.default_rng(0)
    for _ in range(20):
        idx = R.within_phase_index(codes, 21.0, rng)
        assert len(idx) == len(codes)
        np.testing.assert_array_equal(codes[idx], codes)


def test_e6_start_waits_for_every_cell_and_the_dcc(panel: tuple) -> None:
    daily = panel[0]
    weekly = daily.resample("W-FRI").sum()
    dcc = np.full((len(daily), 6, 6), np.nan)
    dcc[503:] = np.eye(6)  # first forecast after 504 rows
    start = R.e6_start({"D": daily, "W": weekly}, dcc)
    assert (daily.index < start).sum() >= 504
    assert (weekly.index < start).sum() >= 104
    earlier = daily.index[daily.index < start]
    month_before = earlier[
        earlier.to_period("M") == (start - pd.offsets.MonthBegin(1)).to_period("M")
    ]
    assert not (
        (daily.index < month_before[0]).sum() >= 504
        and (weekly.index < month_before[0]).sum() >= 104
    )
    assert start.day <= 3 and str(start.date()) >= "2008-06-01"


def test_holm_and_zero_return_share(panel: tuple) -> None:
    adjusted = R.holm({"E5": 0.01, "E6": 0.04, "E7": 0.03}, 0.05)
    assert adjusted == pytest.approx({"E5": 0.03, "E6": 0.06, "E7": 0.06})
    daily = panel[0].copy()
    daily.iloc[:20, 0] = 0.0  # stale days, some of them before the warm-up year starts
    share = R.zero_return_share(daily)
    inside = (daily.index >= R.WARMUP[0]) & (daily.index <= R.WARMUP[1])
    warmup_days = inside.sum()
    before = int((daily.index[:20] < R.WARMUP[0]).sum())
    assert before > 0  # the days before 2006-07-03 must not count
    assert share.loc[daily.columns[0], "warmup"] == pytest.approx((20 - before) / warmup_days)
    assert share.loc[daily.columns[1], "window"] == 0.0


def test_e5_e6_e7_e8_run_end_to_end_on_synthetic_data(
    panel: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    daily, sectors, spy, vix = panel
    project = load_project(ROOT)
    vix_change = np.log(vix).diff().fillna(0.0)

    table, verdict = R.e5(daily, sectors, spy, vix_change, vix, 20)
    assert verdict["verdict"] in ("supported", "partly supported", "not supported")
    assert {"diff_raw", "vol_share"} <= set(table.columns)

    corr = np.corrcoef(daily.to_numpy(), rowvar=False)
    fake = np.where(np.arange(len(daily))[:, None, None] < 503, np.nan, corr[None])
    monkeypatch.setattr(R, "rolling_dcc", lambda *a, **k: SimpleNamespace(corr_next=fake))
    _vol, h5 = R.e6(daily, daily.resample("W-FRI").sum(), project, 20)
    assert h5["n_cells"] == 5 and h5["window"][0] >= "2008-06-01"

    instruments = pd.DataFrame({name: spy for name in ["SPY", "QQQ", "IWM", "TLT", "GLD", "HYG"]})
    instruments.loc[instruments.index < "2007-04-11", "HYG"] = np.nan
    instruments += np.random.default_rng(1).normal(0, 0.003, instruments.shape)
    frames, h6 = R.e7(daily, spy, instruments, vix, project, 20)
    assert h6["window"] == ["2007-07-02", "2013-12-31"]
    assert frames["lasso"]["from"].iloc[0] >= "2007-10-01"

    phases = R.e8(daily, sectors, spy, project, 10)
    assert set(phases["phase"]) == set(R.PHASES)
    assert len(phases) == 6 * (4 + len(R.HEDGE_STATS))
    pre = phases[phases["phase"] == "pre_crisis"].set_index("statistic")
    assert pre.loc["diff_raw", "value"] == pytest.approx(0.0)
    assert pre.loc["r2_adj", "value"] == pytest.approx(pre.loc["r2", "value"])
    assert np.isfinite(phases[["value", "ci_lower", "ci_upper"]].to_numpy()).all()
