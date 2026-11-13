"""Tests for scripts/factor_residual_e9.py and scripts/holdout_e10.py, on synthetic data.

The scripts are loaded by path; nothing here reads the project's data or downloads anything.
"""

import importlib.util
from datetime import date
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from etfcorr.config import load_project

ROOT = Path(__file__).resolve().parents[2]


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E9, E10 = load("factor_residual_e9"), load("holdout_e10")

FRENCH_SAMPLE = """This file was created using the 202608 CRSP database.
The Tbill return is the simple daily rate.

,Mkt-RF,SMB,HML,RMW,CMA,RF
20140102,  -0.88,   0.12,  -0.05,   0.10,  -0.02,   0.000
20140103,   0.03,   0.30,   0.02,  -0.11,   0.07,   0.000

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def test_parse_french_csv_reads_daily_rows_in_decimals_and_stops_at_the_footer() -> None:
    frame = E9.parse_french_csv(FRENCH_SAMPLE)
    assert list(frame.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert list(frame.index) == [pd.Timestamp("2014-01-02"), pd.Timestamp("2014-01-03")]
    assert frame.loc["2014-01-02", "Mkt-RF"] == pytest.approx(-0.0088)
    with pytest.raises(SystemExit):
        E9.parse_french_csv("no data here\n")


def factor_world(
    n: int, residual_rho: tuple[float, float], seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, str]]:
    """Six funds in three sectors on two factors; factor volatility and the residual correlation
    each differ between a calm and a loud half. The VIX tracks the factor volatility."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2014-01-02", periods=n)
    loud = np.arange(n) >= n // 2
    f = rng.normal(0, 0.01, (n, 2)) * np.where(loud, 3.0, 1.0)[:, None]
    betas = rng.uniform(0.5, 1.5, (2, 6))
    resid = np.empty((n, 6))
    for half, rho in ((~loud, residual_rho[0]), (loud, residual_rho[1])):
        cov = np.full((6, 6), rho) + np.eye(6) * (1 - rho)
        resid[half] = rng.multivariate_normal(np.zeros(6), cov * 1e-4, half.sum())
    names = [f"F{k}" for k in range(6)]
    returns = pd.DataFrame(f @ betas + resid, index=index, columns=names)
    factors = pd.DataFrame(f, index=index, columns=["A", "B"])
    vix = pd.Series(np.where(loud, 30.0, 12.0) + rng.uniform(0, 1, n), index=index)
    sectors = dict(zip(names, ["s0", "s0", "s1", "s1", "s2", "s2"], strict=True))
    return returns, factors, vix, sectors


def test_regime_split_is_exact_and_full_betas_break_only_the_identity() -> None:
    returns, factors, _, sectors = factor_world(400, (0.2, 0.2))
    mask = E9.pair_mask([sectors[t] for t in returns.columns], "cross")
    y, f = returns.to_numpy(), factors.to_numpy()
    parts = E9.regime_split(y, f, mask)
    assert parts["raw"] == pytest.approx(parts["factor_part"] + parts["residual_part"], abs=1e-12)
    x = np.column_stack([np.ones(len(f)), f])
    other = np.linalg.lstsq(x[:200], y[:200], rcond=None)[0]
    shifted = E9.regime_split(y, f, mask, other)
    assert shifted["raw"] == pytest.approx(parts["raw"])
    assert shifted["raw"] != pytest.approx(shifted["factor_part"] + shifted["residual_part"])


@pytest.mark.parametrize(("rho", "positive"), [((0.1, 0.1), False), ((0.05, 0.45), True)])
def test_the_statistic_sees_residual_co_movement_and_not_factor_volatility(
    rho: tuple[float, float], positive: bool
) -> None:
    returns, factors, vix, sectors = factor_world(1500, rho, seed=1)
    codes = E9.tercile_codes(vix, 1)
    table = E9.line(returns, factors, codes, sectors, n_bootstrap=60, alpha=0.05)
    assert table.loc["raw_diff", "value"] > 0.05  # the louder factors raise raw correlation anyway
    s = table.loc["residual_corr_diff", "value"]
    p = table.attrs["p_one_sided"]
    if positive:
        assert s == pytest.approx(0.40, abs=0.08) and p < 0.05
    else:
        assert abs(s) < 0.06 and p > 0.05
    assert table.attrs["n_low"] > 0 and table.attrs["n_high"] > 0


def test_e10_refuses_early_or_on_short_data() -> None:
    coverage = pd.DataFrame(
        {
            "ticker": ["A", "B", "SPY"],
            "panel": ["core", "core", "control"],
            "last_date": ["2027-09-30", "2027-09-30", "2027-01-04"],
        }
    )
    E10.check_ready(date(2027, 10, 7), coverage)
    with pytest.raises(SystemExit, match="may not run"):
        E10.check_ready(date(2027, 10, 6), None)  # the date is checked before the data
    with pytest.raises(SystemExit, match="data commands"):
        E10.check_ready(date(2027, 10, 7), None)
    coverage.loc[1, "last_date"] = "2027-09-29"
    with pytest.raises(SystemExit, match="stop before"):
        E10.check_ready(date(2027, 12, 1), coverage)


def test_e10_reads_the_four_long_short_nonlinear_cells() -> None:
    rng = np.random.default_rng(4)
    index = pd.bdate_range("2017-01-02", "2027-09-30")
    common = rng.normal(0, 0.008, len(index))
    daily = pd.DataFrame(
        common[:, None] + rng.normal(0, 0.01, (len(index), 8)),
        index=index,
        columns=[f"F{k}" for k in range(8)],
    )
    panels = {"D": daily}
    for freq, rule in (("W", "W-FRI"), ("M", "ME"), ("Q", "QE")):
        panels[freq] = daily.resample(rule).sum()
    design = load_project(ROOT).experiment.economics
    assert design is not None
    cells, _pooled, reading = E10.evaluate(panels, design, lam=0.94, n_bootstrap=50, alpha=0.05)
    assert reading["n_cells"] == 4 and reading["portfolio"] == "minvar_ls"
    assert set(cells["frequency"]) == {"D", "W", "M", "Q"}
    assert cells["n_days"].iloc[0] == ((index >= "2026-10-01") & (index <= "2027-09-30")).sum()
    assert isinstance(reading["supported"], bool)
