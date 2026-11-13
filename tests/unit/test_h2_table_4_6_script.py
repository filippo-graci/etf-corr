"""Tests for the orchestration in ``scripts/h2_table_4_6.py`` (step 4.6), on synthetic data only.

The script is loaded by path (``scripts/`` is not a package). The real-data load is not exercised
here (no data files offline: Gate 2); what is tested is everything after the load: the window
mask, the study (verdict plus robustness lines), the table and JSON payloads, and the check against
the values recorded for step 4.2.
"""

import importlib.util
import json
import math
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from etfcorr.correlation import h2_verdict as V
from etfcorr.correlation.regime_correlation import H2Stat

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "h2_table_4_6.py"
NAMES = [f"S{s}_{k}" for s in range(3) for k in range(3)]
SECTORS = {n: n.split("_")[0] for n in NAMES}


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("h2_table_4_6", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load_script()


def _panel(seed: int = 7, n: int = 1500) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Factor variance x7 and idiosyncratic x2 in the high regime (persistent labels)."""
    rng = np.random.default_rng(seed)
    labels = np.empty(n + 1)
    t, code = 0, 0
    while t < n + 1:
        length = int(rng.geometric(1.0 / 40.0))
        labels[t : t + length] = code
        t += length
        code = (code + 1 + int(rng.integers(0, 2))) % 3
    gov = labels[:-1].astype(int)
    fvol = np.array([1.0, 3.0**0.5, 7.0**0.5])[gov]
    evol = np.array([1.0, 1.5**0.5, 2.0**0.5])[gov]
    x = 0.01 * 0.8 * fvol[:, None] * rng.standard_normal((n, 1))
    x = x + 0.01 * evol[:, None] * rng.standard_normal((n, len(NAMES)))
    index = pd.bdate_range("2018-01-01", periods=n + 1, name="date")
    returns = pd.DataFrame(x, index=index[1:], columns=NAMES)
    vix_change = pd.Series(
        np.array([1.0, 1.5, 2.5])[gov] * rng.standard_normal(n), index=index[1:], name="^VIX"
    )
    spy = returns.mean(axis=1).rename("SPY")
    return returns, pd.Series(labels, index=index, name="regime"), spy, vix_change


def test_mask_window_blanks_only_the_window_and_keeps_the_calendar() -> None:
    returns, *_ = _panel(n=200)
    start, end = str(returns.index[50].date()), str(returns.index[70].date())
    out = S.mask_window(returns, start, end)
    assert out.index.equals(returns.index) and out.shape == returns.shape
    inside = (out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))
    assert out.loc[inside].isna().all().all()
    pd.testing.assert_frame_equal(out.loc[~inside], returns.loc[~inside])
    assert returns.notna().all().all()  # the input is not modified


@pytest.fixture(scope="module")
def study() -> object:
    returns, labels, spy, vix_change = _panel()
    window = (str(returns.index[600].date()), str(returns.index[700].date()))
    return S.run_study(
        returns, labels, SECTORS, spy, vix_change, n_boot=30, seed=0, covid_window=window
    )


def test_the_study_has_the_rule_lines_and_the_robustness_lines(study: object) -> None:
    assert study.primary.name == "primary" and study.primary.delta_mode == "pair"
    assert [m.name for m in study.market] == ["SPY return delta", "VIX log-change delta"]
    assert all(m.delta_mode == "series" for m in study.market)
    names = [line.name for line, _ in study.variants]
    assert names[0].startswith("day-t") and names[-1].startswith("within-sector")
    assert "mean block length 60" in names and "mean block length 120" in names
    assert sum(n.startswith("without ") for n in names) == 1
    assert study.verdict.verdict == "supported"  # the planted world of the unit tests, seed 7
    # the window variant really lost rows
    window_line = next(line for line, _ in study.variants if line.name.startswith("without"))
    assert window_line.n_rows < study.primary.n_rows


def test_the_verdict_does_not_depend_on_the_robustness_lines() -> None:
    returns, labels, spy, vix_change = _panel()
    a = S.run_study(returns, labels, SECTORS, spy, vix_change, n_boot=30, covid_window=None)
    window = (str(returns.index[600].date()), str(returns.index[700].date()))
    b = S.run_study(returns, labels, SECTORS, spy, vix_change, n_boot=30, covid_window=window)
    assert a.verdict == b.verdict
    assert len(b.variants) == len(a.variants) + 1


def test_table_and_json_payload_are_complete_and_serialisable(study: object) -> None:
    table = S.table_frame(study)
    assert len(table) == 1 + 2 + len(study.variants)
    assert list(table["role"][:3]) == ["primary", "market", "market"]
    for stat in ("diff_raw", "diff_adjusted", "vol_share"):
        assert {stat, f"{stat}_lower", f"{stat}_upper"} <= set(table.columns)
    payload = S.verdict_payload(study, [])
    text = json.dumps(payload, allow_nan=False)  # NaN must have been mapped to null
    back = json.loads(text)
    assert back["verdict"] == study.verdict.verdict and back["reproduces_4_2"] is True
    assert back["settings"]["n_bootstrap"] == 30 and back["settings"]["n_cross_sector_pairs"] == 27
    assert len(back["robustness"]) == len(study.variants)
    assert "VERDICT: H2 SUPPORTED" in "\n".join(S.report_lines(study, []))


def test_json_maps_an_undefined_share_interval_to_null() -> None:
    assert S._num(float("nan")) is None and S._num(0.5) == 0.5


def _stat(name: str, point: float, lo: float, hi: float) -> H2Stat:
    return H2Stat(name, 2, point, lo, hi, np.array([point]))


def _recorded_study(study: object, share_shift: float = 0.0) -> object:
    """A copy of the study whose three rule lines carry exactly the recorded 4.2 numbers."""
    rec = S.RECORDED_4_2

    def line(base: V.H2Line, key: str) -> V.H2Line:
        values = rec[key]
        raw = values.get("diff_raw", (0.235, 0.159, 0.305))
        adj = values["diff_adjusted"]
        share = values["vol_share"]
        lo = share[1] if share[1] is not None else math.nan
        hi = share[2] if share[2] is not None else math.nan
        return replace(
            base,
            diff_raw=_stat("diff_raw", *raw),
            diff_adjusted=_stat("diff_adjusted", *adj),
            vol_share=_stat("vol_share", share[0] + share_shift, lo, hi),
        )

    return replace(
        study,
        primary=line(study.primary, "primary"),
        market=[
            line(study.market[0], "SPY return delta"),
            line(study.market[1], "VIX log-change delta"),
        ],
    )


def test_reproduction_check_accepts_the_recorded_numbers_and_flags_a_difference(
    study: object,
) -> None:
    exact = _recorded_study(study)
    assert S.check_reproduction(exact, with_intervals=True) == []
    problems = S.check_reproduction(_recorded_study(study, share_shift=0.01), with_intervals=True)
    assert problems and all("vol_share" in p for p in problems)
    assert any(p.startswith("primary vol_share point") for p in problems)


def test_reproduction_check_ignores_intervals_in_quick_mode(study: object) -> None:
    shifted = _recorded_study(study)
    shifted = replace(
        shifted,
        primary=replace(shifted.primary, diff_raw=_stat("diff_raw", 0.235, 0.0, 0.5)),
    )
    assert S.check_reproduction(shifted, with_intervals=False) == []
    assert S.check_reproduction(shifted, with_intervals=True) != []


def test_reproduction_check_flags_a_non_finite_value(study: object) -> None:
    broken = _recorded_study(study)
    broken = replace(
        broken, primary=replace(broken.primary, vol_share=_stat("vol_share", math.nan, 0, 0))
    )
    assert any("vol_share point" in p for p in S.check_reproduction(broken, with_intervals=False))
