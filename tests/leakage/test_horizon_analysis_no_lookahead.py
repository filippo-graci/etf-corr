"""Leakage tests for step 2.3's rolling/EWMA coupling indicator.

Unlike `tests/leakage/test_estimators_purity.py` (step 2.2, a bespoke purity suite because that
module's point estimators are ticker-indexed, not date-indexed -- see that file's own docstring),
`rolling_coupling_indicator`/`ewma_coupling_indicator` produce exactly the shape
`etfcorr.leakage`'s shared detectors are built for (one scalar per date), so this file uses
`assert_no_lookahead`/`assert_future_invariant` directly -- the first module in the project able
to. Per the project's O4 rule ("each leakage test must also fail on a deliberately leaky
version"), every use of a real detector here is paired with a deliberately-broken twin shown to
be caught, not merely asserted to work on the honest function.

`mode="noise"` (not "shuffle") is used for the future-invariance checks: the data here is
return-like (small numbers, meant to be roughly centred near zero), not price-valued, so the
`mode="noise"` caveat recorded in `docs/STATUS.md`'s 2026-09-27 step-2.1 "testing finding #1"
entry (noise mode unsuitable for price-valued functions specifically) does not apply.
"""

import numpy as np
import pandas as pd
import pytest

from etfcorr.correlation import horizon_analysis as H
from etfcorr.leakage import LeakageError, assert_future_invariant, assert_no_lookahead

WINDOW = 30


def _small_panel(n_days: int = 250, n_assets: int = 5, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n_days, name="date")
    return pd.DataFrame(
        rng.standard_normal((n_days, n_assets)) * 0.01,
        index=index,
        columns=[f"T{i}" for i in range(n_assets)],
    )


# --------------------------------------------------------------------------------------
# rolling_coupling_indicator: real detectors, honest function
# --------------------------------------------------------------------------------------


def test_rolling_coupling_indicator_has_no_lookahead() -> None:
    data = _small_panel()

    def fn(returns: pd.DataFrame) -> pd.Series:
        return H.rolling_coupling_indicator(returns, "pearson", WINDOW)

    assert_no_lookahead(fn, data)


def test_rolling_coupling_indicator_is_future_invariant() -> None:
    data = _small_panel()

    def fn(returns: pd.DataFrame) -> pd.Series:
        return H.rolling_coupling_indicator(returns, "pearson", WINDOW)

    assert_future_invariant(fn, data, mode="noise")


# --------------------------------------------------------------------------------------
# rolling_coupling_indicator: deliberately-leaky twins, shown to be caught (O4 rule)
# --------------------------------------------------------------------------------------


def test_no_lookahead_catches_a_centred_window_twin() -> None:
    """A centred window (uses `window // 2` rows on each side of `i`, not a trailing window) is
    exactly the leak shape `project_plan.md` section 9's own worked example warns about
    ("it also correctly flags a deliberately leaky centred rolling mean")."""
    data = _small_panel()

    def leaky_centred(returns: pd.DataFrame) -> pd.Series:
        half = WINDOW // 2
        dates = []
        values = []
        n = len(returns)
        for i in range(half, n - half):
            corr = returns.iloc[i - half : i + half].corr()
            dates.append(returns.index[i])
            values.append(H.average_offdiagonal_corr(corr))
        return pd.Series(values, index=pd.Index(dates, name=returns.index.name), dtype=float)

    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky_centred, data)


def test_no_lookahead_catches_a_closure_over_the_full_frame() -> None:
    """A classic leak: the function ignores the frame it was actually given and closes over the
    full, un-truncated panel captured earlier -- e.g. a notebook cell that recomputes from a
    stale outer-scope `returns` variable instead of its own argument."""
    data = _small_panel()
    full = data  # captured once, "forgotten" by the leaky function below

    def leaky_ignores_argument(returns: pd.DataFrame) -> pd.Series:
        return H.rolling_coupling_indicator(full, "pearson", WINDOW)

    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky_ignores_argument, data)


def test_future_invariant_catches_a_function_that_peeks_a_few_rows_ahead() -> None:
    data = _small_panel()

    def leaky_peeks_ahead(returns: pd.DataFrame) -> pd.Series:
        peek = 5
        dates = []
        values = []
        n = len(returns)
        for i in range(WINDOW - 1, n):
            end = min(i + 1 + peek, n)
            corr = returns.iloc[max(0, end - WINDOW) : end].corr()
            dates.append(returns.index[i])
            values.append(H.average_offdiagonal_corr(corr))
        return pd.Series(values, index=pd.Index(dates, name=returns.index.name), dtype=float)

    with pytest.raises(LeakageError):
        assert_future_invariant(leaky_peeks_ahead, data, mode="noise")


# --------------------------------------------------------------------------------------
# ewma_coupling_indicator: real detectors, honest function, plus a leaky twin
# --------------------------------------------------------------------------------------


def test_ewma_coupling_indicator_has_no_lookahead() -> None:
    data = _small_panel()

    def fn(returns: pd.DataFrame) -> pd.Series:
        return H.ewma_coupling_indicator(returns, lam=0.9, min_periods=3)

    assert_no_lookahead(fn, data)


def test_ewma_coupling_indicator_is_future_invariant() -> None:
    data = _small_panel()

    def fn(returns: pd.DataFrame) -> pd.Series:
        return H.ewma_coupling_indicator(returns, lam=0.9, min_periods=3)

    assert_future_invariant(fn, data, mode="noise")


def test_no_lookahead_catches_ewma_ignoring_its_argument() -> None:
    data = _small_panel()
    full = data

    def leaky_ignores_argument(returns: pd.DataFrame) -> pd.Series:
        return H.ewma_coupling_indicator(full, lam=0.9, min_periods=3)

    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky_ignores_argument, data)
