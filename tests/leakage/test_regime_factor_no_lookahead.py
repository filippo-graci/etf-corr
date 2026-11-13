"""Leakage tests for ``etfcorr.correlation.regime_factor`` (step 4.2b, exploratory).

The decomposition is a descriptive, in-sample statistic of a whole sample, not a date-indexed
feature, so the shared truncation detectors apply only to the alignment (tested in 4.2) and are
used here to *prove which parts are not causal*. What can leak in this module, and is tested:

* the regime label of day ``t`` must be the one known at the close of ``t-1`` (predetermined);
* the factor must be joined to the returns by *date*, never by position (a positional join of a
  re-ordered factor silently pairs a return with another day's factor);
* :func:`latent_factors` is a full-sample construction -- the detectors confirm that it is **not**
  causal, which is why its docstring forbids using it inside a training window.

Each check is shown to catch its leaky twin (project_plan.md O4 rule).
"""

import numpy as np
import pandas as pd
import pytest

from etfcorr.correlation import regime_factor as F
from etfcorr.leakage import LeakageError, assert_no_lookahead

N = 700
COLS = list("abcdef")
SECTORS = {c: ("x" if c in "abc" else "y") for c in COLS}


def _panel(seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2018-01-01", periods=N + 1, name="date")
    codes = np.repeat(rng.integers(0, 2, size=N // 50 + 2), 50)[: N + 1].astype(float)
    f = pd.Series(
        0.01 * np.where(codes[:-1] == 1, 2.0, 1.0) * rng.standard_normal(N),
        index=index[1:],
        name="mkt",
    )
    loadings = np.linspace(0.6, 1.4, len(COLS))
    returns = pd.DataFrame(
        np.outer(f.to_numpy(), loadings) + 0.01 * rng.standard_normal((N, len(COLS))),
        index=index[1:],
        columns=COLS,
    )
    return returns, pd.Series(codes, index=index), f


def test_the_last_label_does_not_move_the_table_under_predetermined_alignment() -> None:
    returns, labels, f = _panel()
    flipped = labels.copy()
    flipped.iloc[-1] = 1.0 - labels.iloc[-1]  # the label of a day with no return yet
    a = F.regime_factor_decomposition(returns, labels, SECTORS, f).table
    b = F.regime_factor_decomposition(returns, flipped, SECTORS, f).table
    pd.testing.assert_frame_equal(a, b)
    c = F.regime_factor_decomposition(returns, labels, SECTORS, f, alignment="contemporaneous")
    d = F.regime_factor_decomposition(returns, flipped, SECTORS, f, alignment="contemporaneous")
    assert not c.table.equals(d.table)  # the perturbation can see the selection problem


def test_the_factor_is_joined_by_date_not_by_position() -> None:
    returns, labels, f = _panel()
    base = F.regime_factor_decomposition(returns, labels, SECTORS, f).table
    shuffled = f.sample(frac=1.0, random_state=3)  # same values, same dates, different row order
    assert not shuffled.index.equals(f.index)
    again = F.regime_factor_decomposition(returns, labels, SECTORS, shuffled).table
    pd.testing.assert_frame_equal(base, again)


def test_a_positional_join_of_the_factor_is_caught() -> None:
    returns, labels, f = _panel()
    base = F.regime_factor_decomposition(returns, labels, SECTORS, f).table
    shuffled = f.sample(frac=1.0, random_state=3)
    positional = pd.Series(shuffled.to_numpy(), index=f.index, name="mkt")  # leaky twin
    wrong = F.regime_factor_decomposition(returns, labels, SECTORS, positional).table
    assert not np.allclose(base["loading_ratio"], wrong["loading_ratio"], equal_nan=True)


def test_a_factor_from_the_next_day_changes_the_loadings() -> None:
    """A one-day look-ahead in the factor (tomorrow's factor against today's return) destroys the
    loading the real factor has, so a mis-dated factor cannot pass unnoticed."""
    returns, _, f = _panel()
    real = F.fit_factor_model(returns.to_numpy(), f.to_numpy().reshape(-1, 1))
    ahead = F.fit_factor_model(
        returns.to_numpy()[:-1], f.shift(-1).dropna().to_numpy().reshape(-1, 1)
    )
    assert real.loadings.mean() > 0.5
    assert abs(ahead.loadings.mean()) < 0.15


def test_latent_factors_are_not_causal_and_the_detector_knows_it() -> None:
    """``latent_factors`` uses the whole sample (principal components): the truncation detector
    must flag it, which is the reason it is documented as descriptive-only."""
    returns, _, f = _panel()

    def fn(d: pd.DataFrame) -> pd.DataFrame:
        return F.latent_factors(d, f.reindex(d.index), 1)

    with pytest.raises(LeakageError):
        assert_no_lookahead(fn, returns)


def test_a_causal_factor_passes_the_same_detector() -> None:
    """Proof that the detector does not flag everything: the market factor itself, read at each
    date, is causal."""
    returns, _, f = _panel()

    def fn(d: pd.DataFrame) -> pd.Series:
        return f.reindex(d.index)

    assert_no_lookahead(fn, returns)
