"""Leakage tests for ``etfcorr.correlation.regime_correlation`` (step 4.2).

The H2 engine itself returns a table, not a date-indexed series, so the shared detectors in
``etfcorr.leakage`` apply to the two date-indexed building blocks (regime alignment and
volatility standardisation) and a perturbation test covers the engine's use of the label at the
last date. Every detector is shown to catch its leaky twin (project_plan.md O4 rule). Alignment
and standardisation are the places where H2's look-ahead and selection problems would enter:

* the label used for day ``t`` must be known at the close of ``t-1`` (predetermined alignment);
* the volatility that scales day ``t`` must not contain day ``t``'s own return.
"""

import numpy as np
import pandas as pd
import pytest

from etfcorr.correlation import regime_correlation as R
from etfcorr.leakage import LeakageError, assert_future_invariant, assert_no_lookahead

N = 500


def _label_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    codes = np.repeat(rng.integers(0, 3, size=N // 10 + 1), 10)[:N].astype(float)
    index = pd.bdate_range("2018-01-01", periods=N, name="date")
    return pd.DataFrame({"label": codes}, index=index)


def _returns(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2018-01-01", periods=N, name="date")
    return pd.DataFrame(0.01 * rng.standard_normal((N, 4)), index=index, columns=list("abcd"))


# ---- alignment ----------------------------------------------------------------------


@pytest.mark.parametrize("alignment", ["predetermined", "contemporaneous"])
def test_alignment_has_no_lookahead(alignment: str) -> None:
    def fn(d: pd.DataFrame) -> pd.Series:
        return R.align_regime_labels(d["label"], d.index, alignment)  # type: ignore[arg-type]

    assert_no_lookahead(fn, _label_frame())
    assert_future_invariant(fn, _label_frame(), mode="noise")
    assert_future_invariant(fn, _label_frame(), mode="shuffle")


def test_a_forward_looking_alignment_is_caught() -> None:
    """The look-ahead the predetermined alignment exists to prevent: using tomorrow's label."""

    def leaky(d: pd.DataFrame) -> pd.Series:
        return R.align_regime_labels(d["label"], d.index, "contemporaneous").shift(-1)

    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky, _label_frame())


def test_the_label_of_day_t_never_selects_the_return_of_day_t_when_predetermined() -> None:
    """Changing the regime label of the last day moves nobody under predetermined alignment (it
    would only label tomorrow's return) but does move the day itself under contemporaneous
    alignment -- the selection problem the primary design avoids, and a proof that this
    perturbation can see it."""
    labels = _label_frame()["label"]
    index = labels.index
    flipped = labels.copy()
    flipped.iloc[-1] = (labels.iloc[-1] + 1.0) % 3.0

    pre_a = R.align_regime_labels(labels, index, "predetermined")
    pre_b = R.align_regime_labels(flipped, index, "predetermined")
    pd.testing.assert_series_equal(pre_a, pre_b)

    con_a = R.align_regime_labels(labels, index, "contemporaneous")
    con_b = R.align_regime_labels(flipped, index, "contemporaneous")
    assert not con_a.equals(con_b)


def test_the_engine_ignores_the_last_label_under_predetermined_alignment() -> None:
    rng = np.random.default_rng(5)
    n = 400
    index = pd.bdate_range("2019-01-01", periods=n + 1)
    labels = pd.Series(np.repeat([0.0, 1.0], [200, 201]).astype(float), index=index)
    sectors = {c: ("x" if c in "ab" else "y") for c in "abcd"}
    returns = pd.DataFrame(
        0.01 * rng.standard_normal((n, 4)), index=index[1:], columns=list("abcd")
    )
    flipped = labels.copy()
    flipped.iloc[-1] = 0.0  # the label of a day that has no return yet under predetermined
    a = R.regime_correlations(returns, labels, sectors)
    b = R.regime_correlations(returns, flipped, sectors)
    pd.testing.assert_frame_equal(a.table, b.table)
    c = R.regime_correlations(returns, labels, sectors, alignment="contemporaneous")
    d = R.regime_correlations(returns, flipped, sectors, alignment="contemporaneous")
    assert not c.table["n_obs"].equals(d.table["n_obs"])


# ---- volatility standardisation -----------------------------------------------------


def test_vol_standardize_has_no_lookahead() -> None:
    def fn(d: pd.DataFrame) -> pd.DataFrame:
        return R.vol_standardize(d)

    assert_no_lookahead(fn, _returns())
    assert_future_invariant(fn, _returns(), mode="noise")
    assert_future_invariant(fn, _returns(), mode="shuffle")


def test_ewma_volatility_has_no_lookahead() -> None:
    assert_no_lookahead(lambda d: R.ewma_volatility(d), _returns())


def test_a_centred_volatility_scaling_is_caught() -> None:
    def leaky(d: pd.DataFrame) -> pd.DataFrame:
        return d / d.rolling(21, center=True, min_periods=2).std()

    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky, _returns())


def test_a_full_sample_volatility_scaling_is_caught() -> None:
    def leaky(d: pd.DataFrame) -> pd.DataFrame:
        return d / d.std()

    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky, _returns())
