"""Leakage tests for ``etfcorr.regimes`` (step 4.1), using the shared ``etfcorr.leakage`` detectors.

Every regime function here that is declared causal is checked by truncation and by replacing the
future; the two deliberately non-causal or leaky variants must be *caught*, so a green run proves
the detectors can see the leak they are meant to see (project_plan.md O4 rule).

Mode choice, per the finding recorded in docs/DECISION_LOG.md for step 2.1: the ``noise`` mode of
``assert_future_invariant`` draws Gaussian replacements from the past mean and standard deviation,
which can be negative, so it is used only on a return-like series. Level series that must stay
positive (VIX, prices) use the ``shuffle`` mode, which permutes real, positive future values.
"""

import numpy as np
import pandas as pd
import pytest

from etfcorr.leakage import LeakageError, assert_future_invariant, assert_no_lookahead
from etfcorr.regimes import (
    drawdown,
    drawdown_states,
    lag_regimes,
    quantile_regimes,
    vix_terciles,
)

N = 800
MIN_OBS = 60
TERCILES = (1 / 3, 2 / 3)


def _levels(seed: int = 0) -> pd.DataFrame:
    """Positive VIX-like level (log-AR(1)) and a positive price path, one column each."""
    rng = np.random.default_rng(seed)
    log_v = np.empty(N)
    log_v[0] = np.log(18.0)
    for t in range(1, N):
        log_v[t] = (
            np.log(18.0) + 0.97 * (log_v[t - 1] - np.log(18.0)) + 0.06 * rng.standard_normal()
        )
    price = 100.0 * np.exp(np.cumsum(0.0004 + 0.01 * rng.standard_normal(N)))
    index = pd.bdate_range("2018-01-01", periods=N, name="date")
    return pd.DataFrame({"vix": np.exp(log_v), "price": price}, index=index)


def _gaussian(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2018-01-01", periods=N, name="date")
    return pd.DataFrame({"x": rng.standard_normal(N)}, index=index)


# ---- causal functions pass -----------------------------------------------------------


def test_expanding_vix_terciles_have_no_lookahead() -> None:
    assert_no_lookahead(lambda d: vix_terciles(d["vix"], min_obs=MIN_OBS), _levels())


def test_expanding_vix_terciles_ignore_a_shuffled_future() -> None:
    assert_future_invariant(
        lambda d: vix_terciles(d["vix"], min_obs=MIN_OBS), _levels(), mode="shuffle"
    )


def test_expanding_quantile_regimes_ignore_a_noisy_future() -> None:
    assert_future_invariant(
        lambda d: quantile_regimes(d["x"], TERCILES, min_obs=MIN_OBS), _gaussian(), mode="noise"
    )


def test_expanding_quantile_regimes_have_no_lookahead_on_gaussian_data() -> None:
    assert_no_lookahead(lambda d: quantile_regimes(d["x"], TERCILES, min_obs=MIN_OBS), _gaussian())


def test_drawdown_and_states_have_no_lookahead() -> None:
    data = _levels()
    assert_no_lookahead(lambda d: drawdown(d["price"]), data)
    assert_no_lookahead(lambda d: drawdown_states(d["price"]), data)
    assert_future_invariant(lambda d: drawdown_states(d["price"]), data, mode="shuffle")


def test_lagged_regimes_have_no_lookahead() -> None:
    assert_no_lookahead(
        lambda d: lag_regimes(vix_terciles(d["vix"], min_obs=MIN_OBS), 1), _levels()
    )


# ---- non-causal / leaky variants are caught -----------------------------------------


def test_full_sample_mode_is_flagged_by_the_truncation_detector() -> None:
    with pytest.raises(LeakageError):
        assert_no_lookahead(
            lambda d: vix_terciles(d["vix"], mode="full_sample", min_obs=MIN_OBS), _levels()
        )


def test_full_sample_mode_is_flagged_by_the_noisy_future_detector() -> None:
    with pytest.raises(LeakageError):
        assert_future_invariant(
            lambda d: quantile_regimes(d["x"], TERCILES, mode="full_sample", min_obs=MIN_OBS),
            _gaussian(),
            mode="noise",
        )


def test_shuffle_mode_cannot_see_full_sample_cutpoints() -> None:
    """Documented detector limit (etfcorr.leakage docstring): permuting the future leaves every
    full-sample quantile unchanged, so this variant passes the shuffle detector. Truncation and
    noise (above) are the primary detectors for exactly this reason."""
    assert_future_invariant(
        lambda d: vix_terciles(d["vix"], mode="full_sample", min_obs=MIN_OBS),
        _levels(),
        mode="shuffle",
    )


def test_drawdown_against_the_full_sample_maximum_is_flagged() -> None:
    def leaky(d: pd.DataFrame) -> pd.Series:
        return d["price"] / d["price"].max() - 1.0

    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky, _levels())


def test_a_negative_shift_is_flagged() -> None:
    def leaky(d: pd.DataFrame) -> pd.Series:
        return vix_terciles(d["vix"], min_obs=MIN_OBS).shift(-1)

    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky, _levels())
