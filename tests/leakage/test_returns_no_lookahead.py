"""Leakage tests for ``etfcorr.returns`` (step 2.1).

``project_plan.md`` section 9 requires the no-lookahead and shuffle/noise-the-future tests of
every pure ``f(data_up_to_t) -> value_at_t`` function; ``horizon_returns`` is the first such
function in the codebase outside ``etfcorr.leakage`` itself (Phase 0's tests exercise the
detectors on toy examples, not on a real pipeline function). The one leakage-relevant design
choice in ``returns.py`` is the incomplete-final-period guard (see its docstring); this file also
demonstrates, per the project's own O4 rule ("each leakage test must also fail on a deliberately
leaky version"), that :func:`etfcorr.leakage.assert_no_lookahead` correctly flags
``horizon_returns(..., drop_incomplete=False)`` at W/M/Q.

Uses the session-scoped ``factor_panel`` fixture from ``tests/conftest.py`` (correlated synthetic
returns, not prices); prices are reconstructed as a geometric random walk
(``100 * exp(cumsum(returns))``) purely so ``horizon_returns`` has something price-shaped to
operate on -- the *correlation* structure of ``factor_panel`` is irrelevant here (returns.py
processes each column independently) and is not exercised by these tests.
"""

import numpy as np
import pandas as pd
import pytest

from etfcorr.leakage import LeakageError, assert_future_invariant, assert_no_lookahead
from etfcorr.returns import horizon_returns
from etfcorr.synthetic import SyntheticPanel

HORIZONS = ("D", "W", "M", "Q")

# Explicit cut points, not etfcorr.leakage's default fractions (0.40/0.55/0.70/0.85/0.97).
# factor_panel is 1500 business days starting on a Monday: 1500 = 300 exact 5-day weeks, and
# every default fraction * 1500 happens to be a multiple of 5 -- so every default cut lands
# exactly on a Friday (a week/month/quarter boundary here), computed and confirmed directly
# rather than assumed: with the default cuts, `horizon_returns` never has an incomplete final
# period to drop at all, at W, M, *or* Q, making the no-lookahead check vacuous for the one
# leakage-relevant design choice this module has (the incomplete-period guard). These five cuts
# were instead chosen, and confirmed by direct computation, to fall strictly *inside* a week, a
# month and a quarter simultaneously (Wed/Thu-ish days, never a period's last session).
CUTS = (598, 823, 1048, 1273, 1453)


def _prices_from_returns(panel: SyntheticPanel) -> pd.DataFrame:
    return 100.0 * np.exp(panel.returns.cumsum())


@pytest.fixture(scope="module")
def prices(factor_panel: SyntheticPanel) -> pd.DataFrame:
    return _prices_from_returns(factor_panel)


@pytest.mark.parametrize("horizon", HORIZONS)
def test_horizon_returns_has_no_lookahead(prices: pd.DataFrame, horizon: str) -> None:
    assert_no_lookahead(lambda df: horizon_returns(df, horizon), prices, cuts=CUTS)  # type: ignore[arg-type]


@pytest.mark.parametrize("horizon", HORIZONS)
def test_horizon_returns_is_invariant_to_shuffling_the_future(
    prices: pd.DataFrame, horizon: str
) -> None:
    """``mode="shuffle"`` only, not ``"noise"``: see the module docstring below."""
    assert_future_invariant(
        lambda df: horizon_returns(df, horizon),  # type: ignore[arg-type]
        prices,
        cuts=CUTS,
        mode="shuffle",
    )


def test_noise_mode_is_not_applicable_to_price_valued_functions(prices: pd.DataFrame) -> None:
    """Documents why this file does not also run ``assert_future_invariant(..., mode="noise")``.

    ``replace_future``'s noise mode draws Gaussian replacement values from each column's own
    mean and standard deviation over the *past* rows (``etfcorr.leakage``, by design, for
    return-like series centred near zero). Prices are not centred near zero and must stay
    strictly positive; over a long, drifting geometric random walk like ``factor_panel``'s, a
    column's past mean/std can imply a Gaussian wide enough to sample a non-positive draw, which
    ``etfcorr.returns.log_returns`` -- correctly -- refuses to log. This is not a leak, and not a
    bug in either module: it means the noise-replacement mode is the wrong tool for a
    price-valued feature function, so this file uses ``shuffle`` (a permutation of *actually
    observed* future prices, hence always positive) as its future-invariance check instead.
    Confirmed empirically this session (not assumed): running the noise-mode check against
    ``horizon_returns`` on this exact fixture raises ``etfcorr.returns.ReturnsError`` from a
    negative sampled price, for horizons "D" and "W".
    """
    from etfcorr.returns import ReturnsError

    with pytest.raises(ReturnsError):
        assert_future_invariant(
            lambda df: horizon_returns(df, "D"),  # type: ignore[arg-type]
            prices,
            cuts=CUTS,
            mode="noise",
        )


@pytest.mark.parametrize("horizon", ("W", "M", "Q"))
def test_without_the_incomplete_period_guard_the_detector_correctly_flags_a_leak(
    prices: pd.DataFrame, horizon: str
) -> None:
    """Deliberately leaky twin: skip the incomplete-final-period guard.

    A truncated run then reports one extra return, for the still-forming final period, labelled
    at that period's future end date -- a date strictly after the truncation cut itself. The
    detector must catch this (index/shape mismatch against the un-truncated run restricted to
    the same dates), proving it is not a vacuous check for this module. (Not run for "D": the
    guard does not apply there -- see ``horizon_returns``.) Uses the same non-boundary ``CUTS``
    as above: with the default fractions this test would not detect anything, for the same
    reason the main no-lookahead test would be vacuous with them (see ``CUTS``' comment).
    """

    def leaky(df: pd.DataFrame) -> pd.DataFrame:
        return horizon_returns(df, horizon, drop_incomplete=False)  # type: ignore[arg-type]

    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky, prices, cuts=CUTS)
