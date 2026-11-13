"""Leakage and purity tests for ``etfcorr.correlation.lead_lag`` (step 4.5).

A scan is a whole-window statistic, not a date-indexed feature function, so the shared detectors of
``etfcorr.leakage`` do not apply to it directly (same situation as ``estimators.py``; see
``tests/leakage/test_estimators_purity.py``). What can leak here, and what these tests pin down:

* **direction**: the leader must be the *earlier* series. A forward shift would silently flip every
  sign; a deliberately wrong twin is shown to do exactly that, and the real function does not;
* **the window**: a discovery scan must be a function of the rows inside its window only, so the
  validation and test windows cannot influence which (pair, lag) cells are discovered;
* **selection**: the discovered set must not depend on the validation data it is later replicated
  on;
* **time order**: the statistic must come from the time order (shuffling the rows removes it);
* **downstream use**: a lagged-leader feature built from a discovered pair passes the shared
  detectors when it is a backward ``shift`` and fails when it is a forward one.
"""

import numpy as np
import pandas as pd
import pytest

from etfcorr.correlation import lead_lag as ll
from etfcorr.leakage import LeakageError, assert_future_invariant, assert_no_lookahead

BOOT = 300


def _planted(n: int = 6000, seed: int = 1) -> pd.DataFrame:
    return ll.simulate_lead_lag(n, 4, market_loading=0.0, garch=None, df=None, seed=seed,
                                planted=[(0, 1, 1, 0.3)])  # fmt: skip


def _cell(scan: ll.LeadLagScan, a: str = "A00", b: str = "A01", lag: int = 1) -> pd.Series:
    t = scan.table
    return t[(t["asset_a"] == a) & (t["asset_b"] == b) & (t["lag"] == lag)].iloc[0]


def _leaky_asymmetry(frame: pd.DataFrame, i: int, j: int, k: int) -> float:
    """Deliberately wrong twin: the 'leader' is taken k rows in the FUTURE."""
    u = ll.standardise(frame)
    n = len(u)
    c_ij = float(np.mean(u[: n - k, j] * u[k:, i]))  # u_i at t+k instead of t-k
    c_ji = float(np.mean(u[: n - k, i] * u[k:, j]))
    return c_ij - c_ji


def _correct_asymmetry(frame: pd.DataFrame, i: int, j: int, k: int) -> float:
    u = ll.standardise(frame)
    n = len(u)
    c_ij = float(np.mean(u[k:, j] * u[: n - k, i]))
    c_ji = float(np.mean(u[k:, i] * u[: n - k, j]))
    return c_ij - c_ji


def test_leader_is_the_earlier_series_and_a_forward_shift_flips_the_sign() -> None:
    frame = _planted()
    scan = ll.lead_lag_scan(frame, lags=(1,), n_bootstrap=BOOT)
    real = float(_cell(scan)["estimate"])
    assert real > 0.2
    assert _correct_asymmetry(frame, 0, 1, 1) > 0.2
    leaky = _leaky_asymmetry(frame, 0, 1, 1)
    assert leaky < -0.2  # the leaky twin reports the opposite direction: the test can see it
    assert real * leaky < 0


def test_scan_inside_a_window_ignores_every_row_outside_it() -> None:
    frame = _planted(1500, seed=2)
    cut = str(frame.index[999].date())
    base = ll.lead_lag_scan(ll.slice_window(frame, "2016-01-01", cut), n_bootstrap=BOOT)
    rng = np.random.default_rng(0)
    altered = frame.copy()
    # wreck the "future"
    altered.iloc[1000:] = rng.normal(0.0, 0.05, size=altered.iloc[1000:].shape)
    again = ll.lead_lag_scan(ll.slice_window(altered, "2016-01-01", cut), n_bootstrap=BOOT)
    pd.testing.assert_frame_equal(base.table, again.table)
    assert ll.slice_window(altered, "2016-01-01", cut).index.max() <= pd.Timestamp(cut)


def test_a_scan_that_peeks_past_the_window_is_caught() -> None:
    """Control: scanning the unsliced frame does change the answer, so the test above has power."""
    frame = _planted(1500, seed=2)
    cut = str(frame.index[999].date())
    altered = frame.copy()
    altered.iloc[1000:] = np.random.default_rng(0).normal(0.0, 0.05, size=altered.iloc[1000:].shape)
    a = ll.lead_lag_scan(frame, n_bootstrap=BOOT)
    b = ll.lead_lag_scan(altered, n_bootstrap=BOOT)
    assert not np.allclose(a.table["estimate"], b.table["estimate"])
    assert ll.slice_window(frame, "2016-01-01", cut).index.max() < frame.index[-1]


def test_the_discovered_set_does_not_depend_on_the_validation_data() -> None:
    disc = _planted(6000, seed=3)
    d = ll.lead_lag_scan(disc, lags=(1, 2), n_bootstrap=BOOT, seed=1)
    val_a = _planted(3000, seed=4)
    val_b = ll.simulate_lead_lag(3000, 4, market_loading=0.0, garch=None, df=None, seed=6)
    keys = ["asset_a", "asset_b", "lag"]
    ra = ll.replication_table(d, ll.lead_lag_scan(val_a, lags=(1, 2), n_bootstrap=BOOT, seed=2))
    rb = ll.replication_table(d, ll.lead_lag_scan(val_b, lags=(1, 2), n_bootstrap=BOOT, seed=2))
    pd.testing.assert_frame_equal(ra[keys], rb[keys])
    pd.testing.assert_series_equal(ra["z_disc"], rb["z_disc"])
    # the validation data do matter for the replication verdict
    assert list(ra["replicated"]) != list(rb["replicated"])


def test_shuffling_the_row_order_removes_the_lead_lag() -> None:
    frame = _planted(8000, seed=6)
    ordered = ll.lead_lag_scan(frame, lags=(1,), n_bootstrap=BOOT)
    shuffled = pd.DataFrame(
        frame.to_numpy()[np.random.default_rng(0).permutation(len(frame))],
        index=frame.index,
        columns=frame.columns,
    )
    scrambled = ll.lead_lag_scan(shuffled, lags=(1,), n_bootstrap=BOOT)
    assert abs(_cell(ordered)["z"]) > 6.0
    assert abs(_cell(scrambled)["z"]) < 4.0
    assert not scrambled.table["reject_bh"].any()


def test_scan_is_pure_in_its_input() -> None:
    frame = _planted(2000, seed=7)
    snapshot = frame.copy(deep=True)
    first = ll.lead_lag_scan(frame, n_bootstrap=BOOT, keep_null=True)
    second = ll.lead_lag_scan(frame, n_bootstrap=BOOT, keep_null=True)
    pd.testing.assert_frame_equal(frame, snapshot)
    pd.testing.assert_frame_equal(first.table, second.table)
    assert first.null_z is not None and second.null_z is not None
    np.testing.assert_array_equal(first.null_z, second.null_z)


# --- downstream contract: how a discovered pair may be used as a feature (Phases 5-6) ----------


def _lagged_leader(frame: pd.DataFrame) -> pd.Series:
    return frame["A00"].shift(1)


def _lagged_leader_leaky(frame: pd.DataFrame) -> pd.Series:
    return frame["A00"].shift(-1)  # the leader's NEXT-day return: look-ahead


def test_a_backward_shift_feature_passes_the_shared_detectors() -> None:
    frame = _planted(600, seed=8)
    assert_no_lookahead(lambda d: _lagged_leader(d).to_frame("f"), frame)
    assert_future_invariant(lambda d: _lagged_leader(d).to_frame("f"), frame, mode="noise")
    assert_future_invariant(lambda d: _lagged_leader(d).to_frame("f"), frame, mode="shuffle")


def test_a_forward_shift_feature_is_caught_by_the_shared_detectors() -> None:
    frame = _planted(600, seed=8)
    with pytest.raises(LeakageError):
        assert_no_lookahead(lambda d: _lagged_leader_leaky(d).to_frame("f"), frame)
    with pytest.raises(LeakageError):
        assert_future_invariant(
            lambda d: _lagged_leader_leaky(d).to_frame("f"), frame, mode="noise"
        )
