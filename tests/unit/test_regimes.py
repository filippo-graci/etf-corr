"""Unit tests for ``etfcorr.regimes`` (step 4.1): known answers, edge cases, failure cases."""

import numpy as np
import pandas as pd
import pytest

from etfcorr.regimes import (
    HIGH,
    LOW,
    MID,
    RegimeError,
    drawdown,
    drawdown_states,
    lag_regimes,
    quantile_regimes,
    regime_summary,
    vix_terciles,
)


def _series(values: list[float] | np.ndarray, name: str = "x") -> pd.Series:
    index = pd.bdate_range("2020-01-01", periods=len(values), name="date")
    return pd.Series(np.asarray(values, dtype=float), index=index, name=name)


def _synthetic_vix(n: int, seed: int) -> pd.Series:
    """Positive, persistent, VIX-like level series (log-AR(1))."""
    rng = np.random.default_rng(seed)
    log_v = np.empty(n)
    log_v[0] = np.log(18.0)
    for t in range(1, n):
        log_v[t] = (
            np.log(18.0) + 0.97 * (log_v[t - 1] - np.log(18.0)) + 0.06 * rng.standard_normal()
        )
    return _series(np.exp(log_v), "vix")


# --------------------------------------------------------------------------------------
# quantile_regimes / vix_terciles: known answers
# --------------------------------------------------------------------------------------


def test_codes_are_ordered_floats() -> None:
    assert (LOW, MID, HIGH) == (0.0, 1.0, 2.0)


def test_full_sample_terciles_known_answer() -> None:
    # 1..30: numpy's linear-interpolated terciles are 1 + 29/3 = 10.667 and 1 + 58/3 = 20.333.
    labels = vix_terciles(_series(np.arange(1, 31)), mode="full_sample", min_obs=10)
    assert labels.dtype == float
    assert list(labels.iloc[:10]) == [LOW] * 10
    assert list(labels.iloc[10:20]) == [MID] * 10
    assert list(labels.iloc[20:]) == [HIGH] * 10


def test_expanding_known_answer_hand_computed() -> None:
    # x = [5, 1, 3, 2, 4], min_obs = 3. Cutpoints at each date are the terciles of the data so far:
    #   t=2 sorted [1,3,5]     cuts 2.333, 3.667 -> x=3 is above 1 cut           -> 1
    #   t=3 sorted [1,2,3,5]   cuts 2.0,   3.0   -> x=2 is above 0 (ties go low) -> 0
    #   t=4 sorted [1,2,3,4,5] cuts 2.333, 3.667 -> x=4 is above both            -> 2
    labels = quantile_regimes(_series([5, 1, 3, 2, 4]), (1 / 3, 2 / 3), min_obs=3)
    expected = [np.nan, np.nan, 1.0, 0.0, 2.0]
    np.testing.assert_array_equal(labels.to_numpy(), np.asarray(expected))


def test_expanding_label_equals_full_sample_label_on_truncated_data() -> None:
    """The causal definition: expanding[t] is the full-sample label of the data up to t."""
    x = _synthetic_vix(400, seed=1)
    expanding = vix_terciles(x, mode="expanding", min_obs=50)
    for t in (60, 150, 399):
        truncated = vix_terciles(x.iloc[: t + 1], mode="full_sample", min_obs=50)
        assert expanding.iloc[t] == truncated.iloc[t]


def test_expanding_and_full_sample_differ_on_a_trending_series() -> None:
    # On a strictly rising series every new value is the maximum so far: the causal terciles call
    # every date "high" after warm-up, while full-sample terciles are balanced. This is exactly
    # why the two modes must not be mixed up.
    x = _series(np.arange(1, 301))
    expanding = vix_terciles(x, mode="expanding", min_obs=30).dropna()
    assert set(expanding) == {HIGH}
    full = vix_terciles(x, mode="full_sample", min_obs=30)
    assert full.value_counts().to_dict() == {LOW: 100, MID: 100, HIGH: 100}


def test_warmup_is_nan_in_expanding_and_absent_in_full_sample() -> None:
    x = _synthetic_vix(120, seed=2)
    expanding = vix_terciles(x, mode="expanding", min_obs=40)
    assert expanding.iloc[:39].isna().all()
    assert expanding.iloc[39:].notna().all()
    assert vix_terciles(x, mode="full_sample", min_obs=40).notna().all()


def test_nan_input_gives_nan_label_and_does_not_count_toward_min_obs() -> None:
    values = np.arange(1.0, 61.0)
    values[[10, 11, 30]] = np.nan
    labels = quantile_regimes(_series(values), (1 / 3, 2 / 3), min_obs=20)
    assert labels.iloc[[10, 11, 30]].isna().all()
    # Rows 0-9 give 10 valid values, rows 12-21 the next 10: the 20th valid observation is row 21.
    assert labels.iloc[:21].isna().all()
    assert not np.isnan(labels.iloc[21])


def test_single_cutpoint_gives_a_binary_indicator() -> None:
    labels = quantile_regimes(_series(np.arange(1, 101)), (0.9,), mode="full_sample", min_obs=10)
    assert set(labels) == {0.0, 1.0}
    assert int(labels.sum()) == 10  # the top decile: values 91..100


def test_constant_series_is_all_lowest_regime() -> None:
    labels = quantile_regimes(
        _series(np.full(50, 7.0)), (1 / 3, 2 / 3), mode="full_sample", min_obs=10
    )
    assert set(labels) == {LOW}


def test_output_keeps_index_and_name_and_is_deterministic() -> None:
    x = _synthetic_vix(200, seed=3)
    a = vix_terciles(x, min_obs=30)
    b = vix_terciles(x, min_obs=30)
    assert a.index.equals(x.index)
    assert a.name == x.name
    pd.testing.assert_series_equal(a, b)


def test_function_does_not_mutate_its_input() -> None:
    x = _synthetic_vix(100, seed=4)
    before = x.copy()
    vix_terciles(x, min_obs=20)
    pd.testing.assert_series_equal(x, before)


# --------------------------------------------------------------------------------------
# quantile_regimes / vix_terciles: failure cases
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("probs", [(), (0.0,), (1.0,), (0.7, 0.3), (0.5, 0.5), (-0.1, 0.5)])
def test_bad_probs_rejected(probs: tuple[float, ...]) -> None:
    with pytest.raises(RegimeError):
        quantile_regimes(_series(np.arange(1, 50)), probs, min_obs=10)


def test_unknown_mode_rejected() -> None:
    with pytest.raises(RegimeError, match="mode"):
        quantile_regimes(_series(np.arange(1, 50)), (0.5,), mode="smoothed", min_obs=10)  # type: ignore[arg-type]


def test_min_obs_too_small_rejected() -> None:
    with pytest.raises(RegimeError, match="min_obs"):
        quantile_regimes(_series(np.arange(1, 50)), (0.5,), min_obs=1)


def test_full_sample_requires_min_obs_valid_values() -> None:
    with pytest.raises(RegimeError, match="at least"):
        quantile_regimes(_series([1.0, 2.0, 3.0]), (0.5,), mode="full_sample", min_obs=10)


def test_unsorted_or_duplicate_index_rejected() -> None:
    x = _series(np.arange(1, 50))
    with pytest.raises(RegimeError, match="strictly increasing"):
        quantile_regimes(x.iloc[::-1], (0.5,), min_obs=10)
    dup = x.copy()
    dup.index = dup.index[:-1].append(dup.index[:1])  # last label repeats the first
    with pytest.raises(RegimeError):
        quantile_regimes(dup, (0.5,), min_obs=10)


def test_infinite_values_rejected() -> None:
    values = np.arange(1.0, 50.0)
    values[5] = np.inf
    with pytest.raises(RegimeError, match="inf"):
        quantile_regimes(_series(values), (0.5,), min_obs=10)


def test_non_series_rejected() -> None:
    with pytest.raises(RegimeError, match="Series"):
        quantile_regimes(pd.DataFrame({"a": [1.0, 2.0]}), (0.5,))  # type: ignore[arg-type]


def test_empty_series_rejected_in_full_sample_and_all_nan_in_expanding() -> None:
    empty = pd.Series([], dtype=float, index=pd.DatetimeIndex([], name="date"))
    with pytest.raises(RegimeError):
        quantile_regimes(empty, (0.5,), mode="full_sample", min_obs=5)
    all_nan = quantile_regimes(_series([np.nan] * 10), (0.5,), min_obs=5)
    assert all_nan.isna().all()


def test_vix_must_be_a_positive_level_not_a_return() -> None:
    returns_like = _series(np.r_[np.arange(1, 40), -0.5])
    with pytest.raises(RegimeError, match="strictly positive"):
        vix_terciles(returns_like, min_obs=10)


# --------------------------------------------------------------------------------------
# drawdown / drawdown_states
# --------------------------------------------------------------------------------------


def test_drawdown_known_answer() -> None:
    dd = drawdown(_series([100.0, 110.0, 99.0, 121.0, 100.0]))
    expected = [0.0, 0.0, 99 / 110 - 1, 0.0, 100 / 121 - 1]
    np.testing.assert_allclose(dd.to_numpy(), expected, rtol=0, atol=1e-15)
    assert (dd <= 0).all()


def test_drawdown_nan_does_not_reset_the_running_maximum() -> None:
    dd = drawdown(_series([100.0, 120.0, np.nan, 90.0]))
    assert np.isnan(dd.iloc[2])
    assert dd.iloc[3] == pytest.approx(90 / 120 - 1)


def test_drawdown_states_known_answer() -> None:
    prices = _series([100.0, 110.0, 99.0, 121.0, 100.0])
    # dd: 0, 0, -0.100, 0, -0.1736 -> with thresholds (-0.05, -0.15): 0, 0, 1, 0, 2
    states = drawdown_states(prices, (-0.05, -0.15))
    np.testing.assert_array_equal(states.to_numpy(), [0.0, 0.0, 1.0, 0.0, 2.0])


def test_drawdown_state_boundary_is_inclusive() -> None:
    # 50 / 100 - 1 is exactly -0.5 in floating point.
    states = drawdown_states(_series([100.0, 50.0]), (-0.5,))
    np.testing.assert_array_equal(states.to_numpy(), [0.0, 1.0])


def test_drawdown_states_default_thresholds_are_calm_drawdown_bear() -> None:
    prices = _series([100.0, 98.0, 90.0, 70.0])  # dd 0, -2%, -10%, -30%
    np.testing.assert_array_equal(drawdown_states(prices).to_numpy(), [0.0, 0.0, 1.0, 2.0])


def test_drawdown_states_nan_price_gives_nan_state() -> None:
    states = drawdown_states(_series([100.0, np.nan, 90.0]))
    assert np.isnan(states.iloc[1])
    assert states.iloc[2] == 1.0


@pytest.mark.parametrize("thresholds", [(), (0.05,), (-0.2, -0.05), (-0.1, -0.1), (0.0,)])
def test_bad_drawdown_thresholds_rejected(thresholds: tuple[float, ...]) -> None:
    with pytest.raises(RegimeError):
        drawdown_states(_series([100.0, 90.0]), thresholds)


def test_non_positive_price_rejected() -> None:
    with pytest.raises(RegimeError, match="positive"):
        drawdown(_series([100.0, 0.0, 90.0]))


# --------------------------------------------------------------------------------------
# lag_regimes
# --------------------------------------------------------------------------------------


def test_lag_regimes_is_the_previous_rows_label() -> None:
    labels = _series([0.0, 1.0, 2.0, 1.0])
    lagged = lag_regimes(labels, 1)
    assert np.isnan(lagged.iloc[0])
    np.testing.assert_array_equal(lagged.iloc[1:].to_numpy(), [0.0, 1.0, 2.0])
    pd.testing.assert_series_equal(lag_regimes(labels, 0), labels)


def test_lag_regimes_rejects_a_negative_shift() -> None:
    with pytest.raises(RegimeError, match="look ahead"):
        lag_regimes(_series([0.0, 1.0]), -1)


# --------------------------------------------------------------------------------------
# regime_summary
# --------------------------------------------------------------------------------------


def test_regime_summary_known_answer() -> None:
    labels = _series([0, 0, 1, 1, 1, 0, np.nan, 0, 2, 2])
    summary = regime_summary(labels)
    assert list(summary.columns) == [
        "n_obs",
        "share",
        "n_runs",
        "mean_run_length",
        "max_run_length",
    ]
    assert summary["n_obs"].tolist() == [4, 3, 2]
    assert summary["n_runs"].tolist() == [3, 1, 1]  # the NaN splits the two blocks of zeros
    assert summary["max_run_length"].tolist() == [2, 3, 2]
    assert summary["mean_run_length"].tolist() == pytest.approx([4 / 3, 3.0, 2.0])
    assert summary["share"].sum() == pytest.approx(1.0)
    assert summary["share"].iloc[0] == pytest.approx(4 / 9)


def test_regime_summary_reports_empty_regimes_when_n_regimes_given() -> None:
    summary = regime_summary(_series([0, 0, 1]), n_regimes=3)
    assert summary["n_obs"].tolist() == [2, 1, 0]
    assert summary["n_runs"].tolist() == [1, 1, 0]
    assert np.isnan(summary["mean_run_length"].iloc[2])


def test_regime_summary_failure_cases() -> None:
    with pytest.raises(RegimeError, match="no non-NaN"):
        regime_summary(_series([np.nan, np.nan]))
    with pytest.raises(RegimeError, match="integer codes"):
        regime_summary(_series([0.0, 0.5]))
    with pytest.raises(RegimeError, match="smaller than"):
        regime_summary(_series([0.0, 2.0]), n_regimes=2)


def test_regime_summary_run_counts_match_a_real_tercile_series() -> None:
    x = _synthetic_vix(1500, seed=5)
    labels = vix_terciles(x, mode="full_sample", min_obs=100)
    summary = regime_summary(labels, n_regimes=3)
    assert int(summary["n_obs"].sum()) == 1500
    # A persistent log-AR(1) series (phi=0.97) must form long runs, far fewer than n_obs.
    assert (summary["mean_run_length"] > 5).all()
