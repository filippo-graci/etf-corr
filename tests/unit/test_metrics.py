"""Unit tests for ``etfcorr.evaluation.metrics`` (step 5.6): known answers for every loss."""

import numpy as np
import pandas as pd
import pytest

from etfcorr.evaluation import metrics as M
from etfcorr.features import UNIT

INDEX = pd.bdate_range("2021-01-01", periods=4, name="date")


def long(values: list[list[float]]) -> pd.Series:
    index = pd.MultiIndex.from_product([INDEX[: len(values)], list("abcd")], names=["date", UNIT])
    return pd.Series(np.asarray(values, dtype=float).ravel(), index=index)


def test_squared_and_absolute_error_drop_unknown_rows() -> None:
    y = pd.Series([1.0, 2.0, np.nan, 4.0], index=INDEX)
    f = pd.Series([1.5, np.nan, 3.0, 2.0], index=INDEX)
    assert M.squared_error(y, f).tolist() == [0.25, 4.0]
    assert M.absolute_error(y, f).tolist() == [0.5, 2.0]
    assert M.squared_error(y, f).index.tolist() == [INDEX[0], INDEX[3]]
    with pytest.raises(M.MetricError):
        M.squared_error(y, f.iloc[1:])


def test_qlike_is_zero_at_the_truth_positive_elsewhere_and_asymmetric() -> None:
    v = pd.Series([1e-4, 1e-4, 1e-4], index=INDEX[:3])
    f = pd.Series([1e-4, 2e-4, 0.5e-4], index=INDEX[:3])
    out = M.qlike(v, f)
    assert out.iloc[0] == pytest.approx(0.0)
    assert out.iloc[1] == pytest.approx(0.5 - np.log(0.5) - 1.0)
    assert out.iloc[2] == pytest.approx(2.0 - np.log(2.0) - 1.0)
    assert out.iloc[2] > out.iloc[1] > 0  # under-forecasting variance costs more
    with pytest.raises(M.MetricError):
        M.qlike(v, f * 0.0)


def test_by_date_averages_units() -> None:
    loss = long([[1, 2, 3, 4], [0, 0, 0, 8]])
    assert M.by_date(loss).tolist() == [2.5, 2.0]
    flat = pd.Series([1.0, 2.0], index=INDEX[:2])
    assert M.by_date(flat) is flat


def test_pair_rmse_known_value() -> None:
    true = np.tile(np.eye(3), (2, 1, 1))
    forecast = true.copy()
    forecast[1, 0, 1] = forecast[1, 1, 0] = 0.3  # one of three pairs off by 0.3
    out = M.pair_rmse(true, forecast)
    assert out[0] == 0.0 and out[1] == pytest.approx(0.3 / np.sqrt(3))
    with pytest.raises(M.MetricError):
        M.pair_rmse(true, forecast[:1])


def test_rank_ic_known_values_and_the_constant_forecast_rule() -> None:
    y = long([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    f = long([[10, 20, 30, 40], [4, 3, 2, 1], [5, 5, 5, 5], [1, 2, 4, 3]])
    out = M.rank_ic(y, f)
    assert out.tolist() == pytest.approx([1.0, -1.0, 0.0, 0.8])
    with pytest.raises(M.MetricError):
        M.rank_ic(pd.Series([1.0], index=INDEX[:1]), pd.Series([1.0], index=INDEX[:1]))


def test_hit_rate_ignores_zeros() -> None:
    y = long([[1, -1, 1, 0], [1, 1, -1, -1]])
    f = long([[2, 3, -1, 5], [1, 1, -1, -1]])
    assert M.hit_rate(y, f).tolist() == pytest.approx([1 / 3, 1.0])


def test_r2_oos_known_values() -> None:
    bench = pd.Series([1.0, 1.0, 2.0], index=INDEX[:3])
    assert M.r2_oos(bench * 0.5, bench) == pytest.approx(0.5)
    assert M.r2_oos(bench, bench) == pytest.approx(0.0)
    assert M.r2_oos(bench * 2.0, bench) == pytest.approx(-1.0)
    assert M.r2_oos(bench.iloc[:2], bench) == pytest.approx(0.0)  # common dates only
    assert np.isnan(M.r2_oos(bench, bench * 0.0))


def test_mean_ci_brackets_the_mean_and_is_reproducible() -> None:
    rng = np.random.default_rng(0)
    values = pd.Series(1.0 + rng.standard_normal(400))
    a = M.mean_ci(values, n_bootstrap=300, seed=1)
    b = M.mean_ci(values, n_bootstrap=300, seed=1)
    assert a == b and a.n == 400
    assert a.lower < a.mean < a.upper and a.mean == pytest.approx(values.mean())
    # i.i.d. data, i.i.d. resampling, many draws: the 95% width is 2 x 1.96 standard errors and
    # the 50% width is 2 x 0.674 (this separates alpha / 2 from alpha in the quantiles)
    se = values.std() / np.sqrt(400)
    wide = M.mean_ci(values, mean_block_length=1.0, n_bootstrap=4000, seed=2)
    half = M.mean_ci(values, mean_block_length=1.0, n_bootstrap=4000, seed=2, alpha=0.5)
    assert wide.upper - wide.lower == pytest.approx(2 * 1.96 * se, rel=0.08)
    assert half.upper - half.lower == pytest.approx(2 * 0.674 * se, rel=0.08)


def test_mean_ci_widens_with_the_block_length_on_persistent_data() -> None:
    """An AR(1) series with coefficient 0.95: an i.i.d. bootstrap understates the uncertainty."""
    rng = np.random.default_rng(3)
    x = np.zeros(600)
    for t in range(1, 600):
        x[t] = 0.95 * x[t - 1] + rng.standard_normal()
    values = pd.Series(x)
    iid = M.mean_ci(values, mean_block_length=1.0, n_bootstrap=400)
    blocked = M.mean_ci(values, mean_block_length=40.0, n_bootstrap=400)
    assert (blocked.upper - blocked.lower) > 2.0 * (iid.upper - iid.lower)


def test_mean_ci_on_degenerate_input() -> None:
    assert np.isnan(M.mean_ci(pd.Series([], dtype=float)).mean)
    one = M.mean_ci(pd.Series([2.0]))
    assert one.mean == 2.0 and np.isnan(one.lower) and one.n == 1


def test_block_length_rule() -> None:
    assert M.block_length_for(21, 480) == 42.0
    assert M.block_length_for(1, 1000) == 10.0  # floored at n^(1/3)


def test_project_to_correlation_leaves_psd_alone_and_repairs_the_rest() -> None:
    good = np.array([[1.0, 0.5], [0.5, 1.0]])
    same, changed = M.project_to_correlation(good)
    assert not changed and same is not None and np.array_equal(same, good)
    bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    assert np.linalg.eigvalsh(bad).min() < -0.1
    fixed, changed = M.project_to_correlation(bad)
    assert changed
    assert np.linalg.eigvalsh(fixed).min() > -1e-10
    assert np.allclose(np.diag(fixed), 1.0) and np.allclose(fixed, fixed.T)
    assert np.sign(fixed[0, 2]) == np.sign(bad[0, 2])
    with pytest.raises(M.MetricError):
        M.project_to_correlation(np.full((2, 2), np.nan))


def test_ic_ir_known_value_and_degenerate_cases() -> None:
    ic = pd.Series([0.1, 0.3, np.nan, 0.2])
    assert M.ic_ir(ic) == pytest.approx(0.2 / 0.1)
    assert np.isnan(M.ic_ir(pd.Series([0.1]))) and np.isnan(M.ic_ir(pd.Series([0.2, 0.2])))
