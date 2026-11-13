"""Unit tests for ``etfcorr.correlation.lead_lag_diagnostics`` (step 4.5b)."""

import numpy as np
import pandas as pd
import pytest

from etfcorr.correlation import lead_lag as ll
from etfcorr.correlation import lead_lag_diagnostics as dg


def _indep(n: int, p: int, seed: int) -> pd.DataFrame:
    return ll.simulate_lead_lag(n, p, market_loading=0.0, garch=None, df=None, seed=seed)


def _with_spikes(frame: pd.DataFrame, days: list[int], size: float = 15.0) -> pd.DataFrame:
    out = frame.copy()
    for i, d in enumerate(days):
        out.iloc[d] = out.iloc[d] + size * out.std().to_numpy() * (1.0 if i % 2 == 0 else -1.0)
    return out


# ------------------------------------------------------------------------ simple descriptors


def test_excess_kurtosis_is_near_zero_for_gaussian_and_large_with_spikes() -> None:
    clean = _indep(4000, 3, 1)
    k0 = dg.excess_kurtosis(clean)
    assert (k0.abs() < 0.4).all()
    k1 = dg.excess_kurtosis(_with_spikes(clean, [10, 500, 1000, 2000]))
    assert (k1 > 5.0).all()


def test_excess_kurtosis_rejects_nan() -> None:
    frame = _indep(100, 3, 2)
    frame.iloc[5, 1] = np.nan
    with pytest.raises(ll.LeadLagError, match="NaN"):
        dg.excess_kurtosis(frame)


def test_top_days_finds_the_planted_spike_in_order() -> None:
    frame = _indep(500, 4, 3)
    spiked = _with_spikes(frame, [100, 300], size=20.0)
    out = dg.top_days(spiked, n=2)
    assert set(out["date"]) == {frame.index[100], frame.index[300]}
    assert out["mean_sq"].is_monotonic_decreasing
    assert np.sign(out["ew_return"].iloc[0]) in (-1.0, 1.0)
    with pytest.raises(ll.LeadLagError, match="n must be"):
        dg.top_days(frame, n=0)


# ------------------------------------------------------------------------------ winsorising


def test_winsorise_clips_at_k_keeps_index_and_does_not_modify_input() -> None:
    frame = _with_spikes(_indep(800, 3, 4), [50, 400], size=25.0)
    before = frame.copy()
    out = dg.winsorise_standardised(frame, k=4.0)
    assert out.abs().to_numpy().max() == pytest.approx(4.0)
    assert out.index.equals(frame.index) and list(out.columns) == list(frame.columns)
    pd.testing.assert_frame_equal(frame, before)
    unclipped = dg.winsorise_standardised(_indep(800, 3, 4), k=50.0)
    np.testing.assert_allclose(unclipped.to_numpy(), ll.standardise(_indep(800, 3, 4)))
    with pytest.raises(ll.LeadLagError, match="k must be"):
        dg.winsorise_standardised(frame, k=0.0)


def test_clustered_extreme_days_inflate_the_raw_se_and_winsorising_removes_it() -> None:
    """Consecutive alternating spikes (like a crash fortnight) dominate the raw lagged products."""
    spiked = _with_spikes(_indep(1200, 6, 5), [100, 101, 102, 103], size=15.0)
    raw = ll.lead_lag_scan(spiked, statistic="raw", n_bootstrap=300, seed=1)
    wins = ll.lead_lag_scan(
        dg.winsorise_standardised(spiked, 4.0), statistic="raw", n_bootstrap=300, seed=1
    )
    assert raw.table["se"].median() > 2.0 * wins.table["se"].median()
    assert (wins.table["p_value"] < 0.05).sum() > (raw.table["p_value"] < 0.05).sum()


# ------------------------------------------------------------------------------- shuffling


def test_shuffle_rows_permutes_whole_rows_deterministically() -> None:
    frame = _indep(300, 4, 6)
    a, b = dg.shuffle_rows(frame, 7), dg.shuffle_rows(frame, 7)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a.to_numpy(), frame.to_numpy())
    rows = {tuple(np.round(r, 12)) for r in frame.to_numpy()}
    assert {tuple(np.round(r, 12)) for r in a.to_numpy()} == rows  # the cross-section is kept
    assert a.index.equals(frame.index)
    assert not np.allclose(dg.shuffle_rows(frame, 8).to_numpy(), a.to_numpy())


def test_shuffle_rows_destroys_planted_lead_lag() -> None:
    frame = ll.simulate_lead_lag(
        3000, 4, market_loading=0.0, garch=None, df=None, seed=9, planted=[(0, 1, 1, 0.3)]
    )
    real = ll.lead_lag_scan(frame, lags=(1,), n_bootstrap=200, seed=1)
    shuf = ll.lead_lag_scan(dg.shuffle_rows(frame, 1), lags=(1,), n_bootstrap=200, seed=1)
    assert real.table["reject_bh"].any()
    assert not shuf.table["reject_bh"].any()


# ----------------------------------------------------------------------------- top_day_share


def test_top_day_share_matches_an_explicit_loop() -> None:
    rng = np.random.default_rng(10)
    z = rng.standard_normal((200, 5)) ** 3
    got = dg.top_day_share(z, k=4)
    for j in range(5):
        sq = np.sort(z[:, j] ** 2)[::-1]
        assert got[j] == pytest.approx(sq[:4].sum() / sq.sum())


def test_top_day_share_is_one_for_a_single_spike_and_small_for_iid() -> None:
    z = np.zeros((100, 2))
    z[17, 0] = 5.0
    z[:, 1] = np.tile([1.0, -1.0], 50)
    got = dg.top_day_share(z, k=1)
    assert got[0] == pytest.approx(1.0)
    assert got[1] == pytest.approx(1.0 / 100.0)
    big = np.random.default_rng(11).standard_normal((5000, 3))
    assert (dg.top_day_share(big, k=5) < 0.02).all()
    assert dg.top_day_share(np.zeros((50, 2)), k=3).tolist() == [0.0, 0.0]


def test_top_day_share_validation() -> None:
    with pytest.raises(ll.LeadLagError):
        dg.top_day_share(np.zeros(10), k=1)
    with pytest.raises(ll.LeadLagError):
        dg.top_day_share(np.zeros((10, 2)), k=10)
    with pytest.raises(ll.LeadLagError):
        dg.top_day_share(np.zeros((10, 2)), k=0)


def test_spikes_raise_the_top_day_share_of_the_product_series() -> None:
    clean = _indep(1500, 6, 12)
    spiked = _with_spikes(clean, [100, 500, 900, 1300], size=15.0)
    share = {}
    for name, f in (("clean", clean), ("spiked", spiked)):
        z, _ = ll.lagged_products(ll.standardise(f), (1, 2), "raw")
        share[name] = float(np.median(dg.top_day_share(z, 5)))
    assert share["spiked"] > 3.0 * share["clean"]
    assert share["spiked"] > 0.3


# ------------------------------------------------------------------- factor autocorrelation


def test_factor_autocorrelation_detects_a_persistent_factor_and_not_an_iid_one() -> None:
    ar = ll.simulate_lead_lag(
        3000, 8, market_loading=0.9, factor_ar=0.3, garch=None, df=None, seed=13
    )
    out = dg.factor_autocorrelation(ar, lags=(1, 2), n_bootstrap=300, seed=1)
    assert list(out.columns) == ["lag", "autocorr", "se", "z", "p_value"]
    assert out["autocorr"].iloc[0] == pytest.approx(0.3, abs=0.07)
    assert out["z"].iloc[0] > 4.0
    iid = dg.factor_autocorrelation(_indep(3000, 8, 14), lags=(1, 2), n_bootstrap=300, seed=1)
    assert iid["z"].abs().max() < 3.5
    assert (out["se"] > 0).all()


def test_factor_autocorrelation_known_answer_and_se_scale() -> None:
    frame = _indep(2000, 5, 18)
    out = dg.factor_autocorrelation(frame, lags=(1, 3), n_bootstrap=400, seed=2)
    f = ll.standardise(frame).mean(axis=1)
    f = (f - f.mean()) / f.std(ddof=1)
    kmax = 3
    for row, k in zip(out.itertuples(), (1, 3), strict=True):
        assert row.autocorr == pytest.approx(float(np.mean(f[kmax - k : len(f) - k] * f[kmax:])))
        assert row.z == pytest.approx(row.autocorr / row.se)
        assert row.se == pytest.approx(1.0 / np.sqrt(len(f) - kmax), rel=0.2)  # iid Gaussian


def test_factor_autocorrelation_validation() -> None:
    with pytest.raises(ll.LeadLagError):
        dg.factor_autocorrelation(_indep(100, 3, 15), lags=(0,))


# ----------------------------------------------------------------------- null distribution


def test_empirical_position_known_values() -> None:
    null = np.array([1.0, 2.0, 3.0, 4.0])
    assert dg.empirical_position(null, 0.0) == pytest.approx((1 / 5, 5 / 5))
    assert dg.empirical_position(null, 10.0) == pytest.approx((5 / 5, 1 / 5))
    assert dg.empirical_position(null, 2.0) == pytest.approx((3 / 5, 4 / 5))
    with pytest.raises(ll.LeadLagError):
        dg.empirical_position(np.array([]), 1.0)


def test_shuffle_null_is_deterministic_and_centred_on_alpha_times_tests() -> None:
    frame = _indep(600, 6, 16)
    a = dg.shuffle_null(frame, lags=(1, 2), n_shuffles=12, n_bootstrap=200, seed=3)
    b = dg.shuffle_null(frame, lags=(1, 2), n_shuffles=12, n_bootstrap=200, seed=3)
    np.testing.assert_array_equal(a.counts, b.counts)
    assert a.n_tests == 30 and len(a.counts) == len(a.median_se) == 12
    assert abs(a.counts.mean() - 0.05 * 30) < 1.5
    q = a.quantiles()
    assert q[0.05] <= q[0.5] <= q[0.95]
    raw = dg.shuffle_null(frame, statistic="raw", lags=(1,), n_shuffles=4, n_bootstrap=200)
    assert raw.n_tests == 30


def test_shuffle_null_rejects_too_few_shuffles() -> None:
    with pytest.raises(ll.LeadLagError, match="n_shuffles"):
        dg.shuffle_null(_indep(100, 3, 17), n_shuffles=1)


# ------------------------------------------------------------------------------ the verdict


def test_classify_conservatism_branches() -> None:
    null = np.arange(50.0, 101.0)  # 5th percentile 52.5, median 75, 95th percentile 97.5
    kw = {"se_ratio": 1.0, "winsorised_count": 0.0}
    assert dg.classify_conservatism(75, null, **kw).startswith("H_dep")
    assert dg.classify_conservatism(52.5, null, **kw).startswith("H_dep")  # closed interval
    assert dg.classify_conservatism(97.5, null, **kw).startswith("H_dep")
    assert dg.classify_conservatism(120, null, **kw).startswith("above null")
    assert dg.classify_conservatism(10, null, **kw).startswith("unexplained")
    tail = dg.classify_conservatism(10, null, se_ratio=1.5, winsorised_count=60)
    assert tail.startswith("H_tail")
    far = dg.classify_conservatism(10, null, se_ratio=1.5, winsorised_count=5)
    assert far.startswith("unexplained")  # winsorising did not move the count to the null
    low_ratio = dg.classify_conservatism(10, null, se_ratio=1.2, winsorised_count=60)
    assert low_ratio.startswith("unexplained")  # 1.2 is not "more than 20%"
    with pytest.raises(ll.LeadLagError):
        dg.classify_conservatism(1, np.array([1.0]), se_ratio=1.0, winsorised_count=1)
