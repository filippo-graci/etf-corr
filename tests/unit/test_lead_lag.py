"""Unit tests for ``etfcorr.correlation.lead_lag`` (step 4.5)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from etfcorr.correlation import lead_lag as ll
from etfcorr.correlation.nonlinear import benjamini_hochberg_qvalues

BOOT = 300  # bootstrap draws in tests (the production default is 2000)


def _frame(u: np.ndarray, start: str = "2016-01-04") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(u), name="date")
    return pd.DataFrame(u, index=idx, columns=[f"X{i}" for i in range(u.shape[1])])


def _indep(n: int, p: int, seed: int) -> pd.DataFrame:
    """Gaussian independent assets, no market factor, no clustering: the textbook null."""
    return ll.simulate_lead_lag(n, p, market_loading=0.0, garch=None, df=None, seed=seed)


# ---------------------------------------------------------------------------------- windows


def test_slice_window_is_inclusive_on_both_ends() -> None:
    frame = _frame(np.arange(20.0).reshape(10, 2))
    out = ll.slice_window(frame, str(frame.index[2].date()), str(frame.index[5].date()))
    assert list(out.index) == list(frame.index[2:6])


def test_slice_window_rejects_reversed_empty_and_bad_frames() -> None:
    frame = _frame(np.arange(20.0).reshape(10, 2))
    with pytest.raises(ll.LeadLagError, match="after end"):
        ll.slice_window(frame, "2016-02-01", "2016-01-01")
    with pytest.raises(ll.LeadLagError, match="no rows"):
        ll.slice_window(frame, "2030-01-01", "2030-02-01")
    with pytest.raises(ll.LeadLagError, match="strictly increasing"):
        ll.slice_window(frame.iloc[::-1], "2016-01-01", "2016-12-31")


# ----------------------------------------------------------------------------- standardise


def test_standardise_gives_mean_zero_unit_sd() -> None:
    u = ll.standardise(_indep(300, 4, 1))
    np.testing.assert_allclose(u.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(u.std(axis=0, ddof=1), 1.0, atol=1e-12)


def test_standardise_rank_equals_standardised_ranks() -> None:
    frame = _indep(200, 3, 2)
    ranks = frame.rank().to_numpy()
    expected = (ranks - ranks.mean(axis=0)) / ranks.std(axis=0, ddof=1)
    np.testing.assert_allclose(ll.standardise(frame, "rank"), expected, atol=1e-12)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_standardise_rejects_non_finite(bad: float) -> None:
    frame = _indep(50, 3, 3)
    frame.iloc[10, 1] = bad
    with pytest.raises(ll.LeadLagError, match="NaN or infinite"):
        ll.standardise(frame)


def test_standardise_rejects_constant_column_bad_transform_and_one_column() -> None:
    frame = _indep(50, 3, 3)
    frame["A01"] = 0.01
    with pytest.raises(ll.LeadLagError, match="constant"):
        ll.standardise(frame)
    with pytest.raises(ll.LeadLagError, match="transform"):
        ll.standardise(_indep(50, 3, 3), "zscore")  # type: ignore[arg-type]
    with pytest.raises(ll.LeadLagError, match="at least 2 columns"):
        ll.standardise(_indep(50, 3, 3).iloc[:, :1])


# ------------------------------------------------------------------------------ products


def _loop_estimate(u: np.ndarray, i: int, j: int, k: int, kmax: int) -> float:
    """mean over t = kmax..n-1 of u[t, j] * u[t-k, i], written as a plain loop."""
    return float(np.mean([u[t, j] * u[t - k, i] for t in range(kmax, len(u))]))


def test_lagged_products_match_explicit_loops() -> None:
    rng = np.random.default_rng(0)
    u = rng.standard_normal((60, 4))
    lags = (1, 3)
    for statistic in ("raw", "asymmetry"):
        z, labels = ll.lagged_products(u, lags, statistic)  # type: ignore[arg-type]
        assert z.shape == (60 - 3, len(labels))
        for col, (i, j, k) in enumerate(labels):
            expected = _loop_estimate(u, i, j, k, 3)
            if statistic == "asymmetry":
                assert i < j
                expected -= _loop_estimate(u, j, i, k, 3)
            assert z[:, col].mean() == pytest.approx(expected, abs=1e-12)
    # family sizes: 4 assets, 2 lags
    assert ll.lagged_products(u, lags, "raw")[0].shape[1] == 4 * 3 * 2
    assert ll.lagged_products(u, lags, "asymmetry")[0].shape[1] == 6 * 2


def test_lagged_products_validation() -> None:
    u = np.random.default_rng(0).standard_normal((100, 3))
    with pytest.raises(ll.LeadLagError, match="statistic"):
        ll.lagged_products(u, (1,), "bogus")  # type: ignore[arg-type]
    with pytest.raises(ll.LeadLagError, match="lags"):
        ll.lagged_products(u, (), "raw")
    with pytest.raises(ll.LeadLagError, match=">= 1"):
        ll.lagged_products(u, (0,), "raw")
    with pytest.raises(ll.LeadLagError, match="distinct"):
        ll.lagged_products(u, (1, 1), "raw")
    with pytest.raises(ll.LeadLagError, match="usable rows"):
        ll.lagged_products(u[:40], (20,), "raw")
    with pytest.raises(ll.LeadLagError, match="2-D"):
        ll.lagged_products(u[:, 0], (1,), "raw")


# ---------------------------------------------------------------- symmetry and direction


def test_swapping_two_assets_negates_the_asymmetry() -> None:
    frame = ll.simulate_lead_lag(500, 2, planted=[(0, 1, 1, 0.3)], seed=4)
    a = ll.lead_lag_scan(frame, lags=(1, 2), n_bootstrap=BOOT)
    b = ll.lead_lag_scan(frame[["A01", "A00"]], lags=(1, 2), n_bootstrap=BOOT)
    np.testing.assert_allclose(a.table["estimate"], -b.table["estimate"], atol=1e-12)
    np.testing.assert_allclose(a.table["se"], b.table["se"], rtol=1e-12)
    assert list(a.table["leader"]) == list(b.table["leader"])


def test_time_reversal_negates_the_asymmetry_for_a_single_lag() -> None:
    frame = ll.simulate_lead_lag(400, 3, planted=[(0, 1, 2, 0.3)], seed=5)
    reversed_frame = pd.DataFrame(frame.to_numpy()[::-1], index=frame.index, columns=frame.columns)
    fwd = ll.lead_lag_scan(frame, lags=(2,), n_bootstrap=BOOT)
    rev = ll.lead_lag_scan(reversed_frame, lags=(2,), n_bootstrap=BOOT)
    np.testing.assert_allclose(fwd.table["estimate"], -rev.table["estimate"], atol=1e-12)


def test_planted_lead_lag_known_answer_and_orientation() -> None:
    """r1_t = x1_t + 0.3 r0_(t-1): D = 0.3 / sqrt(1.09) = 0.2873 at lag 1, zero elsewhere."""
    frame = ll.simulate_lead_lag(40000, 4, market_loading=0.0, garch=None, df=None, seed=6,
                                 planted=[(0, 1, 1, 0.3)])  # fmt: skip
    scan = ll.lead_lag_scan(frame, lags=(1, 2, 3), n_bootstrap=BOOT)
    t = scan.table
    cell = t[(t["asset_a"] == "A00") & (t["asset_b"] == "A01") & (t["lag"] == 1)].iloc[0]
    assert cell["estimate"] == pytest.approx(0.3 / np.sqrt(1.09), abs=0.025)
    assert cell["leader"] == "A00"
    assert cell["follower"] == "A01"
    assert bool(cell["reject_bh"]) and bool(cell["reject_by"])
    others = t.drop(cell.name)
    assert others["z"].abs().max() < 4.5
    assert not others["reject_bh"].any()


def test_oriented_columns_follow_the_sign_of_the_estimate() -> None:
    frame = ll.simulate_lead_lag(3000, 3, seed=7, planted=[(1, 0, 1, 0.3)])
    t = ll.lead_lag_scan(frame, lags=(1,), n_bootstrap=BOOT).table
    for _, row in t.iterrows():
        if row["estimate"] >= 0:
            assert (row["leader"], row["follower"]) == (row["asset_a"], row["asset_b"])
        else:
            assert (row["leader"], row["follower"]) == (row["asset_b"], row["asset_a"])
    cell = t[(t["asset_a"] == "A00") & (t["asset_b"] == "A01")].iloc[0]
    assert cell["estimate"] < 0
    assert cell["leader"] == "A01"


def test_raw_statistic_leader_is_always_the_lagged_asset() -> None:
    frame = ll.simulate_lead_lag(500, 3, seed=8)
    t = ll.lead_lag_scan(frame, lags=(1,), statistic="raw", n_bootstrap=BOOT).table
    assert len(t) == 3 * 2
    assert (t["leader"] == t["asset_a"]).all()
    assert (t["follower"] == t["asset_b"]).all()


# --------------------------------------------------- the Lo-MacKinlay point: raw vs asymmetry


def test_common_factor_autocorrelation_fools_raw_but_not_asymmetry() -> None:
    """Factor AR(0.3): every pair gets a lag-1 cross-correlation of about 0.49 * 0.3, both ways."""
    frame = ll.simulate_lead_lag(3000, 8, market_loading=0.7, factor_ar=0.3, garch=None, df=None,
                                 seed=9)  # fmt: skip
    raw = ll.lead_lag_scan(frame, lags=(1,), statistic="raw", n_bootstrap=BOOT)
    asym = ll.lead_lag_scan(frame, lags=(1,), statistic="asymmetry", n_bootstrap=BOOT)
    assert raw.table["reject_bh"].mean() > 0.9
    assert raw.table["estimate"].mean() == pytest.approx(0.49 * 0.3, abs=0.03)
    assert asym.table["reject_bh"].sum() == 0
    assert asym.table["estimate"].abs().max() < 0.12


# ---------------------------------------------------------------------- inference quality


def test_bootstrap_se_matches_the_iid_gaussian_formula() -> None:
    n, k = 4000, 1
    frame = _indep(n, 4, 10)
    n_eff = n - k
    raw = ll.lead_lag_scan(frame, lags=(k,), statistic="raw", n_bootstrap=1000, seed=1)
    asym = ll.lead_lag_scan(frame, lags=(k,), statistic="asymmetry", n_bootstrap=1000, seed=1)
    assert raw.table["se"].mean() == pytest.approx(1 / np.sqrt(n_eff), rel=0.08)
    assert asym.table["se"].mean() == pytest.approx(np.sqrt(2) / np.sqrt(n_eff), rel=0.08)


def test_null_p_values_have_roughly_nominal_size_with_garch_t_returns() -> None:
    sizes = []
    any_rejection = 0
    for seed in range(12):
        frame = ll.simulate_lead_lag(1000, 12, seed=seed)
        scan = ll.lead_lag_scan(frame, lags=(1, 2, 3), n_bootstrap=BOOT, seed=seed)
        sizes.append(float((scan.table["p_value"] < 0.05).mean()))
        any_rejection += int(scan.table["reject_bh"].any())
    assert np.mean(sizes) == pytest.approx(0.05, abs=0.012)
    assert any_rejection <= 2  # under a global null FDR control means P(any rejection) <= alpha


def test_fdr_q_values_match_the_reference_implementations() -> None:
    frame = ll.simulate_lead_lag(800, 8, seed=11, planted=[(0, 1, 1, 0.25), (2, 3, 2, 0.25)])
    t = ll.lead_lag_scan(frame, lags=(1, 2, 3), n_bootstrap=BOOT).table
    p = t["p_value"].to_numpy()
    np.testing.assert_allclose(t["q_bh"], benjamini_hochberg_qvalues(p), atol=1e-12)
    m = len(p)
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    np.testing.assert_allclose(t["q_by"], np.minimum(1.0, c_m * t["q_bh"]), atol=1e-12)
    assert (t["q_by"] >= t["q_bh"] - 1e-15).all()
    assert (t["reject_bh"] == (t["q_bh"] <= 0.05)).all()


def test_p_values_are_two_sided_normal_of_z() -> None:
    t = ll.lead_lag_scan(ll.simulate_lead_lag(500, 5, seed=12), n_bootstrap=BOOT).table
    np.testing.assert_allclose(t["p_value"], 2 * stats.norm.sf(np.abs(t["z"])), atol=1e-12)
    np.testing.assert_allclose(t["z"], t["estimate"] / t["se"], atol=1e-12)


def test_minimum_detectable_formula_and_top() -> None:
    scan = ll.lead_lag_scan(ll.simulate_lead_lag(500, 5, seed=13), n_bootstrap=BOOT)
    expected = stats.norm.isf(0.05 / (2 * scan.n_tests)) * float(scan.table["se"].median())
    assert scan.minimum_detectable == pytest.approx(expected, rel=1e-12)
    top = scan.top(5)
    assert len(top) == 5
    assert top["p_value"].is_monotonic_increasing
    assert top["p_value"].iloc[0] == scan.table["p_value"].min()


def test_rank_transform_limits_the_influence_of_one_extreme_day() -> None:
    frame = _indep(1000, 3, 14)
    values = frame.to_numpy().copy()
    values[500, 0] = 0.5 * 100  # one enormous return on A00 ...
    values[501, 1] = 0.5 * 100  # ... followed by one enormous return on A01
    spiked = pd.DataFrame(values, index=frame.index, columns=frame.columns)
    plain = ll.lead_lag_scan(spiked, lags=(1,), n_bootstrap=BOOT)
    ranked = ll.lead_lag_scan(spiked, lags=(1,), transform="rank", n_bootstrap=BOOT)

    def cell(scan: ll.LeadLagScan) -> pd.Series:
        t = scan.table
        return t[(t["asset_a"] == "A00") & (t["asset_b"] == "A01")].iloc[0]

    assert abs(cell(plain)["estimate"]) > 0.5
    assert abs(cell(ranked)["estimate"]) < 0.15
    assert abs(cell(ranked)["z"]) < 4.0


# ---------------------------------------------------------------------- determinism, input


def test_scan_is_deterministic_and_seed_changes_only_the_se() -> None:
    frame = ll.simulate_lead_lag(400, 5, seed=15)
    a = ll.lead_lag_scan(frame, n_bootstrap=BOOT, seed=3)
    b = ll.lead_lag_scan(frame, n_bootstrap=BOOT, seed=3)
    c = ll.lead_lag_scan(frame, n_bootstrap=BOOT, seed=4)
    pd.testing.assert_frame_equal(a.table, b.table)
    np.testing.assert_array_equal(a.table["estimate"], c.table["estimate"])
    assert not np.allclose(a.table["se"], c.table["se"])


def test_scan_does_not_modify_its_input() -> None:
    frame = ll.simulate_lead_lag(300, 4, seed=16)
    before = frame.copy()
    ll.lead_lag_scan(frame, n_bootstrap=BOOT, keep_null=True)
    pd.testing.assert_frame_equal(frame, before)


def test_scan_validation_errors() -> None:
    frame = ll.simulate_lead_lag(300, 4, seed=17)
    with pytest.raises(ll.LeadLagError, match="alpha"):
        ll.lead_lag_scan(frame, alpha=1.5, n_bootstrap=BOOT)
    with pytest.raises(ll.LeadLagError, match="n_bootstrap"):
        ll.lead_lag_scan(frame, n_bootstrap=1)
    with pytest.raises(ll.LeadLagError, match="mean_block_length"):
        ll.lead_lag_scan(frame, mean_block_length=0.0, n_bootstrap=BOOT)
    with pytest.raises(ll.LeadLagError, match="DataFrame"):
        ll.lead_lag_scan(frame.to_numpy(), n_bootstrap=BOOT)  # type: ignore[arg-type]
    dup = pd.concat([frame, frame.iloc[:, :1]], axis=1)
    with pytest.raises(ll.LeadLagError, match="duplicate column"):
        ll.lead_lag_scan(dup, n_bootstrap=BOOT)
    with pytest.raises(ll.LeadLagError, match="usable rows"):
        ll.lead_lag_scan(frame.iloc[:20], n_bootstrap=BOOT)


def test_sectors_add_a_cross_sector_flag_and_must_cover_every_asset() -> None:
    frame = ll.simulate_lead_lag(300, 4, seed=18)
    sectors = {"A00": "x", "A01": "x", "A02": "y", "A03": "y"}
    t = ll.lead_lag_scan(frame, lags=(1,), n_bootstrap=BOOT, sectors=sectors).table
    for _, row in t.iterrows():
        assert row["cross_sector"] == (sectors[row["asset_a"]] != sectors[row["asset_b"]])
    assert int(t["cross_sector"].sum()) == 4
    with pytest.raises(ll.LeadLagError, match="sectors has no entry"):
        ll.lead_lag_scan(frame, n_bootstrap=BOOT, sectors={"A00": "x"})


def test_bootstrap_means_average_to_the_sample_mean_and_share_draws_across_columns() -> None:
    rng = np.random.default_rng(0)
    z = rng.standard_normal((200, 3))
    z[:, 2] = z[:, 0]  # identical columns must stay identical in every replicate
    means = ll.bootstrap_means(z, 8.0, 400, 1)
    assert means.shape == (400, 3)
    np.testing.assert_array_equal(means[:, 0], means[:, 2])
    assert means[:, 1].mean() == pytest.approx(z[:, 1].mean(), abs=0.02)


# --------------------------------------------------------------- replication and agreement


def _two_windows(planted: list[tuple[int, int, int, float]], seeds: tuple[int, int]) -> tuple:
    disc = ll.simulate_lead_lag(6000, 4, market_loading=0.0, garch=None, df=None, seed=seeds[0],
                                planted=planted)  # fmt: skip
    val = ll.simulate_lead_lag(3000, 4, market_loading=0.0, garch=None, df=None, seed=seeds[1],
                               planted=planted)  # fmt: skip
    return disc, val


def test_replication_table_replicates_a_persistent_effect_and_not_a_vanished_one() -> None:
    planted = [(0, 1, 1, 0.3)]
    disc, val = _two_windows(planted, (20, 21))
    _, val_null = _two_windows([], (20, 22))
    d = ll.lead_lag_scan(disc, lags=(1, 2), n_bootstrap=BOOT, seed=1)
    v = ll.lead_lag_scan(val, lags=(1, 2), n_bootstrap=BOOT, seed=2)
    v0 = ll.lead_lag_scan(val_null, lags=(1, 2), n_bootstrap=BOOT, seed=2)
    rep = ll.replication_table(d, v)
    assert len(rep) == int(d.table["reject_bh"].sum()) >= 1
    row = rep[(rep["asset_a"] == "A00") & (rep["asset_b"] == "A01") & (rep["lag"] == 1)].iloc[0]
    assert bool(row["replicated"])
    assert row["estimate_val"] == pytest.approx(0.3 / np.sqrt(1.09), abs=0.1)  # about 4 se
    gone = ll.replication_table(d, v0)
    assert not gone["replicated"].any()
    assert (gone["p_one_sided"] > 0.01).all()


def test_replication_uses_the_discovery_sign_one_sided() -> None:
    disc, _ = _two_windows([(0, 1, 1, 0.3)], (23, 24))
    _, val_reverse = _two_windows([(1, 0, 1, 0.3)], (23, 25))  # opposite direction in validation
    d = ll.lead_lag_scan(disc, lags=(1,), n_bootstrap=BOOT, seed=1)
    v = ll.lead_lag_scan(val_reverse, lags=(1,), n_bootstrap=BOOT, seed=2)
    rep = ll.replication_table(d, v)
    row = rep[(rep["asset_a"] == "A00") & (rep["asset_b"] == "A01")].iloc[0]
    assert row["estimate_disc"] > 0 > row["estimate_val"]
    assert row["p_one_sided"] > 0.99
    assert not bool(row["replicated"])


def test_replication_of_a_negative_discovery_uses_its_own_sign() -> None:
    """A1 leads A0 gives a negative asymmetry for the (A00, A01) cell; it must replicate."""
    planted = [(1, 0, 1, 0.3)]
    disc, val = _two_windows(planted, (40, 41))
    d = ll.lead_lag_scan(disc, lags=(1,), n_bootstrap=BOOT, seed=1)
    v = ll.lead_lag_scan(val, lags=(1,), n_bootstrap=BOOT, seed=2)
    rep = ll.replication_table(d, v)
    row = rep[(rep["asset_a"] == "A00") & (rep["asset_b"] == "A01")].iloc[0]
    assert row["estimate_disc"] < 0 and row["estimate_val"] < 0
    assert row["p_one_sided"] < 0.01
    assert bool(row["replicated"])


def test_replication_table_is_empty_when_nothing_is_discovered() -> None:
    a, b = _two_windows([], (26, 27))
    d = ll.lead_lag_scan(a, lags=(1,), n_bootstrap=BOOT, seed=1)
    v = ll.lead_lag_scan(b, lags=(1,), n_bootstrap=BOOT, seed=2)
    rep = ll.replication_table(d, v)
    assert len(rep) == 0
    assert "replicated" in rep.columns


def test_replication_and_agreement_reject_mismatched_families() -> None:
    frame = ll.simulate_lead_lag(300, 4, seed=28)
    a = ll.lead_lag_scan(frame, lags=(1, 2), n_bootstrap=BOOT, seed=1, keep_null=True)
    b = ll.lead_lag_scan(frame, lags=(1,), n_bootstrap=BOOT, seed=2, keep_null=True)
    c = ll.lead_lag_scan(frame, lags=(1, 2), statistic="raw", n_bootstrap=BOOT, seed=2)
    d = ll.lead_lag_scan(frame.iloc[:, :3], lags=(1, 2), n_bootstrap=BOOT, seed=2)
    for other in (b, c, d):
        with pytest.raises(ll.LeadLagError, match="same family"):
            ll.replication_table(a, other)
        with pytest.raises(ll.LeadLagError, match="same family"):
            ll.agreement_test(a, other)


def test_agreement_needs_null_replicates_and_distinct_seeds() -> None:
    frame = ll.simulate_lead_lag(300, 4, seed=29)
    with_null = ll.lead_lag_scan(frame, n_bootstrap=BOOT, seed=1, keep_null=True)
    without = ll.lead_lag_scan(frame, n_bootstrap=BOOT, seed=2)
    same_seed = ll.lead_lag_scan(frame, n_bootstrap=BOOT, seed=1, keep_null=True)
    with pytest.raises(ll.LeadLagError, match="keep_null"):
        ll.agreement_test(with_null, without)
    with pytest.raises(ll.LeadLagError, match="different seeds"):
        ll.agreement_test(with_null, same_seed)


def test_agreement_detects_many_weak_persistent_effects_that_fdr_misses() -> None:
    planted = [(0, 1, 1, 0.12), (2, 3, 1, 0.12), (4, 5, 1, 0.12), (6, 7, 1, 0.12),
               (1, 4, 2, 0.12), (3, 6, 2, 0.12)]  # fmt: skip

    def window(seed: int) -> pd.DataFrame:
        return ll.simulate_lead_lag(1500, 10, market_loading=0.0, garch=None, df=None, seed=seed,
                                    planted=planted)  # fmt: skip

    d = ll.lead_lag_scan(window(30), lags=(1, 2, 3), n_bootstrap=BOOT, seed=1, keep_null=True)
    v = ll.lead_lag_scan(window(31), lags=(1, 2, 3), n_bootstrap=BOOT, seed=2, keep_null=True)
    out = ll.agreement_test(d, v)
    assert out.n_tests == d.n_tests == 45 * 3
    assert out.n_null == BOOT
    assert out.statistic > 0.15
    assert out.p_value < 0.02


def test_agreement_p_value_is_never_zero_and_has_the_plus_one_floor() -> None:
    planted = [(i, j, 1, 0.3) for i in range(0, 5) for j in range(5, 10)]

    def window(seed: int) -> pd.DataFrame:
        return ll.simulate_lead_lag(2000, 10, market_loading=0.0, garch=None, df=None, seed=seed,
                                    planted=planted)  # fmt: skip

    d = ll.lead_lag_scan(window(50), lags=(1,), n_bootstrap=BOOT, seed=1, keep_null=True)
    v = ll.lead_lag_scan(window(51), lags=(1,), n_bootstrap=BOOT, seed=2, keep_null=True)
    out = ll.agreement_test(d, v)
    assert out.p_value == pytest.approx(1.0 / (1 + BOOT))  # no null draw reaches the statistic


def test_default_block_length_is_the_data_driven_one_and_exceeds_one() -> None:
    from etfcorr.correlation.horizon_analysis import default_mean_block_length

    scan = ll.lead_lag_scan(_indep(1000, 4, 52), lags=(1, 2), n_bootstrap=BOOT, seed=1)
    assert scan.mean_block_length == pytest.approx(default_mean_block_length(scan.n_effective))
    assert scan.mean_block_length > 1.0
    fixed = ll.lead_lag_scan(_indep(1000, 4, 52), n_bootstrap=BOOT, mean_block_length=7.0)
    assert fixed.mean_block_length == 7.0


def test_agreement_is_not_significant_under_the_null() -> None:
    pvals = []
    for seed in range(10):
        a = ll.simulate_lead_lag(1000, 10, seed=100 + seed)
        b = ll.simulate_lead_lag(500, 10, seed=200 + seed)
        d = ll.lead_lag_scan(a, lags=(1, 2), n_bootstrap=BOOT, seed=1, keep_null=True)
        v = ll.lead_lag_scan(b, lags=(1, 2), n_bootstrap=BOOT, seed=2, keep_null=True)
        pvals.append(ll.agreement_test(d, v).p_value)
    assert sum(p < 0.05 for p in pvals) <= 2
    assert 0.25 < float(np.mean(pvals)) < 0.75


# --------------------------------------------------------------------------- reporting, sim


def test_summary_lines_carry_the_caveats_and_counts() -> None:
    scan = ll.lead_lag_scan(ll.simulate_lead_lag(300, 4, seed=31), n_bootstrap=BOOT)
    text = "\n".join(ll.scan_summary_lines(scan))
    assert "tests=30" in text
    assert "BH rejects" in text and "BY rejects" in text
    assert "exploratory, in-sample, descriptive" in text
    assert "not a causal one" in text


def test_simulator_is_deterministic_has_the_right_shape_and_unit_scale() -> None:
    a = ll.simulate_lead_lag(500, 6, seed=1)
    b = ll.simulate_lead_lag(500, 6, seed=1)
    c = ll.simulate_lead_lag(500, 6, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)
    assert a.shape == (500, 6)
    assert list(a.columns) == ["A00", "A01", "A02", "A03", "A04", "A05"]
    big = ll.simulate_lead_lag(20000, 4, seed=3)
    assert float(big.to_numpy().std()) == pytest.approx(0.01, rel=0.1)


def test_simulator_planted_effect_shows_up_in_the_cross_correlation() -> None:
    frame = ll.simulate_lead_lag(20000, 3, market_loading=0.0, garch=None, df=None, seed=4,
                                 planted=[(0, 1, 3, 0.4)])  # fmt: skip
    x = frame.to_numpy()
    c_lead = np.corrcoef(x[3:, 1], x[:-3, 0])[0, 1]
    c_back = np.corrcoef(x[3:, 0], x[:-3, 1])[0, 1]
    assert c_lead == pytest.approx(0.4 / np.sqrt(1.16), abs=0.02)
    assert abs(c_back) < 0.03


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_days": 1},
        {"n_assets": 1},
        {"market_loading": 1.0},
        {"factor_ar": 1.0},
        {"df": 2.0},
        {"garch": (0.5, 0.6)},
        {"planted": [(0, 0, 1, 0.1)]},
        {"planted": [(0, 1, 0, 0.1)]},
        {"planted": [(0, 9, 1, 0.1)]},
    ],
)
def test_simulator_rejects_invalid_parameters(kwargs: dict) -> None:
    args: dict = {"n_days": 100, "n_assets": 3}
    args.update(kwargs)
    with pytest.raises(ll.LeadLagError):
        ll.simulate_lead_lag(**args)
