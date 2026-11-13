"""Unit tests for step 2.3's `correlation/horizon_analysis.py` (basic scope: per-horizon
correlation matrices, rolling/EWMA coupling indicator) and step 3.2's addition (the H1
cross-horizon similarity engine, before vs after detoning, with stationary-bootstrap CIs -- see
the module docstring's "Step 3.2 addition" section for the full method notes)."""

import numpy as np
import pandas as pd
import pytest

from etfcorr.correlation import estimators as E
from etfcorr.correlation import horizon_analysis as H
from etfcorr.correlation import rmt as R
from etfcorr.synthetic import make_noise_returns, true_correlation

# --------------------------------------------------------------------------------------
# average_offdiagonal_corr
# --------------------------------------------------------------------------------------


def test_average_offdiagonal_identity_is_zero() -> None:
    corr = pd.DataFrame(np.eye(4), index=list("ABCD"), columns=list("ABCD"))
    assert H.average_offdiagonal_corr(corr) == pytest.approx(0.0)


def test_average_offdiagonal_constant_value() -> None:
    n = 5
    rho = 0.6
    values = np.full((n, n), rho)
    np.fill_diagonal(values, 1.0)
    corr = pd.DataFrame(values, index=range(n), columns=range(n))
    assert H.average_offdiagonal_corr(corr) == pytest.approx(rho)


def test_average_offdiagonal_hand_computed_3x3() -> None:
    # off-diagonal entries 0.2, 0.4, 0.8 (upper triangle), mean = 1.4/3
    values = np.array(
        [
            [1.0, 0.2, 0.4],
            [0.2, 1.0, 0.8],
            [0.4, 0.8, 1.0],
        ]
    )
    corr = pd.DataFrame(values, index=list("XYZ"), columns=list("XYZ"))
    assert H.average_offdiagonal_corr(corr) == pytest.approx((0.2 + 0.4 + 0.8) / 3)


def test_average_offdiagonal_rejects_fewer_than_two_assets() -> None:
    corr = pd.DataFrame([[1.0]], index=["A"], columns=["A"])
    with pytest.raises(H.HorizonAnalysisError, match="at least 2 assets"):
        H.average_offdiagonal_corr(corr)


def test_average_offdiagonal_rejects_non_square() -> None:
    corr = pd.DataFrame(np.ones((2, 3)))
    with pytest.raises(H.HorizonAnalysisError, match="square"):
        H.average_offdiagonal_corr(corr)


def test_average_offdiagonal_rejects_non_dataframe() -> None:
    with pytest.raises(H.HorizonAnalysisError, match="DataFrame"):
        H.average_offdiagonal_corr(np.eye(3))  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# corr_by_horizon
# --------------------------------------------------------------------------------------


def test_corr_by_horizon_matches_direct_corr_report_call(factor_panel) -> None:
    returns = factor_panel.returns
    by_horizon = {"D": returns, "W": returns.iloc[::5]}
    out = H.corr_by_horizon(by_horizon, "pearson", window=200)

    assert list(out.keys()) == ["D", "W"]  # order preserved
    for horizon, returns_h in by_horizon.items():
        expected = E.corr_report(returns_h, "pearson", window=200)
        pd.testing.assert_frame_equal(out[horizon].corr, expected.corr)
        assert out[horizon].n_obs == expected.n_obs
        assert out[horizon].condition_number == expected.condition_number


def test_corr_by_horizon_rejects_empty_mapping() -> None:
    with pytest.raises(H.HorizonAnalysisError, match="empty"):
        H.corr_by_horizon({}, "pearson")


def test_corr_by_horizon_forwards_method_and_kw(factor_panel) -> None:
    returns = factor_panel.returns
    out = H.corr_by_horizon({"D": returns}, "ledoit_wolf")
    assert out["D"].method == "ledoit_wolf"
    E.validate_correlation_matrix(out["D"].corr)


# --------------------------------------------------------------------------------------
# rolling_coupling_indicator
# --------------------------------------------------------------------------------------


def test_rolling_coupling_indicator_length_and_index(factor_panel) -> None:
    returns = factor_panel.returns.iloc[:400]
    window = 60
    out = H.rolling_coupling_indicator(returns, "pearson", window)

    expected_positions = list(range(window - 1, len(returns), 1))
    assert len(out) == len(expected_positions)
    assert list(out.index) == [returns.index[i] for i in expected_positions]


def test_rolling_coupling_indicator_step_downsamples(factor_panel) -> None:
    returns = factor_panel.returns.iloc[:400]
    window = 60
    out = H.rolling_coupling_indicator(returns, "pearson", window, step=10)
    expected_positions = list(range(window - 1, len(returns), 10))
    assert len(out) == len(expected_positions)
    assert list(out.index) == [returns.index[i] for i in expected_positions]


def test_rolling_coupling_indicator_spot_check_matches_direct_call(factor_panel) -> None:
    """Cross-check: a value picked out of the rolling series must equal an independent, direct
    estimate()+average_offdiagonal_corr() call on the same trailing window -- not just trust the
    loop that produced it."""
    returns = factor_panel.returns.iloc[:400]
    window = 60
    out = H.rolling_coupling_indicator(returns, "pearson", window, step=25)

    # pick an actual index position present in `out` rather than assuming step arithmetic
    some_date = out.index[10]
    pos = returns.index.get_loc(some_date)
    direct = E.estimate(returns.iloc[: pos + 1], "pearson", window=window)
    assert out.loc[some_date] == pytest.approx(H.average_offdiagonal_corr(direct))


def test_rolling_coupling_indicator_recovers_population_truth(factor_panel) -> None:
    """Known-answer test against the synthetic panel's population correlation (project_plan.md's
    own synthetic-data validation style, not just plumbing): at a large trailing window the
    estimated average off-diagonal correlation should be close to the true one."""
    returns = factor_panel.returns
    window = 1400
    out = H.rolling_coupling_indicator(returns, "pearson", window, step=1)
    true_corr = pd.DataFrame(factor_panel.true_corr, index=returns.columns, columns=returns.columns)
    true_avg = H.average_offdiagonal_corr(true_corr)
    # T=1400, N=9: sampling noise is small but not zero; a loose tolerance avoids a flaky test.
    assert out.iloc[-1] == pytest.approx(true_avg, abs=0.05)


def test_rolling_coupling_indicator_rejects_ewma() -> None:
    returns = pd.DataFrame(np.random.default_rng(0).standard_normal((100, 4)))
    with pytest.raises(H.HorizonAnalysisError, match="ewma_coupling_indicator"):
        H.rolling_coupling_indicator(returns, "ewma", 20)


def test_rolling_coupling_indicator_rejects_too_small_window(factor_panel) -> None:
    with pytest.raises(H.HorizonAnalysisError, match="window"):
        H.rolling_coupling_indicator(factor_panel.returns, "pearson", 1)


def test_rolling_coupling_indicator_rejects_too_small_step(factor_panel) -> None:
    with pytest.raises(H.HorizonAnalysisError, match="step"):
        H.rolling_coupling_indicator(factor_panel.returns, "pearson", 60, step=0)


def test_rolling_coupling_indicator_rejects_fewer_rows_than_window() -> None:
    returns = pd.DataFrame(np.random.default_rng(0).standard_normal((10, 3)))
    with pytest.raises(H.HorizonAnalysisError, match="fewer than window"):
        H.rolling_coupling_indicator(returns, "pearson", 20)


def test_rolling_coupling_indicator_values_in_valid_range(factor_panel) -> None:
    out = H.rolling_coupling_indicator(factor_panel.returns.iloc[:300], "pearson", 60, step=15)
    assert (out >= -1.0 - 1e-9).all()
    assert (out <= 1.0 + 1e-9).all()


# --------------------------------------------------------------------------------------
# ewma_coupling_indicator
# --------------------------------------------------------------------------------------


def test_ewma_coupling_indicator_last_value_matches_direct_call(factor_panel) -> None:
    returns = factor_panel.returns.iloc[:300]
    out = H.ewma_coupling_indicator(returns, lam=0.9, min_periods=2, step=1)
    direct = E.ewma_corr(returns, lam=0.9, min_periods=2)
    assert out.iloc[-1] == pytest.approx(H.average_offdiagonal_corr(direct))


def test_ewma_coupling_indicator_length_and_index(factor_panel) -> None:
    returns = factor_panel.returns.iloc[:200]
    out = H.ewma_coupling_indicator(returns, min_periods=5, step=1)
    expected_positions = list(range(4, len(returns), 1))
    assert list(out.index) == [returns.index[i] for i in expected_positions]


def test_ewma_coupling_indicator_step_downsamples(factor_panel) -> None:
    returns = factor_panel.returns.iloc[:200]
    out = H.ewma_coupling_indicator(returns, min_periods=5, step=20)
    expected_positions = list(range(4, len(returns), 20))
    assert list(out.index) == [returns.index[i] for i in expected_positions]


def test_ewma_coupling_indicator_window_cap_matches_direct_call(factor_panel) -> None:
    returns = factor_panel.returns.iloc[:300]
    out = H.ewma_coupling_indicator(returns, lam=0.8, min_periods=2, window=50, step=1)
    direct = E.ewma_corr(returns, lam=0.8, min_periods=2, window=50)
    assert out.iloc[-1] == pytest.approx(H.average_offdiagonal_corr(direct))


def test_ewma_coupling_indicator_rejects_too_few_rows() -> None:
    returns = pd.DataFrame(np.random.default_rng(0).standard_normal((3, 3)))
    with pytest.raises(H.HorizonAnalysisError, match="min_periods"):
        H.ewma_coupling_indicator(returns, min_periods=10)


def test_ewma_coupling_indicator_rejects_bad_step(factor_panel) -> None:
    with pytest.raises(H.HorizonAnalysisError, match="step"):
        H.ewma_coupling_indicator(factor_panel.returns, step=0)


# --------------------------------------------------------------------------------------
# coupling_indicator (dispatcher)
# --------------------------------------------------------------------------------------


def test_coupling_indicator_dispatches_to_rolling(factor_panel) -> None:
    returns = factor_panel.returns.iloc[:300]
    dispatched = H.coupling_indicator(returns, "pearson", window=60, step=10)
    direct = H.rolling_coupling_indicator(returns, "pearson", 60, step=10)
    pd.testing.assert_series_equal(dispatched, direct)


def test_coupling_indicator_dispatches_to_ewma(factor_panel) -> None:
    returns = factor_panel.returns.iloc[:300]
    dispatched = H.coupling_indicator(returns, "ewma", lam=0.9, min_periods=3, step=5)
    direct = H.ewma_coupling_indicator(returns, lam=0.9, min_periods=3, step=5)
    pd.testing.assert_series_equal(dispatched, direct)


def test_coupling_indicator_ewma_defaults_lambda(factor_panel) -> None:
    returns = factor_panel.returns.iloc[:300]
    dispatched = H.coupling_indicator(returns, "ewma", step=5)
    direct = H.ewma_coupling_indicator(returns, E.DEFAULT_EWMA_LAMBDA, step=5)
    pd.testing.assert_series_equal(dispatched, direct)


def test_coupling_indicator_non_ewma_requires_window(factor_panel) -> None:
    with pytest.raises(H.HorizonAnalysisError, match="requires window"):
        H.coupling_indicator(factor_panel.returns, "pearson")


# --------------------------------------------------------------------------------------
# Step 3.2: frobenius_distance
# --------------------------------------------------------------------------------------


def _equicorr(n: int, rho: float, labels: list[str]) -> pd.DataFrame:
    values = np.full((n, n), rho)
    np.fill_diagonal(values, 1.0)
    return pd.DataFrame(values, index=labels, columns=labels)


def test_frobenius_distance_identical_matrices_is_zero() -> None:
    a = _equicorr(4, 0.4, list("ABCD"))
    assert H.frobenius_distance(a, a) == pytest.approx(0.0, abs=1e-12)


def test_frobenius_distance_hand_computed() -> None:
    a = pd.DataFrame([[1.0, 0.2], [0.2, 1.0]], index=["X", "Y"], columns=["X", "Y"])
    b = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["X", "Y"], columns=["X", "Y"])
    # off-diagonal differs by 0.3 in both (symmetric) entries; diagonal contributes 0.
    expected = float(np.sqrt(2 * 0.3**2))
    assert H.frobenius_distance(a, b) == pytest.approx(expected)


def test_frobenius_distance_symmetric_in_its_arguments() -> None:
    a = _equicorr(5, 0.3, list("ABCDE"))
    b = _equicorr(5, 0.6, list("ABCDE"))
    assert H.frobenius_distance(a, b) == pytest.approx(H.frobenius_distance(b, a))


def test_frobenius_distance_rejects_mismatched_labels() -> None:
    a = _equicorr(3, 0.3, list("ABC"))
    b = _equicorr(3, 0.3, list("ABD"))
    with pytest.raises(H.HorizonAnalysisError, match="identical row/column labels"):
        H.frobenius_distance(a, b)


def test_frobenius_distance_rejects_invalid_matrix() -> None:
    a = _equicorr(3, 0.3, list("ABC"))
    bad = pd.DataFrame(
        np.diag([2.0, 2.0, 2.0]), index=list("ABC"), columns=list("ABC")
    )  # diagonal is not all ones
    with pytest.raises(E.CorrelationError):
        H.frobenius_distance(a, bad)


# --------------------------------------------------------------------------------------
# Step 3.2: leading_eigenvector_overlap
# --------------------------------------------------------------------------------------


def test_leading_eigenvector_overlap_identical_matrix_is_one() -> None:
    a = _equicorr(6, 0.35, list("ABCDEF"))
    assert H.leading_eigenvector_overlap(a, a) == pytest.approx(1.0, abs=1e-10)


def test_leading_eigenvector_overlap_known_answer_equicorrelation() -> None:
    """The top eigenvector of *any* equicorrelation matrix is exactly the all-ones vector,
    regardless of rho (a hand-derivable fact: an equicorrelation matrix has constant row sums,
    so the all-ones vector is always an eigenvector). Two equicorrelation matrices at different
    rho must therefore have overlap exactly 1, not merely approximately."""
    a = _equicorr(7, 0.1, list("ABCDEFG"))
    b = _equicorr(7, 0.8, list("ABCDEFG"))
    assert H.leading_eigenvector_overlap(a, b) == pytest.approx(1.0, abs=1e-9)


def test_leading_eigenvector_overlap_symmetric_in_its_arguments() -> None:
    a = _equicorr(5, 0.2, list("ABCDE"))
    b = _equicorr(5, 0.5, list("ABCDE"))
    assert H.leading_eigenvector_overlap(a, b) == pytest.approx(H.leading_eigenvector_overlap(b, a))


def test_leading_eigenvector_overlap_bounded_in_unit_interval(factor_panel) -> None:
    returns = factor_panel.returns
    a = E.estimate(returns.iloc[:700], "pearson")
    b = E.estimate(returns.iloc[700:1400], "pearson")
    overlap = H.leading_eigenvector_overlap(a, b)
    assert -1e-9 <= overlap <= 1.0 + 1e-9


def test_leading_eigenvector_overlap_rejects_invalid_n_top() -> None:
    a = _equicorr(4, 0.3, list("ABCD"))
    with pytest.raises(H.HorizonAnalysisError, match="n_top"):
        H.leading_eigenvector_overlap(a, a, n_top=4)
    with pytest.raises(H.HorizonAnalysisError, match="n_top"):
        H.leading_eigenvector_overlap(a, a, n_top=0)


def test_leading_eigenvector_overlap_rejects_mismatched_labels() -> None:
    a = _equicorr(3, 0.3, list("ABC"))
    b = _equicorr(3, 0.3, list("ABD"))
    with pytest.raises(H.HorizonAnalysisError, match="identical row/column labels"):
        H.leading_eigenvector_overlap(a, b)


# --------------------------------------------------------------------------------------
# Step 3.2: horizon_pair_similarity -- wiring, and a deterministic, hand-constructed H1
# mechanism example (analytic population matrices, no sampling noise)
# --------------------------------------------------------------------------------------


def test_horizon_pair_similarity_identical_matrices_gives_zero_diffs() -> None:
    a = _equicorr(5, 0.4, list("ABCDE"))
    sim = H.horizon_pair_similarity(a, a, "D", "W")
    assert sim.frobenius_before == pytest.approx(0.0, abs=1e-12)
    assert sim.frobenius_after == pytest.approx(0.0, abs=1e-9)
    assert sim.frobenius_diff == pytest.approx(0.0, abs=1e-9)
    assert sim.eigvec_diff == pytest.approx(0.0, abs=1e-6)


def test_horizon_pair_similarity_matches_direct_calls(factor_panel) -> None:
    returns = factor_panel.returns
    a = E.estimate(returns.iloc[:700], "pearson")
    b = E.estimate(returns.iloc[700:1400], "pearson")
    sim = H.horizon_pair_similarity(a, b, "D", "W", n_market=1)

    detoned_a = R.detone_corr(a, n_market=1)
    detoned_b = R.detone_corr(b, n_market=1)
    assert sim.frobenius_before == pytest.approx(H.frobenius_distance(a, b))
    assert sim.frobenius_after == pytest.approx(H.frobenius_distance(detoned_a, detoned_b))
    assert sim.eigvec_before == pytest.approx(H.leading_eigenvector_overlap(a, b))
    assert sim.eigvec_after == pytest.approx(H.leading_eigenvector_overlap(detoned_a, detoned_b))
    assert sim.frobenius_diff == pytest.approx(sim.frobenius_before - sim.frobenius_after)
    assert sim.eigvec_diff == pytest.approx(sim.eigvec_after - sim.eigvec_before)


def test_horizon_pair_similarity_n_top_defaults_to_n_market(factor_panel) -> None:
    # an equicorrelation matrix's non-market eigenvalue is degenerate (multiplicity n-1), which
    # makes detone_corr(n_market=2) remove an arbitrary basis of that degenerate subspace and can
    # leave near-zero residual variance (see leading_eigenvector_overlap's own docstring caveat);
    # a real estimated matrix (9 assets, no exact degeneracy) is used here instead, purely to
    # check the n_top wiring, not the numeric detoning result.
    returns = factor_panel.returns
    a = E.estimate(returns.iloc[:700], "pearson")
    b = E.estimate(returns.iloc[700:1400], "pearson")
    sim = H.horizon_pair_similarity(a, b, "D", "M", n_market=2)
    assert sim.n_top == 2


def test_horizon_pair_similarity_known_answer_market_loading_difference() -> None:
    """Hand-constructed H1 mechanism example, fully deterministic (analytic population
    matrices, no sampling): two panels share the *same* sector structure (sector_loading=0.5,
    sector_sizes=(2, 3, 4)) but differ in market loading (0.3 vs 0.7) -- exactly modelling "the
    market mode's strength differs across horizons, the rest of the structure does not," H1's own
    premise (Borghesi, Marsili, Micciche 2007). Values computed once this session via
    horizon_pair_similarity itself and fixed here as a regression/known-answer check (not
    re-derived by hand for every entry -- the *sign and direction* is the substantive claim under
    test: detoning must make these two matrices strictly more similar by both metrics, since they
    differ from each other only in their market-mode strength)."""
    sector_sizes = (2, 3, 4)
    labels = [f"S{s}_{k}" for s, size in enumerate(sector_sizes) for k in range(size)]
    a = pd.DataFrame(
        true_correlation(sector_sizes, market_loading=0.3, sector_loading=0.5),
        index=labels,
        columns=labels,
    )
    b = pd.DataFrame(
        true_correlation(sector_sizes, market_loading=0.7, sector_loading=0.5),
        index=labels,
        columns=labels,
    )
    sim = H.horizon_pair_similarity(a, b, "D", "W")

    assert sim.frobenius_before == pytest.approx(3.3941125496954276)
    assert sim.frobenius_after == pytest.approx(1.3256429927107836)
    assert sim.eigvec_before == pytest.approx(0.9708037536143556)
    assert sim.eigvec_after == pytest.approx(0.98139424923717)
    # the substantive claim: detoning makes them more similar by both metrics.
    assert sim.frobenius_diff > 0
    assert sim.eigvec_diff > 0


# --------------------------------------------------------------------------------------
# Step 3.2: cross_horizon_similarity
# --------------------------------------------------------------------------------------


def test_cross_horizon_similarity_covers_every_pair(factor_panel) -> None:
    returns = factor_panel.returns
    by_horizon = {"D": returns, "W": returns.iloc[::5], "M": returns.iloc[::21]}
    out = H.cross_horizon_similarity(by_horizon, "pearson", window=200)
    assert set(out.keys()) == {("D", "W"), ("D", "M"), ("W", "M")}
    for (a, b), sim in out.items():
        assert sim.horizon_a == a
        assert sim.horizon_b == b


def test_cross_horizon_similarity_default_horizons_exclude_quarterly(factor_panel) -> None:
    returns = factor_panel.returns
    by_horizon = {
        "D": returns,
        "W": returns.iloc[::5],
        "M": returns.iloc[::21],
        "Q": returns.iloc[::63],
    }
    out = H.cross_horizon_similarity(by_horizon, "pearson", window=200)
    assert ("D", "Q") not in out
    assert ("W", "Q") not in out
    assert ("M", "Q") not in out


def test_cross_horizon_similarity_rejects_fewer_than_two_horizons(factor_panel) -> None:
    with pytest.raises(H.HorizonAnalysisError, match="at least 2 horizons"):
        H.cross_horizon_similarity({"D": factor_panel.returns}, "pearson", horizons=("D",))


def test_cross_horizon_similarity_rejects_missing_horizon() -> None:
    returns = pd.DataFrame(np.random.default_rng(0).standard_normal((300, 5)))
    with pytest.raises(H.HorizonAnalysisError, match="not present"):
        H.cross_horizon_similarity({"D": returns}, "pearson", horizons=("D", "W"))


# --------------------------------------------------------------------------------------
# Step 3.2: stationary_bootstrap_indices
# --------------------------------------------------------------------------------------


def test_stationary_bootstrap_indices_length_and_range() -> None:
    rng = np.random.default_rng(0)
    idx = H.stationary_bootstrap_indices(100, 10.0, rng)
    assert len(idx) == 100
    assert idx.min() >= 0
    assert idx.max() < 100


def test_stationary_bootstrap_indices_deterministic_given_seed() -> None:
    idx_a = H.stationary_bootstrap_indices(50, 5.0, np.random.default_rng(42))
    idx_b = H.stationary_bootstrap_indices(50, 5.0, np.random.default_rng(42))
    np.testing.assert_array_equal(idx_a, idx_b)


def test_stationary_bootstrap_indices_large_block_length_is_one_contiguous_wrap() -> None:
    """With mean_block_length >> n, the single drawn block length almost certainly exceeds n, so
    the whole resample is one contiguous (possibly wrapping) run starting at a random position."""
    rng = np.random.default_rng(7)
    n = 40
    idx = H.stationary_bootstrap_indices(n, 1e9, rng)
    start = int(idx[0])
    expected = np.array([(start + i) % n for i in range(n)])
    np.testing.assert_array_equal(idx, expected)


def test_stationary_bootstrap_indices_unit_block_length_is_not_just_arange() -> None:
    """mean_block_length=1 means every position independently restarts a new random block (the
    ordinary i.i.d. bootstrap); the result should not just be the identity permutation."""
    rng = np.random.default_rng(3)
    idx = H.stationary_bootstrap_indices(200, 1.0, rng)
    assert not np.array_equal(idx, np.arange(200))


def test_stationary_bootstrap_indices_rejects_invalid_n() -> None:
    with pytest.raises(H.HorizonAnalysisError, match="n must be"):
        H.stationary_bootstrap_indices(0, 5.0, np.random.default_rng(0))


def test_stationary_bootstrap_indices_rejects_invalid_mean_block_length() -> None:
    with pytest.raises(H.HorizonAnalysisError, match="mean_block_length"):
        H.stationary_bootstrap_indices(10, 0.0, np.random.default_rng(0))


# --------------------------------------------------------------------------------------
# Step 3.2: default_mean_block_length
# --------------------------------------------------------------------------------------


def test_default_mean_block_length_known_values() -> None:
    assert H.default_mean_block_length(8) == pytest.approx(2.0)  # cube root of 8 is exactly 2
    assert H.default_mean_block_length(1000) == pytest.approx(10.0)  # cube root of 1000 is 10


def test_default_mean_block_length_floors_at_two() -> None:
    assert H.default_mean_block_length(2) == pytest.approx(2.0)


def test_default_mean_block_length_rejects_too_few_obs() -> None:
    with pytest.raises(H.HorizonAnalysisError, match="n_obs"):
        H.default_mean_block_length(1)


# --------------------------------------------------------------------------------------
# Step 3.2: bootstrap_horizon_pair_similarity
# --------------------------------------------------------------------------------------

_BOOT_N = 150  # kept small for test runtime; production runs use config/experiment.yaml's 2000


def test_bootstrap_horizon_pair_similarity_is_deterministic(factor_panel) -> None:
    returns = factor_panel.returns.iloc[:400]
    returns_w = returns.iloc[::5]
    point1, fro1, eig1 = H.bootstrap_horizon_pair_similarity(
        returns, returns_w, "D", "W", n_bootstrap=_BOOT_N, seed=11
    )
    point2, fro2, eig2 = H.bootstrap_horizon_pair_similarity(
        returns, returns_w, "D", "W", n_bootstrap=_BOOT_N, seed=11
    )
    assert point1 == point2
    assert fro1.replicates == fro2.replicates
    assert eig1.replicates == eig2.replicates


def test_bootstrap_horizon_pair_similarity_point_estimate_matches_direct_call(factor_panel) -> None:
    returns = factor_panel.returns.iloc[:400]
    returns_w = returns.iloc[::5]
    point, _, _ = H.bootstrap_horizon_pair_similarity(
        returns, returns_w, "D", "W", n_bootstrap=_BOOT_N, seed=0
    )
    direct = H.horizon_pair_similarity(
        E.estimate(returns, "pearson"), E.estimate(returns_w, "pearson"), "D", "W"
    )
    assert point == direct


def test_bootstrap_horizon_pair_similarity_ci_bounds_are_ordered(factor_panel) -> None:
    returns = factor_panel.returns.iloc[:400]
    returns_w = returns.iloc[::5]
    _, fro_ci, eig_ci = H.bootstrap_horizon_pair_similarity(
        returns, returns_w, "D", "W", n_bootstrap=_BOOT_N, seed=2
    )
    assert fro_ci.lower <= fro_ci.upper
    assert eig_ci.lower <= eig_ci.upper
    assert len(fro_ci.replicates) == _BOOT_N
    assert len(eig_ci.replicates) == _BOOT_N
    assert fro_ci.alpha == pytest.approx(0.05)


def test_bootstrap_horizon_pair_similarity_window_restricts_before_resampling(factor_panel) -> None:
    returns = factor_panel.returns.iloc[:400]
    returns_w = returns.iloc[::5]
    _, fro_ci, _ = H.bootstrap_horizon_pair_similarity(
        returns, returns_w, "D", "W", window=100, n_bootstrap=_BOOT_N, seed=0
    )
    assert fro_ci.mean_block_length_a == pytest.approx(H.default_mean_block_length(100))


def test_bootstrap_horizon_pair_similarity_respects_explicit_block_lengths(factor_panel) -> None:
    returns = factor_panel.returns.iloc[:400]
    returns_w = returns.iloc[::5]
    _, fro_ci, _ = H.bootstrap_horizon_pair_similarity(
        returns,
        returns_w,
        "D",
        "W",
        n_bootstrap=_BOOT_N,
        seed=0,
        mean_block_length_a=3.0,
        mean_block_length_b=2.0,
    )
    assert fro_ci.mean_block_length_a == pytest.approx(3.0)
    assert fro_ci.mean_block_length_b == pytest.approx(2.0)


def test_bootstrap_ci_excludes_zero_property() -> None:
    ci_excl = H.BootstrapCI(
        point_estimate=1.0,
        lower=0.2,
        upper=1.8,
        alpha=0.05,
        n_bootstrap=10,
        mean_block_length_a=2.0,
        mean_block_length_b=2.0,
        replicates=(1.0,) * 10,
    )
    ci_incl = H.BootstrapCI(
        point_estimate=0.1,
        lower=-0.5,
        upper=0.5,
        alpha=0.05,
        n_bootstrap=10,
        mean_block_length_a=2.0,
        mean_block_length_b=2.0,
        replicates=(0.1,) * 10,
    )
    assert ci_excl.excludes_zero
    assert not ci_incl.excludes_zero


def test_bootstrap_horizon_pair_similarity_null_noise_eigvec_ci_contains_zero() -> None:
    """A characterization/synthetic-noise check, in the spirit of the project's O4 rule (skill
    should be ~0 under a null of no true structure) -- **with an important, disclosed, measured
    exception for the Frobenius metric** (see the module docstring's "Known limitation,
    empirically measured" paragraph): under a pure-noise population (identity true correlation),
    the eigenvector-overlap difference's CI reliably contains 0 across independent seeds, but the
    Frobenius difference's does **not** (it is systematically negative -- an artifact of
    renormalizing after removing a noise-inflated top eigenvalue, not evidence of anything). This
    test checks the metric that actually behaves like a null here; it does not assert the
    Frobenius behaviour, which is checked (and its sign direction pinned down) by
    ``test_bootstrap_horizon_pair_similarity_null_noise_frobenius_is_biased_negative`` below."""
    contains_zero = 0
    n_seeds = 5
    for seed in range(n_seeds):
        panel = make_noise_returns(500, 8, seed=seed)
        returns_d = panel.returns
        returns_w = panel.returns.iloc[::5]
        _, _, eig_ci = H.bootstrap_horizon_pair_similarity(
            returns_d, returns_w, "D", "W", n_bootstrap=_BOOT_N, seed=seed, alpha=0.1
        )
        if not eig_ci.excludes_zero:
            contains_zero += 1
    # loose: allow up to one seed's CI to miss (a 90% CI is expected to miss sometimes), but the
    # large majority should contain 0 under a genuine null.
    assert contains_zero >= n_seeds - 1


def test_bootstrap_horizon_pair_similarity_null_noise_frobenius_is_biased_negative() -> None:
    """Pins down the disclosed Frobenius-under-null finding (see the module docstring) as a
    regression check: every one of several independent noise-panel seeds gives a negative
    ``frobenius_diff`` point estimate. If this ever turns up mixed-sign or positive, the module
    docstring's disclosed finding (and this test) need to be revisited, not silently left stale."""
    for seed in range(5):
        panel = make_noise_returns(500, 8, seed=seed)
        returns_d = panel.returns
        returns_w = panel.returns.iloc[::5]
        point, _, _ = H.bootstrap_horizon_pair_similarity(
            returns_d, returns_w, "D", "W", n_bootstrap=_BOOT_N, seed=seed
        )
        assert point.frobenius_diff < 0


# --------------------------------------------------------------------------------------
# Step 3.2: h1_cross_horizon_test (the full entry point)
# --------------------------------------------------------------------------------------


def test_h1_cross_horizon_test_covers_every_pair_and_matches_direct_calls(factor_panel) -> None:
    returns = factor_panel.returns.iloc[:400]
    by_horizon = {"D": returns, "W": returns.iloc[::5], "M": returns.iloc[::21]}
    out = H.h1_cross_horizon_test(
        by_horizon, "pearson", horizons=("D", "W", "M"), n_bootstrap=_BOOT_N, seed=5
    )
    assert set(out.keys()) == {("D", "W"), ("D", "M"), ("W", "M")}

    # cross-check one pair against a direct bootstrap_horizon_pair_similarity call with the same
    # (offset) seed -- pair ("D", "W") is first in itertools.combinations order, offset 0.
    point, fro_ci, eig_ci = H.bootstrap_horizon_pair_similarity(
        by_horizon["D"], by_horizon["W"], "D", "W", n_bootstrap=_BOOT_N, seed=5
    )
    dw = out[("D", "W")]
    assert dw.similarity == point
    assert dw.frobenius_ci.replicates == fro_ci.replicates
    assert dw.eigvec_ci.replicates == eig_ci.replicates


def test_h1_cross_horizon_test_rejects_fewer_than_two_horizons(factor_panel) -> None:
    with pytest.raises(H.HorizonAnalysisError, match="at least 2 horizons"):
        H.h1_cross_horizon_test(
            {"D": factor_panel.returns}, "pearson", horizons=("D",), n_bootstrap=_BOOT_N
        )


def test_h1_cross_horizon_test_rejects_missing_horizon() -> None:
    returns = pd.DataFrame(np.random.default_rng(0).standard_normal((300, 5)))
    with pytest.raises(H.HorizonAnalysisError, match="not present"):
        H.h1_cross_horizon_test({"D": returns}, "pearson", horizons=("D", "W"), n_bootstrap=_BOOT_N)


# --------------------------------------------------------------------------------------
# Step 3.2: h1_pair_summary_line
# --------------------------------------------------------------------------------------


def test_h1_pair_summary_line_contains_horizon_labels_and_metrics(factor_panel) -> None:
    returns = factor_panel.returns.iloc[:400]
    returns_w = returns.iloc[::5]
    point, fro_ci, eig_ci = H.bootstrap_horizon_pair_similarity(
        returns, returns_w, "D", "W", n_bootstrap=_BOOT_N, seed=0
    )
    result = H.H1PairResult(similarity=point, frobenius_ci=fro_ci, eigvec_ci=eig_ci)
    line = H.h1_pair_summary_line(result)
    assert "D-W" in line
    assert "Frobenius" in line
    assert "eigvec" in line
