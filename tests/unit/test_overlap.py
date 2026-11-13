"""``etfcorr.correlation.overlap`` (step 2.5).

Covers: matrix alignment (common ticker set, dropped-ticker reporting, the minimum-tickers
guard), ``pairs_frame``'s pair extraction, a hand-computable known-answer regression (a perfect
noiseless linear relationship recovers the exact slope/intercept/R^2 and gives an extreme QAP
p-value), the zero-overlap-variance guard, ``qap_permutation_test``'s basic contract (p-value in
[0, 1], reproducibility under a fixed seed, sensitivity to the seed), the residual-matrix
construction, and the horizon convenience wrapper.
"""

import numpy as np
import pandas as pd
import pytest

from etfcorr.correlation import overlap as O
from etfcorr.correlation.estimators import CorrelationReport

# --------------------------------------------------------------------------------------
# Fixtures: small, hand-controllable overlap/correlation matrices
# --------------------------------------------------------------------------------------

_TICKERS = ("AAA", "BBB", "CCC", "DDD", "EEE")


def _matrix_from_pairs(
    values: dict[tuple[str, str], float], tickers: tuple[str, ...] = _TICKERS
) -> pd.DataFrame:
    n = len(tickers)
    data = np.eye(n)
    for i, a in enumerate(tickers):
        for j, b in enumerate(tickers):
            if i == j:
                continue
            key = (a, b) if (a, b) in values else (b, a)
            data[i, j] = values[key]
    return pd.DataFrame(data, index=tickers, columns=tickers)


_overlap_matrix = _matrix_from_pairs
_corr_matrix = _matrix_from_pairs


def _linear_pair_values(tickers: tuple[str, ...], intercept: float, slope: float, seed: int = 0):
    """Overlap values spread over [0.05, 0.95] and correlation values exactly
    ``intercept + slope * overlap`` for every pair -- a noiseless known-answer fixture."""
    rng = np.random.default_rng(seed)
    overlap_vals: dict[tuple[str, str], float] = {}
    corr_vals: dict[tuple[str, str], float] = {}
    for i, a in enumerate(tickers):
        for b in tickers[i + 1 :]:
            ov = float(rng.uniform(0.05, 0.95))
            overlap_vals[(a, b)] = ov
            corr_vals[(a, b)] = intercept + slope * ov
    return overlap_vals, corr_vals


@pytest.fixture
def linear_matrices() -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    intercept, slope = 0.10, 0.60
    overlap_vals, corr_vals = _linear_pair_values(_TICKERS, intercept, slope, seed=42)
    return _overlap_matrix(overlap_vals), _corr_matrix(corr_vals), intercept, slope


# --------------------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------------------


def test_align_restricts_to_the_common_ticker_set_and_reports_drops(linear_matrices) -> None:
    overlap, corr, _, _ = linear_matrices
    # correlation matrix has an extra ticker the overlap matrix does not (e.g. a control with no
    # holdings snapshot); overlap matrix has an extra ticker the correlation matrix does not.
    extra_corr = corr.copy()
    extra_corr.loc["ZZZ", :] = 0.0
    extra_corr.loc[:, "ZZZ"] = 0.0
    extra_corr.loc["ZZZ", "ZZZ"] = 1.0

    extra_overlap = overlap.copy()
    extra_overlap.loc["YYY", :] = 0.0
    extra_overlap.loc[:, "YYY"] = 0.0
    extra_overlap.loc["YYY", "YYY"] = 1.0

    result = O.fit_overlap_regression(extra_overlap, extra_corr, n_permutations=50)
    assert result.n_tickers == len(_TICKERS)
    assert result.dropped_from_overlap == ("YYY",)
    assert result.dropped_from_corr == ("ZZZ",)


def test_align_rejects_fewer_than_the_minimum_common_tickers() -> None:
    tickers = ("AAA", "BBB", "CCC")
    overlap_vals, corr_vals = _linear_pair_values(tickers, 0.1, 0.5)
    overlap = _overlap_matrix(overlap_vals, tickers)
    corr = _corr_matrix(corr_vals, tickers)
    with pytest.raises(O.OverlapAnalysisError, match="at least"):
        O.pairs_frame(overlap, corr)


def test_align_reuses_each_matrix_s_own_structural_validator() -> None:
    """A structurally invalid overlap matrix (not symmetric) is rejected by
    ``holdings.validate_overlap_matrix`` itself, reused rather than reimplemented here."""
    from etfcorr.data.holdings import HoldingsError

    bad = pd.DataFrame(
        np.array([[1.0, 0.9, 0.1], [0.1, 1.0, 0.2], [0.1, 0.2, 1.0]]),
        index=["AAA", "BBB", "CCC"],
        columns=["AAA", "BBB", "CCC"],
    )
    _, corr_vals = _linear_pair_values(("AAA", "BBB", "CCC"), 0.1, 0.5)
    corr = _corr_matrix(corr_vals, ("AAA", "BBB", "CCC"))
    with pytest.raises(HoldingsError, match="symmetric"):
        O.pairs_frame(bad, corr)


# --------------------------------------------------------------------------------------
# pairs_frame
# --------------------------------------------------------------------------------------


def test_pairs_frame_has_one_row_per_unordered_pair(linear_matrices) -> None:
    overlap, corr, _, _ = linear_matrices
    pairs = O.pairs_frame(overlap, corr)
    n = len(_TICKERS)
    assert len(pairs) == n * (n - 1) // 2
    assert set(pairs.columns) == {"ticker_i", "ticker_j", "overlap", "correlation"}
    # every pair value matches the source matrices exactly
    for _, row in pairs.iterrows():
        assert row["overlap"] == pytest.approx(overlap.loc[row["ticker_i"], row["ticker_j"]])
        assert row["correlation"] == pytest.approx(corr.loc[row["ticker_i"], row["ticker_j"]])


def test_pairs_frame_is_invariant_to_input_row_column_order(linear_matrices) -> None:
    overlap, corr, _, _ = linear_matrices
    reordered = list(reversed(_TICKERS))
    pairs_a = O.pairs_frame(overlap, corr)
    pairs_b = O.pairs_frame(overlap.loc[reordered, reordered], corr.loc[reordered, reordered])

    # same set of pairs and values, regardless of how the inputs were ordered (both realign to
    # the same sorted common ticker set internally)
    def _by_pair(df: pd.DataFrame) -> pd.DataFrame:
        pairs = list(zip(df["ticker_i"], df["ticker_j"], strict=True))
        pair_index = pd.MultiIndex.from_tuples(pairs, names=["ticker_i", "ticker_j"])
        return df[["overlap", "correlation"]].set_axis(pair_index)

    pd.testing.assert_frame_equal(_by_pair(pairs_a), _by_pair(pairs_b))


# --------------------------------------------------------------------------------------
# fit_overlap_regression: noiseless known-answer test
# --------------------------------------------------------------------------------------


def test_fit_overlap_regression_recovers_exact_slope_and_intercept_noiselessly(
    linear_matrices,
) -> None:
    overlap, corr, intercept, slope = linear_matrices
    result = O.fit_overlap_regression(overlap, corr, n_permutations=200, seed=1)
    assert result.intercept == pytest.approx(intercept, abs=1e-9)
    assert result.slope == pytest.approx(slope, abs=1e-9)
    assert result.r_squared == pytest.approx(1.0, abs=1e-9)
    assert result.pearson_r == pytest.approx(1.0, abs=1e-9)
    # a perfect, noiseless relationship must be flagged as extreme by the permutation test: only a
    # random node relabelling that happens to reproduce the identity mapping (or an accidental
    # symmetry of this specific draw) can tie the observed statistic of 1.0, so the p-value must be
    # small but need not hit the theoretical minimum of 1/(n_permutations + 1) exactly -- the
    # identity permutation itself has a nonzero chance of being drawn among the 200 permutations
    # (~200/5! = 1.67 expected hits here), which would count as a tie, not a bug.
    assert result.qap_pearson.observed == pytest.approx(1.0, abs=1e-9)
    assert result.qap_pearson.p_value < 0.05
    assert result.qap_pearson.p_value >= 1.0 / 201.0


def test_residual_matrix_is_near_zero_off_diagonal_for_a_perfect_fit_and_nan_on_the_diagonal(
    linear_matrices,
) -> None:
    overlap, corr, _, _ = linear_matrices
    result = O.fit_overlap_regression(overlap, corr, n_permutations=50)
    resid = result.residual_matrix
    assert resid.index.equals(pd.Index(sorted(_TICKERS)))
    values = resid.to_numpy()
    off_diag_mask = ~np.eye(len(_TICKERS), dtype=bool)
    assert np.all(np.abs(values[off_diag_mask]) < 1e-8)
    assert np.all(np.isnan(np.diag(values)))
    # symmetric by construction (both inputs are symmetric)
    np.testing.assert_allclose(values[off_diag_mask], values.T[off_diag_mask])


def test_fit_overlap_regression_rejects_zero_variance_overlap() -> None:
    tickers = _TICKERS
    overlap = pd.DataFrame(
        np.where(np.eye(len(tickers), dtype=bool), 1.0, 0.5), index=tickers, columns=tickers
    )
    _, corr_vals = _linear_pair_values(tickers, 0.1, 0.5, seed=3)
    corr = _corr_matrix(corr_vals, tickers)
    with pytest.raises(O.OverlapAnalysisError, match="zero variance"):
        O.fit_overlap_regression(overlap, corr)


def test_naive_t_stat_is_reported_but_always_labelled_invalid_in_the_summary(
    linear_matrices,
) -> None:
    overlap, corr, _, _ = linear_matrices
    result = O.fit_overlap_regression(overlap, corr, n_permutations=50)
    lines = O.summary_lines(result)
    naive_line = next(line for line in lines if "naive OLS t-stat" in line)
    assert "INVALID" in naive_line


# --------------------------------------------------------------------------------------
# qap_permutation_test
# --------------------------------------------------------------------------------------


def test_qap_permutation_test_p_value_is_in_unit_interval(linear_matrices) -> None:
    overlap, corr, _, _ = linear_matrices
    result = O.qap_permutation_test(overlap, corr, n_permutations=300, seed=5)
    assert 0.0 < result.p_value <= 1.0
    assert result.n_permutations == 300


def test_qap_permutation_test_is_reproducible_under_the_same_seed(linear_matrices) -> None:
    overlap, corr, _, _ = linear_matrices
    a = O.qap_permutation_test(overlap, corr, n_permutations=200, seed=11)
    b = O.qap_permutation_test(overlap, corr, n_permutations=200, seed=11)
    assert a == b


def test_qap_permutation_test_changes_with_a_different_seed(linear_matrices) -> None:
    overlap, corr, _, _ = linear_matrices
    a = O.qap_permutation_test(overlap, corr, n_permutations=200, seed=11)
    b = O.qap_permutation_test(overlap, corr, n_permutations=200, seed=12)
    assert a.null_mean != pytest.approx(b.null_mean) or a.null_std != pytest.approx(b.null_std)


def test_qap_permutation_test_rejects_fewer_than_two_permutations(linear_matrices) -> None:
    overlap, corr, _, _ = linear_matrices
    with pytest.raises(O.OverlapAnalysisError, match="n_permutations must be >= 2"):
        O.qap_permutation_test(overlap, corr, n_permutations=1)


def test_qap_permutation_test_rejects_an_unknown_statistic(linear_matrices) -> None:
    overlap, corr, _, _ = linear_matrices
    with pytest.raises(O.OverlapAnalysisError, match="unknown statistic"):
        O.qap_permutation_test(overlap, corr, statistic="not_a_statistic")  # type: ignore[arg-type]


def test_qap_permutation_test_applies_a_genuine_joint_node_relabelling(linear_matrices) -> None:
    """With ``n_permutations=2`` (the minimum allowed) the null is exactly two draws, so it can be
    checked by hand: the same two permutations ``np.random.default_rng(seed).permutation(n)`` would
    draw are reproduced independently here and used to relabel BOTH the rows and columns of the
    correlation matrix (a valid node relabelling, per the module docstring); their average must
    match the production function's ``null_mean`` exactly. This is the concrete test that would
    catch a broken implementation that permuted rows only, columns only, or rows and columns
    independently, instead of jointly relabelling nodes."""
    overlap, corr, _, _ = linear_matrices
    n = len(overlap)
    seed = 42

    rng = np.random.default_rng(seed)
    expected_perms = [rng.permutation(n) for _ in range(2)]
    iu = np.triu_indices(n, k=1)
    ov_values = overlap.to_numpy()[iu]
    expected_stats = [
        float(np.corrcoef(ov_values, corr.to_numpy()[np.ix_(perm, perm)][iu])[0, 1])
        for perm in expected_perms
    ]
    expected_null_mean = float(np.mean(expected_stats))

    result = O.qap_permutation_test(overlap, corr, n_permutations=2, seed=seed)
    assert result.null_mean == pytest.approx(expected_null_mean)


def test_qap_permutation_test_rejects_zero_variance_overlap() -> None:
    tickers = _TICKERS
    overlap = pd.DataFrame(
        np.where(np.eye(len(tickers), dtype=bool), 1.0, 0.5), index=tickers, columns=tickers
    )
    _, corr_vals = _linear_pair_values(tickers, 0.1, 0.5, seed=4)
    corr = _corr_matrix(corr_vals, tickers)
    with pytest.raises(O.OverlapAnalysisError, match="zero variance"):
        O.qap_permutation_test(overlap, corr)


# --------------------------------------------------------------------------------------
# overlap_regression_by_horizon
# --------------------------------------------------------------------------------------


def test_overlap_regression_by_horizon_runs_one_regression_per_horizon(linear_matrices) -> None:
    overlap, corr, _, _ = linear_matrices
    by_horizon = {
        "D": CorrelationReport(
            corr=corr,
            method="pearson",
            window=None,
            n_obs=100,
            n_assets=5,
            condition_number=1.0,
            is_psd=True,
            min_eigenvalue=0.1,
        ),
        "W": CorrelationReport(
            corr=corr,
            method="pearson",
            window=None,
            n_obs=20,
            n_assets=5,
            condition_number=1.0,
            is_psd=True,
            min_eigenvalue=0.1,
        ),
    }
    results = O.overlap_regression_by_horizon(overlap, by_horizon, n_permutations=30)
    assert set(results) == {"D", "W"}
    for result in results.values():
        assert result.n_tickers == len(_TICKERS)


def test_overlap_regression_by_horizon_rejects_an_empty_mapping(linear_matrices) -> None:
    overlap, _, _, _ = linear_matrices
    with pytest.raises(O.OverlapAnalysisError, match="empty"):
        O.overlap_regression_by_horizon(overlap, {})
