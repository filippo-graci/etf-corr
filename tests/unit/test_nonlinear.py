"""Unit tests for ``correlation/nonlinear.py`` (step 3.4a): known-answer cases, structural
invariants, and the permutation-test / FDR machinery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etfcorr.correlation import nonlinear as NL

SEED = 20260928


def test_distance_correlation_is_one_for_a_perfect_linear_relationship() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(300)
    y = 2.0 * x + 3.0
    assert NL.distance_correlation(x, y) == pytest.approx(1.0, abs=1e-8)


def test_distance_correlation_is_small_for_independent_gaussians() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(2000)
    y = rng.standard_normal(2000)
    value = NL.distance_correlation(x, y)
    assert 0.0 <= value < 0.1


def test_distance_correlation_catches_nonlinear_dependence_pearson_misses() -> None:
    """Classic textbook case: Y = X^2 (centred) has Pearson corr near 0 but real dependence."""
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(3000)
    y = x**2 - 1.0  # E[X^2] = 1 for standard normal
    pearson = float(np.corrcoef(x, y)[0, 1])
    dcor = NL.distance_correlation(x, y)
    assert abs(pearson) < 0.1
    assert dcor > 0.2
    assert dcor > abs(pearson)


def test_distance_correlation_rejects_mismatched_lengths() -> None:
    with pytest.raises(NL.NonlinearError, match="same length"):
        NL.distance_correlation(np.zeros(10), np.zeros(9))


def test_distance_correlation_rejects_nan() -> None:
    x = np.array([1.0, 2.0, np.nan, 4.0])
    with pytest.raises(NL.NonlinearError, match="NaN"):
        NL.distance_correlation(x, x)


def test_distance_correlation_matrix_is_symmetric_unit_diagonal(factor_panel) -> None:
    corr = NL.distance_correlation_matrix(factor_panel.returns)
    values = corr.to_numpy(dtype=float)
    np.testing.assert_allclose(values, values.T, atol=1e-8)
    np.testing.assert_allclose(np.diag(values), 1.0, atol=1e-8)
    assert values.min() >= -1e-8
    assert values.max() <= 1.0 + 1e-8


def test_distance_correlation_matrix_diagonal_matches_pairwise_function(factor_panel) -> None:
    """Cross-check: the (i, j) entry of the matrix equals a direct pairwise call."""
    returns = factor_panel.returns.iloc[:, :4]
    matrix = NL.distance_correlation_matrix(returns)
    direct = NL.distance_correlation(
        returns.iloc[:, 0].to_numpy(dtype=float), returns.iloc[:, 2].to_numpy(dtype=float)
    )
    assert matrix.iloc[0, 2] == pytest.approx(direct, abs=1e-10)


def test_distance_correlation_matrix_rejects_too_few_columns() -> None:
    with pytest.raises(NL.NonlinearError, match="at least 2 columns"):
        NL.distance_correlation_matrix(pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]}))


def test_distance_correlation_matrix_window_is_trailing(factor_panel) -> None:
    returns = factor_panel.returns
    window = 200
    matrix = NL.distance_correlation_matrix(returns, window=window)
    manual = NL.distance_correlation_matrix(returns.tail(window))
    pd.testing.assert_frame_equal(matrix, manual)


# --------------------------------------------------------------------------------------
# Mutual information
# --------------------------------------------------------------------------------------


def test_mutual_information_matrix_has_nan_diagonal_and_is_symmetric(factor_panel) -> None:
    returns = factor_panel.returns.iloc[:, :4]
    mi = NL.mutual_information_matrix(returns, random_state=0)
    values = mi.to_numpy(dtype=float)
    assert np.all(np.isnan(np.diag(values)))
    off_diag = ~np.eye(values.shape[0], dtype=bool)
    np.testing.assert_allclose(values[off_diag], values.T[off_diag], atol=1e-8)


def test_mutual_information_is_higher_for_dependent_than_independent_pairs() -> None:
    rng = np.random.default_rng(SEED)
    n = 2000
    x = rng.standard_normal(n)
    dependent = x + 0.01 * rng.standard_normal(n)
    independent = rng.standard_normal(n)
    frame = pd.DataFrame({"x": x, "dep": dependent, "indep": independent})
    mi = NL.mutual_information_matrix(frame, random_state=0)
    assert mi.loc["x", "dep"] > mi.loc["x", "indep"]


def test_mutual_information_matrix_is_deterministic_given_random_state(factor_panel) -> None:
    returns = factor_panel.returns.iloc[:, :4]
    a = NL.mutual_information_matrix(returns, random_state=7)
    b = NL.mutual_information_matrix(returns, random_state=7)
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------------------
# nonlinearity_gap
# --------------------------------------------------------------------------------------


def test_nonlinearity_gap_masks_diagonal_and_computes_expected_values() -> None:
    labels = ["A", "B"]
    dcor = pd.DataFrame([[1.0, 0.8], [0.8, 1.0]], index=labels, columns=labels)
    pearson = pd.DataFrame([[1.0, -0.1], [-0.1, 1.0]], index=labels, columns=labels)
    gap = NL.nonlinearity_gap(dcor, pearson)
    assert np.isnan(gap.loc["A", "A"])
    assert gap.loc["A", "B"] == pytest.approx(0.8 - 0.1)


def test_nonlinearity_gap_rejects_mismatched_labels() -> None:
    a = pd.DataFrame([[1.0, 0.5]], columns=["A", "B"], index=["A"])
    b = pd.DataFrame([[1.0]], columns=["A"], index=["A"])
    with pytest.raises(NL.NonlinearError, match="identical labels"):
        NL.nonlinearity_gap(a, b)


# --------------------------------------------------------------------------------------
# Permutation test and FDR
# --------------------------------------------------------------------------------------


def test_permutation_test_rejects_a_strong_linear_relationship() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(200)
    y = x + 0.05 * rng.standard_normal(200)
    result = NL.distance_correlation_permutation_test(x, y, n_permutations=199, seed=0)
    assert result.p_value < 0.01


def test_permutation_test_is_seed_deterministic() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(150)
    y = rng.standard_normal(150)
    a = NL.distance_correlation_permutation_test(x, y, n_permutations=99, seed=3)
    b = NL.distance_correlation_permutation_test(x, y, n_permutations=99, seed=3)
    assert a == b


def test_permutation_test_p_value_bounds() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(100)
    y = rng.standard_normal(100)
    result = NL.distance_correlation_permutation_test(x, y, n_permutations=50, seed=1)
    assert 1.0 / 51.0 <= result.p_value <= 1.0


def test_benjamini_hochberg_qvalues_known_example() -> None:
    """Standard textbook example: 5 p-values, alpha=0.05 (Benjamini and Hochberg 1995)."""
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.5])
    q = NL.benjamini_hochberg_qvalues(p)
    # BH step-up: q_i = min over j>=i of (p_(j) * m / j), monotone non-decreasing from most
    # significant to least.
    expected = np.array([0.05, 0.05, 0.05, 0.05, 0.5])
    np.testing.assert_allclose(np.sort(q), expected, atol=1e-8)


def test_benjamini_hochberg_qvalues_is_monotone_in_sorted_order() -> None:
    rng = np.random.default_rng(SEED)
    p = rng.uniform(0, 1, size=50)
    q = NL.benjamini_hochberg_qvalues(p)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_benjamini_hochberg_qvalues_rejects_out_of_range() -> None:
    with pytest.raises(NL.NonlinearError, match=r"\[0, 1\]"):
        NL.benjamini_hochberg_qvalues(np.array([0.5, 1.5]))


def test_pairwise_distance_correlation_significance_shape_and_columns(factor_panel) -> None:
    returns = factor_panel.returns.iloc[:200, :4]
    table = NL.pairwise_distance_correlation_significance(returns, n_permutations=49, seed=0)
    n = returns.shape[1]
    assert len(table) == n * (n - 1) // 2
    assert set(table.columns) == {
        "asset_a",
        "asset_b",
        "distance_correlation",
        "p_value",
        "q_value",
        "reject",
    }
    assert (table["q_value"] >= table["p_value"] - 1e-9).all()
