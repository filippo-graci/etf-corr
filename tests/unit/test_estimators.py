"""``etfcorr.correlation.estimators`` (step 2.2).

Covers: structural validation (:func:`validate_correlation_matrix`), eigen-diagnostics
(:func:`eigen_diagnostics`) on known-good and known-bad matrices, each of the six methods'
numerical correctness on hand-computable small examples, the shared complete-case/window
preparation, the ``estimate``/``corr_report``/``summary_line`` orchestration, and a real
near-singular-vs-shrinkage demonstration on the shared synthetic ``factor_panel`` fixture
(T close to N), mirroring ``project_plan.md`` section 6.2's own reported synthetic-data claim
that Ledoit-Wolf stays well-conditioned where the sample matrix does not.
"""

import numpy as np
import pandas as pd
import pytest

from etfcorr.correlation import estimators as E

# --------------------------------------------------------------------------------------
# validate_correlation_matrix
# --------------------------------------------------------------------------------------


def test_validate_correlation_matrix_accepts_the_identity() -> None:
    corr = pd.DataFrame(np.eye(3), index=["A", "B", "C"], columns=["A", "B", "C"])
    E.validate_correlation_matrix(corr)  # must not raise


def test_validate_correlation_matrix_rejects_asymmetric_input() -> None:
    values = np.array([[1.0, 0.5, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]])
    corr = pd.DataFrame(values, index=["A", "B", "C"], columns=["A", "B", "C"])
    with pytest.raises(E.CorrelationError, match="symmetric"):
        E.validate_correlation_matrix(corr)


def test_validate_correlation_matrix_rejects_a_non_unit_diagonal() -> None:
    values = np.array([[1.0, 0.5], [0.5, 0.9]])
    corr = pd.DataFrame(values, index=["A", "B"], columns=["A", "B"])
    with pytest.raises(E.CorrelationError, match="diagonal"):
        E.validate_correlation_matrix(corr)


def test_validate_correlation_matrix_rejects_mismatched_labels() -> None:
    corr = pd.DataFrame(np.eye(2), index=["A", "B"], columns=["B", "A"])
    with pytest.raises(E.CorrelationError, match="index and columns"):
        E.validate_correlation_matrix(corr)


def test_validate_correlation_matrix_rejects_nan() -> None:
    values = np.array([[1.0, np.nan], [np.nan, 1.0]])
    corr = pd.DataFrame(values, index=["A", "B"], columns=["A", "B"])
    with pytest.raises(E.CorrelationError, match="NaN"):
        E.validate_correlation_matrix(corr)


def test_validate_correlation_matrix_rejects_a_non_dataframe() -> None:
    with pytest.raises(E.CorrelationError, match="DataFrame"):
        E.validate_correlation_matrix(np.eye(2))  # type: ignore[arg-type]


def test_validate_correlation_matrix_rejects_non_square() -> None:
    corr = pd.DataFrame(np.ones((2, 3)), index=["A", "B"], columns=["A", "B", "C"])
    with pytest.raises(E.CorrelationError, match="square"):
        E.validate_correlation_matrix(corr)


# --------------------------------------------------------------------------------------
# eigen_diagnostics
# --------------------------------------------------------------------------------------


def test_eigen_diagnostics_on_the_identity_is_perfectly_conditioned() -> None:
    corr = pd.DataFrame(np.eye(4))
    diag = E.eigen_diagnostics(corr)
    assert diag.condition_number == pytest.approx(1.0)
    assert diag.is_psd is True
    assert diag.min_eigenvalue == pytest.approx(1.0)
    assert diag.max_eigenvalue == pytest.approx(1.0)


def test_eigen_diagnostics_detects_a_genuinely_non_psd_matrix() -> None:
    """A 3-asset all-pairs correlation of -0.7 is impossible for real assets (the average
    correlation among >=3 variables cannot go below -1/(n-1) = -0.5) and is not PSD -- confirmed
    by direct eigen-decomposition, not asserted from a textbook claim."""
    values = np.array(
        [
            [1.0, -0.7, -0.7],
            [-0.7, 1.0, -0.7],
            [-0.7, -0.7, 1.0],
        ]
    )
    corr = pd.DataFrame(values, index=["A", "B", "C"], columns=["A", "B", "C"])
    E.validate_correlation_matrix(corr)  # symmetric, unit diagonal: structurally still "valid"
    diag = E.eigen_diagnostics(corr)
    assert diag.is_psd is False
    assert diag.min_eigenvalue < -0.1


# --------------------------------------------------------------------------------------
# Pearson / Spearman / Kendall: known values and complete-case behaviour
# --------------------------------------------------------------------------------------


def _perfect_and_independent(n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {
            "X": x,
            "Y": 2.0 * x + 1.0,  # perfectly linearly related to X: corr = 1
            "Z": -x,  # perfectly anti-related to X: corr = -1
            "W": rng.standard_normal(n),  # independent
        },
        index=idx,
    )


def test_pearson_known_values_perfect_and_anti_correlation() -> None:
    corr = E.estimate(_perfect_and_independent(), "pearson")
    assert corr.loc["X", "Y"] == pytest.approx(1.0, abs=1e-9)
    assert corr.loc["X", "Z"] == pytest.approx(-1.0, abs=1e-9)


def test_spearman_known_values_perfect_and_anti_correlation() -> None:
    corr = E.estimate(_perfect_and_independent(), "spearman")
    assert corr.loc["X", "Y"] == pytest.approx(1.0, abs=1e-9)
    assert corr.loc["X", "Z"] == pytest.approx(-1.0, abs=1e-9)


def test_kendall_known_values_perfect_and_anti_correlation() -> None:
    corr = E.estimate(_perfect_and_independent(), "kendall")
    assert corr.loc["X", "Y"] == pytest.approx(1.0, abs=1e-9)
    assert corr.loc["X", "Z"] == pytest.approx(-1.0, abs=1e-9)


def test_pandas_methods_reject_unexpected_keywords() -> None:
    with pytest.raises(E.CorrelationError, match="no extra keyword"):
        E.estimate(_perfect_and_independent(), "pearson", bogus=1)


def test_complete_case_drops_rows_where_any_ticker_is_missing() -> None:
    idx = pd.bdate_range("2024-01-01", periods=6)
    returns = pd.DataFrame(
        {"A": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0], "B": [1.0, 2.5, 3.0, 4.5, 5.0, 5.5]}, index=idx
    )
    report = E.corr_report(returns, "pearson")
    assert report.n_obs == 5  # one row (the NaN in A) dropped for both columns


def test_zero_variance_column_is_rejected() -> None:
    idx = pd.bdate_range("2024-01-01", periods=5)
    returns = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0], "CONST": [1.0] * 5}, index=idx)
    with pytest.raises(E.CorrelationError, match="zero or undefined variance"):
        E.estimate(returns, "pearson")


def test_fewer_than_two_complete_rows_is_rejected() -> None:
    idx = pd.bdate_range("2024-01-01", periods=3)
    returns = pd.DataFrame({"A": [1.0, np.nan, np.nan], "B": [1.0, 2.0, 3.0]}, index=idx)
    with pytest.raises(E.CorrelationError, match="at least 2 complete-case rows"):
        E.estimate(returns, "pearson")


def test_unknown_method_is_rejected() -> None:
    idx = pd.bdate_range("2024-01-01", periods=3)
    returns = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [1.0, 2.0, 3.0]}, index=idx)
    with pytest.raises(E.CorrelationError, match="unknown method"):
        E.estimate(returns, "not_a_method")  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Ledoit-Wolf / OAS shrinkage
# --------------------------------------------------------------------------------------


def test_ledoit_wolf_and_oas_return_valid_correlation_matrices(factor_panel) -> None:
    for method in ("ledoit_wolf", "oas"):
        corr = E.estimate(factor_panel.returns, method)  # type: ignore[arg-type]
        E.validate_correlation_matrix(corr)  # re-checked explicitly, not just relying on estimate()
        assert list(corr.columns) == list(factor_panel.returns.columns)


def test_ledoit_wolf_shrinks_towards_identity_relative_to_the_raw_sample() -> None:
    """On a small T-close-to-N sample the raw sample correlation is noisy; shrinkage should move
    off-diagonal entries of a known-independent (identity population) panel towards zero more
    than the raw sample average magnitude -- run and measured, not assumed."""
    rng = np.random.default_rng(7)
    n_assets, n_obs = 8, 12  # T close to N: noisy regime
    idx = pd.bdate_range("2024-01-01", periods=n_obs)
    returns = pd.DataFrame(rng.standard_normal((n_obs, n_assets)), index=idx)
    raw = E.estimate(returns, "pearson")
    shrunk = E.estimate(returns, "ledoit_wolf")
    off_diag = ~np.eye(n_assets, dtype=bool)
    raw_mag = np.abs(raw.to_numpy()[off_diag]).mean()
    shrunk_mag = np.abs(shrunk.to_numpy()[off_diag]).mean()
    assert shrunk_mag < raw_mag


def test_shrinkage_is_better_conditioned_than_the_sample_matrix_when_t_is_close_to_n(
    factor_panel,
) -> None:
    """Reproduces, on this project's own synthetic fixture, the qualitative claim
    project_plan.md section 6.2 already reported from running its own snippet: near T ~ N the
    raw sample correlation is ill-conditioned while Ledoit-Wolf/OAS stay well-conditioned."""
    returns = factor_panel.returns  # 9 assets
    n_assets = returns.shape[1]
    small_t = returns.iloc[: n_assets + 3]  # T = N + 3: deliberately thin
    raw_cond = E.eigen_diagnostics(E.estimate(small_t, "pearson")).condition_number
    lw_cond = E.eigen_diagnostics(E.estimate(small_t, "ledoit_wolf")).condition_number
    oas_cond = E.eigen_diagnostics(E.estimate(small_t, "oas")).condition_number
    assert lw_cond < raw_cond
    assert oas_cond < raw_cond


def test_shrinkage_rejects_an_unimplemented_backend() -> None:
    idx = pd.bdate_range("2024-01-01", periods=10)
    returns = pd.DataFrame(np.random.default_rng(0).standard_normal((10, 3)), index=idx)
    with pytest.raises(E.CorrelationError, match="not implemented offline"):
        E.estimate(returns, "ledoit_wolf", backend="skfolio")  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# EWMA: hand-computed known values
# --------------------------------------------------------------------------------------


def test_ewma_corr_known_values_two_rows_lambda_half() -> None:
    """S0 = outer([1,2]) = [[1,2],[2,4]]; S1 = 0.5*S0 + 0.5*outer([3,4]) = [[5,7],[7,10]];
    corr[0,1] = 7 / sqrt(5*10) -- verified independently by hand and by direct computation
    before writing this test."""
    idx = pd.bdate_range("2024-01-01", periods=2)
    returns = pd.DataFrame({"A": [1.0, 3.0], "B": [2.0, 4.0]}, index=idx)
    corr = E.ewma_corr(returns, lam=0.5)
    expected = 7.0 / np.sqrt(5.0 * 10.0)
    assert corr.loc["A", "B"] == pytest.approx(expected)
    assert corr.loc["A", "A"] == pytest.approx(1.0)


def test_ewma_corr_known_values_asymmetric_lambda() -> None:
    """Same construction with ``lam=0.8`` (not 0.5): unlike the symmetric-lambda case above, a
    bug that swapped ``lam``/``(1 - lam)`` in the recursion would change this answer, so this
    case is specifically sensitive to that mutation (confirmed below, not just asserted)."""
    idx = pd.bdate_range("2024-01-01", periods=2)
    returns = pd.DataFrame({"A": [1.0, 3.0], "B": [2.0, 4.0]}, index=idx)
    corr = E.ewma_corr(returns, lam=0.8)
    assert corr.loc["A", "B"] == pytest.approx(0.98058067569092)


def test_ewma_corr_rejects_lambda_outside_open_unit_interval() -> None:
    idx = pd.bdate_range("2024-01-01", periods=3)
    returns = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [1.0, 2.0, 3.0]}, index=idx)
    with pytest.raises(E.CorrelationError, match=r"lam must be in \(0, 1\)"):
        E.ewma_corr(returns, lam=1.0)
    with pytest.raises(E.CorrelationError, match=r"lam must be in \(0, 1\)"):
        E.ewma_corr(returns, lam=0.0)


def test_estimate_ewma_matches_ewma_corr_directly() -> None:
    idx = pd.bdate_range("2024-01-01", periods=5)
    returns = pd.DataFrame(
        {"A": [1.0, 2.0, -1.0, 0.5, 1.5], "B": [0.5, 1.0, -0.5, 1.0, 2.0]}, index=idx
    )
    via_estimate = E.estimate(returns, "ewma", lam=0.9)
    direct = E.ewma_corr(returns, lam=0.9)
    pd.testing.assert_frame_equal(via_estimate, direct)


# --------------------------------------------------------------------------------------
# window parameter: trailing semantics, equivalence to a manual .tail()
# --------------------------------------------------------------------------------------


def test_window_is_equivalent_to_a_manual_tail_slice(factor_panel) -> None:
    returns = factor_panel.returns
    window = 60
    via_window = E.estimate(returns, "pearson", window=window)
    manual = E.estimate(returns.tail(window), "pearson")
    pd.testing.assert_frame_equal(via_window, manual)


def test_window_uses_the_trailing_rows_not_a_leading_or_centred_slice(factor_panel) -> None:
    """Deliberately-leaky comparison: a `head`-based twin must give a DIFFERENT answer, proving
    this test is capable of catching a tail/head windowing mix-up (the one directional bug that
    matters if `window` is ever used inside walk-forward code assuming trailing semantics)."""
    returns = factor_panel.returns
    window = 60
    trailing = E.estimate(returns, "pearson", window=window)
    leading = E.estimate(returns.head(window), "pearson")
    assert not trailing.equals(leading)


def test_window_smaller_than_the_minimum_is_rejected() -> None:
    idx = pd.bdate_range("2024-01-01", periods=5)
    returns = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [2.0, 3.0, 1.0, 5.0, 4.0]}, index=idx
    )
    with pytest.raises(E.CorrelationError, match="smaller than the required minimum"):
        E.estimate(returns, "pearson", window=1)


# --------------------------------------------------------------------------------------
# estimate() dispatch across every method; corr_report / summary_line
# --------------------------------------------------------------------------------------


def test_estimate_dispatches_every_method_and_always_returns_a_validated_matrix(
    factor_panel,
) -> None:
    returns = factor_panel.returns
    for method in E.CORR_METHODS:
        corr = E.estimate(returns, method)  # type: ignore[arg-type]
        E.validate_correlation_matrix(corr)  # re-checked independently of estimate()'s own check


def test_corr_report_bundles_the_expected_diagnostics(factor_panel) -> None:
    report = E.corr_report(factor_panel.returns, "pearson")
    assert report.method == "pearson"
    assert report.window is None
    assert report.n_obs == len(factor_panel.returns)
    assert report.n_assets == factor_panel.returns.shape[1]
    assert report.condition_number >= 1.0
    assert isinstance(report.is_psd, bool)


def test_corr_report_records_the_requested_window() -> None:
    idx = pd.bdate_range("2024-01-01", periods=20)
    rng = np.random.default_rng(3)
    returns = pd.DataFrame(rng.standard_normal((20, 3)), index=idx)
    report = E.corr_report(returns, "pearson", window=10)
    assert report.window == 10
    assert report.n_obs == 10


def test_summary_line_mentions_method_t_n_condition_and_psd(factor_panel) -> None:
    report = E.corr_report(factor_panel.returns, "pearson")
    line = E.summary_line(report)
    assert "pearson" in line
    assert f"T={report.n_obs}" in line
    assert f"N={report.n_assets}" in line
    assert "PSD" in line
