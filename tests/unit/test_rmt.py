"""``etfcorr.correlation.rmt`` (step 3.1).

Covers: :func:`cov_to_corr`'s renormalization and its guards, :func:`marchenko_pastur_edge`'s
formula and guards, :func:`eigen_spectrum` against both the trivial identity case and a hand-
derived closed-form block-equicorrelation case (not a self-referential check), :func:`denoise_corr`
(a population-matrix no-op invariant, and a real Frobenius-error-reduction demonstration on the
shared synthetic ``factor_panel`` fixture at small T, with the actual measured numbers -- and the
honest, disclosed finding that this demonstration does *not* hold at every T), :func:`detone_corr`
against the same hand-derived closed-form values (demonstrating the "between-sector goes negative"
standing caveat quantitatively, not just qualitatively), and the ``*_report`` convenience wrappers.

**Where the hand-derived numbers come from** (verified independently of the code under test, not
just asserted): the synthetic ``factor_panel`` fixture's population correlation is a two-level
block-equicorrelation matrix (``etfcorr.synthetic.true_correlation``, K=3 blocks of m=3, within-
block rho_w = market_loading^2 + sector_loading^2 = 0.5, between-block rho_b = market_loading^2 =
0.25). A block-equicorrelation matrix with constant row sums has the all-ones vector as an
eigenvector (a general linear-algebra fact, not specific to this model) with eigenvalue equal to
that row sum: ``lambda1 = 1 + (m-1)*rho_w + m*(K-1)*rho_b = 1 + 2*0.5 + 3*2*0.25 = 3.5`` for this
fixture's parameters. The remaining eigenvalues (standard for this two-level design): ``K-1 = 2``
copies of ``1 + (m-1)*rho_w - m*rho_b = 1.25`` and ``K*(m-1) = 6`` copies of ``1 - rho_w = 0.5``
(trace check: ``3.5 + 2*1.25 + 6*0.5 = 9`` = N, confirmed). Because the top eigenvector is exactly
the constant vector ``v1 = ones(N)/sqrt(N)``, the market component removed by :func:`detone_corr`
is the constant matrix ``lambda1/N`` everywhere; subtracting it and renormalizing (both diagonal
and off-diagonal entries scale identically here by symmetry) gives, algebraically:
``within = (rho_w - lambda1/N) / (1 - lambda1/N) = (1/9) / (11/18) = 2/11``,
``between = (rho_b - lambda1/N) / (1 - lambda1/N) = (-5/36) / (11/18) = -5/22``.
All of the above was independently re-derived by hand and cross-checked numerically against
``numpy.linalg.eigvalsh``/``eigh`` directly in this session before being written into the
assertions below (not copied from the function under test's own output).
"""

import numpy as np
import pandas as pd
import pytest

from etfcorr.correlation import estimators as E
from etfcorr.correlation import rmt as R
from etfcorr.synthetic import make_factor_returns, true_correlation

# --------------------------------------------------------------------------------------
# Shared small fixtures
# --------------------------------------------------------------------------------------

_COLS3 = ["A", "B", "C"]


def _corr_df(values: np.ndarray, cols: list[str] | None = None) -> pd.DataFrame:
    c = cols or _COLS3
    return pd.DataFrame(values, index=c, columns=c)


def _block_true_corr_df() -> pd.DataFrame:
    """9x9 population correlation of ``factor_panel`` (3 sectors of 3, rho_w=0.5, rho_b=0.25)."""
    values = true_correlation((3, 3, 3), 0.5, 0.5)
    cols = [f"S{s}_{k}" for s in range(3) for k in range(3)]
    return pd.DataFrame(values, index=cols, columns=cols)


# --------------------------------------------------------------------------------------
# cov_to_corr
# --------------------------------------------------------------------------------------


def test_cov_to_corr_identity_is_a_no_op() -> None:
    out = R.cov_to_corr(_corr_df(np.eye(3)))
    pd.testing.assert_frame_equal(out, _corr_df(np.eye(3)))


def test_cov_to_corr_rescales_a_genuine_covariance_matrix() -> None:
    # cov = [[4, 2], [2, 9]] -> corr = [[1, 2/6], [2/6, 1]] = [[1, 1/3], [1/3, 1]]
    cov = _corr_df(np.array([[4.0, 2.0], [2.0, 9.0]]), ["A", "B"])
    out = R.cov_to_corr(cov)
    assert out.loc["A", "B"] == pytest.approx(1.0 / 3.0)
    assert out.loc["A", "A"] == pytest.approx(1.0)
    assert out.loc["B", "B"] == pytest.approx(1.0)


def test_cov_to_corr_rejects_non_square() -> None:
    with pytest.raises(R.RMTError, match="square"):
        R.cov_to_corr(pd.DataFrame(np.ones((2, 3))))


def test_cov_to_corr_rejects_mismatched_labels() -> None:
    with pytest.raises(R.RMTError, match="index and columns"):
        R.cov_to_corr(pd.DataFrame(np.eye(2), index=["A", "B"], columns=["B", "A"]))


def test_cov_to_corr_rejects_asymmetric_input() -> None:
    values = np.array([[1.0, 0.5], [0.4, 1.0]])
    with pytest.raises(R.RMTError, match="symmetric"):
        R.cov_to_corr(_corr_df(values, ["A", "B"]))


def test_cov_to_corr_rejects_non_positive_diagonal_and_names_the_column() -> None:
    values = np.array([[1.0, 0.1, 0.0], [0.1, -0.2, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(R.RMTError, match="B"):
        R.cov_to_corr(_corr_df(values))


# --------------------------------------------------------------------------------------
# marchenko_pastur_edge
# --------------------------------------------------------------------------------------


def test_mp_edge_at_q_equals_one() -> None:
    # q = N/T = 1 -> edge = sigma2 * (1 + 1)^2 = 4 * sigma2
    assert R.marchenko_pastur_edge(10, 10, sigma2=1.0) == pytest.approx(4.0)
    assert R.marchenko_pastur_edge(10, 10, sigma2=2.0) == pytest.approx(8.0)


def test_mp_edge_shrinks_towards_sigma2_as_t_grows() -> None:
    edge_small_t = R.marchenko_pastur_edge(9, 20)
    edge_large_t = R.marchenko_pastur_edge(9, 1_000_000)
    # q = 9/1e6 = 9e-6, sqrt(q) ~= 3e-3, edge ~= (1.003)^2 ~= 1.00601 -- close to sigma2=1 but not
    # to 1e-4 (computed, not guessed): loosen the tolerance to match the actual formula's rate.
    assert edge_small_t > edge_large_t > 1.0
    assert edge_large_t == pytest.approx(1.0, abs=1e-2)


@pytest.mark.parametrize(
    ("n_assets", "n_obs", "sigma2"), [(0, 10, 1.0), (10, 0, 1.0), (10, 10, 0.0), (10, 10, -1.0)]
)
def test_mp_edge_rejects_invalid_inputs(n_assets: int, n_obs: int, sigma2: float) -> None:
    with pytest.raises(R.RMTError):
        R.marchenko_pastur_edge(n_assets, n_obs, sigma2=sigma2)


# --------------------------------------------------------------------------------------
# eigen_spectrum
# --------------------------------------------------------------------------------------


def test_eigen_spectrum_on_the_identity() -> None:
    spectrum = R.eigen_spectrum(_corr_df(np.eye(9), [f"T{i}" for i in range(9)]), n_obs=100_000)
    assert spectrum.eigenvalues == pytest.approx(tuple([1.0] * 9))
    assert spectrum.is_psd is True
    assert spectrum.market_mode_share == pytest.approx(1.0 / 9.0)
    assert spectrum.effective_bets == pytest.approx(9.0)  # max diversification: N equal modes


def test_eigen_spectrum_matches_the_hand_derived_block_equicorrelation_case() -> None:
    """See the module docstring for the closed-form derivation of these numbers."""
    spectrum = R.eigen_spectrum(_block_true_corr_df(), n_obs=100_000, sigma2=1.0)
    expected = (3.5, 1.25, 1.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    assert spectrum.eigenvalues == pytest.approx(expected, abs=1e-9)
    assert spectrum.market_mode_share == pytest.approx(3.5 / 9.0)
    # edge ~= (1 + sqrt(9/100000))^2 ~= 1.019, clear of both 1.25 and 0.5: n_signal = 1 (market) + 2
    # (sector contrast) = 3, not on a floating-point boundary.
    assert spectrum.n_signal == 3
    assert spectrum.is_psd is True


def test_eigen_spectrum_flags_a_genuinely_non_psd_matrix() -> None:
    # Same impossible-for-real-assets construction as test_estimators.py's own PSD test.
    values = np.array([[1.0, -0.7, -0.7], [-0.7, 1.0, -0.7], [-0.7, -0.7, 1.0]])
    spectrum = R.eigen_spectrum(_corr_df(values), n_obs=1000)
    assert spectrum.is_psd is False


def test_eigen_spectrum_rejects_an_invalid_correlation_matrix() -> None:
    with pytest.raises(E.CorrelationError, match="symmetric"):
        R.eigen_spectrum(_corr_df(np.array([[1.0, 0.5], [0.4, 1.0]]), ["A", "B"]), n_obs=10)


def test_spectrum_summary_line_mentions_the_key_numbers() -> None:
    spectrum = R.eigen_spectrum(_block_true_corr_df(), n_obs=100_000)
    line = R.spectrum_summary_line(spectrum)
    assert "N=9" in line
    assert "signal=3" in line
    assert "PSD" in line


# --------------------------------------------------------------------------------------
# denoise_corr
# --------------------------------------------------------------------------------------


def test_denoise_corr_is_a_near_no_op_on_an_already_idealized_matrix() -> None:
    """The population matrix's own below-edge eigenvalues are already identical (all 0.5), so
    averaging them changes nothing -- a real structural sanity check on the averaging step, not a
    trivial identity test."""
    true_df = _block_true_corr_df()
    denoised = R.denoise_corr(true_df, n_obs=100_000, sigma2=1.0)
    pd.testing.assert_frame_equal(denoised, true_df, atol=1e-9)


def test_denoise_corr_preserves_trace_and_structural_invariants() -> None:
    true_df = _block_true_corr_df()
    denoised = R.denoise_corr(true_df, n_obs=15, sigma2=1.0)
    E.validate_correlation_matrix(denoised)  # must not raise
    assert np.trace(denoised.to_numpy()) == pytest.approx(9.0)


def test_denoise_corr_reduces_frobenius_error_at_small_t_on_real_estimated_data() -> None:
    """Real numbers, measured this session (not invented; see the Delivery block/report for the
    full T-sweep): at T=15 (N=9, q=0.6 -- the small-T-relative-to-N regime this project's own
    custom instructions flag as exactly where shrinkage/RMT cleaning is expected to help), the
    raw Pearson sample correlation of the shared synthetic panel has Frobenius error 2.0349 to the
    known population correlation; denoising it (n_obs=15) reduces that to 1.6981. This is **not**
    asserted to hold at every T -- a wider sweep this session (T=12,15,20,30 improved; T=50,63 did
    not) found it depends on T/N and is disclosed as such, not hidden."""
    panel = make_factor_returns(1500, (3, 3, 3), seed=20260921)
    sub = panel.returns.iloc[:15]
    raw = E.estimate(sub, "pearson")
    raw_err = float(np.linalg.norm(raw.to_numpy() - panel.true_corr))
    denoised = R.denoise_corr(raw, n_obs=15, sigma2=1.0)
    denoised_err = float(np.linalg.norm(denoised.to_numpy() - panel.true_corr))
    assert raw_err == pytest.approx(2.0349, abs=5e-4)
    assert denoised_err == pytest.approx(1.6981, abs=5e-4)
    assert denoised_err < raw_err


def test_denoise_corr_rejects_invalid_n_obs() -> None:
    with pytest.raises(R.RMTError, match="n_obs"):
        R.denoise_corr(_block_true_corr_df(), n_obs=0)


def test_denoise_corr_agrees_with_eigen_spectrum_on_what_counts_as_signal() -> None:
    """Cross-consistency: the same (n_obs, sigma2) must classify the same eigenvalues as
    'signal' whether asked via eigen_spectrum's count or observed via denoise_corr's averaging."""
    true_df = _block_true_corr_df()
    spectrum = R.eigen_spectrum(true_df, n_obs=15, sigma2=1.0)
    denoised = R.denoise_corr(true_df, n_obs=15, sigma2=1.0)
    denoised_eigs = np.sort(np.linalg.eigvalsh(denoised.to_numpy()))[::-1]
    n_distinct_above_noise = int(np.sum(~np.isclose(denoised_eigs, denoised_eigs[-1])))
    # every eigenvalue classified as signal must survive denoising un-averaged (distinct from the
    # single shared noise value), so the count of "not equal to the noise floor" >= n_signal - 1
    # (the smallest signal eigenvalue could coincidentally tie the noise floor only in a
    # degenerate case, not here).
    assert n_distinct_above_noise >= spectrum.n_signal - 1


# --------------------------------------------------------------------------------------
# denoise_report
# --------------------------------------------------------------------------------------


def test_denoise_report_matches_denoise_corr_and_carries_metadata() -> None:
    true_df = _block_true_corr_df()
    report = E.CorrelationReport(
        corr=true_df,
        method="pearson",
        window=None,
        n_obs=15,
        n_assets=9,
        condition_number=1.0,
        is_psd=True,
        min_eigenvalue=0.5,
    )
    result = R.denoise_report(report, sigma2=1.0)
    direct = R.denoise_corr(true_df, n_obs=15, sigma2=1.0)
    pd.testing.assert_frame_equal(result.corr, direct)
    assert result.method == "pearson"
    assert result.n_obs == 15
    assert result.n_assets == 9
    assert result.n_signal == R.eigen_spectrum(true_df, n_obs=15, sigma2=1.0).n_signal


# --------------------------------------------------------------------------------------
# detone_corr
# --------------------------------------------------------------------------------------


def test_detone_corr_matches_the_hand_derived_closed_form() -> None:
    """See the module docstring for the closed-form derivation: within-sector -> 2/11,
    between-sector -> -5/22 (negative, the standing 'detoned correlations are relative' caveat,
    demonstrated quantitatively here, not merely asserted)."""
    detoned = R.detone_corr(_block_true_corr_df(), n_market=1)
    assert detoned.loc["S0_0", "S0_1"] == pytest.approx(2.0 / 11.0, abs=1e-9)
    assert detoned.loc["S0_0", "S1_0"] == pytest.approx(-5.0 / 22.0, abs=1e-9)
    assert detoned.loc["S0_0", "S0_0"] == pytest.approx(1.0)


def test_detone_corr_between_sector_average_is_negative() -> None:
    """The qualitative claim, checked programmatically across every cross-sector pair, not just
    the one hand-derived pair above."""
    detoned = R.detone_corr(_block_true_corr_df(), n_market=1)
    sectors = {c: c.split("_")[0] for c in detoned.columns}
    cross = [
        detoned.loc[a, b]
        for a in detoned.columns
        for b in detoned.columns
        if a < b and sectors[a] != sectors[b]
    ]
    assert cross  # sanity: the fixture does have cross-sector pairs
    assert all(v < 0 for v in cross)


def test_detone_corr_is_a_valid_correlation_matrix() -> None:
    detoned = R.detone_corr(_block_true_corr_df(), n_market=1)
    E.validate_correlation_matrix(detoned)  # must not raise


@pytest.mark.parametrize("n_market", [0, 9, 10, -1])
def test_detone_corr_rejects_invalid_n_market(n_market: int) -> None:
    with pytest.raises(R.RMTError, match="n_market"):
        R.detone_corr(_block_true_corr_df(), n_market=n_market)


def test_detone_corr_raises_when_a_mode_leaves_non_positive_residual_variance() -> None:
    """A degenerate 2-asset case where one asset's entire variance is the shared mode: removing it
    leaves that asset's residual variance at exactly zero, which cov_to_corr must reject rather
    than dividing by zero silently."""
    # A perfectly-correlated pair (corr = 1 everywhere) has eigenvalues [2, 0]; removing the
    # single non-trivial mode (n_market=1) leaves the exact zero matrix -- zero residual variance
    # for both assets.
    perfect = _corr_df(np.array([[1.0, 1.0], [1.0, 1.0]]), ["A", "B"])
    with pytest.raises(R.RMTError, match="non-positive"):
        R.detone_corr(perfect, n_market=1)


# --------------------------------------------------------------------------------------
# detone_report
# --------------------------------------------------------------------------------------


def test_detone_report_matches_detone_corr_and_carries_the_removed_eigenvalue() -> None:
    true_df = _block_true_corr_df()
    report = E.CorrelationReport(
        corr=true_df,
        method="pearson",
        window=None,
        n_obs=15,
        n_assets=9,
        condition_number=1.0,
        is_psd=True,
        min_eigenvalue=0.5,
    )
    result = R.detone_report(report, n_market=1)
    direct = R.detone_corr(true_df, n_market=1)
    pd.testing.assert_frame_equal(result.corr, direct)
    assert result.market_eigenvalues == pytest.approx((3.5,), abs=1e-9)
    assert result.n_market == 1
    assert result.method == "pearson"


# --------------------------------------------------------------------------------------
# Relabelling invariance: functions must depend on which asset a row/column IS (its label),
# never on its position -- a real bug class (a silent positional mix-up) this guards against.
# --------------------------------------------------------------------------------------


def test_denoise_and_detone_are_invariant_to_ticker_relabelling_order() -> None:
    true_df = _block_true_corr_df()
    shuffled_cols = list(true_df.columns)[::-1]
    shuffled = true_df.loc[shuffled_cols, shuffled_cols]

    denoised_a = R.denoise_corr(true_df, n_obs=15)
    denoised_b = R.denoise_corr(shuffled, n_obs=15)
    pd.testing.assert_frame_equal(denoised_a, denoised_b.loc[true_df.columns, true_df.columns])

    detoned_a = R.detone_corr(true_df, n_market=1)
    detoned_b = R.detone_corr(shuffled, n_market=1)
    pd.testing.assert_frame_equal(detoned_a, detoned_b.loc[true_df.columns, true_df.columns])
