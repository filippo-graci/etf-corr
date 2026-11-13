"""Unit tests for ``etfcorr.correlation.regime_factor`` (step 4.2b, exploratory).

Known answers first (a two-asset hand calculation, the OLS covariance identity, Shapley values of
toy value functions), then agreement with the 4.2 engine, then planted-truth recovery on simulated
one- and two-factor panels whose population answer is computed analytically in the test (including
the case the 4.2 per-asset Forbes-Rigobon adjustment under-corrects), then the bootstrap, the
latent-factor helper and input validation.
"""

import math

import numpy as np
import pandas as pd
import pytest

from etfcorr.correlation import regime_correlation as R
from etfcorr.correlation import regime_factor as F

SIZES = (3, 3, 3)
N_ASSETS = sum(SIZES)
COLS = [f"S{s}_{k}" for s, size in enumerate(SIZES) for k in range(size)]
SECTORS = {f"S{s}_{k}": f"sector{s}" for s, size in enumerate(SIZES) for k in range(size)}
SCALE = 0.01


def _label_path(n: int, n_regimes: int, rng: np.random.Generator) -> np.ndarray:
    out = np.empty(n)
    t, code = 0, int(rng.integers(0, n_regimes))
    while t < n:
        length = int(rng.geometric(1.0 / 40.0))
        out[t : t + length] = code
        t += length
        code = int((code + 1 + rng.integers(0, n_regimes - 1)) % n_regimes)
    return out


def _simulate(
    n: int,
    seed: int,
    mult: tuple[float, ...],
    *,
    load: tuple[float, ...] | None = None,
    idio: tuple[float, ...] | None = None,
    extra: tuple[float, ...] | None = None,
    b: float = 1.0,
    c: float = 0.9,
    s2: float = 1.5,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Returns, labels (n + 1 days) and the observable factors (market, and the second factor when
    ``extra`` is given).

    Day ``j`` (return date ``index[j + 1]``) is governed by regime ``labels[j]``, the label known at
    the close of the day before, so ``predetermined`` alignment recovers the true regime. In regime
    ``g`` the market factor has variance ``mult[g]`` (times ``SCALE**2``), every asset loads
    ``b * load[g]`` on it, the second factor (loading ``c`` on every asset) has variance
    ``extra[g]``, and the idiosyncratic variance is ``s2 * idio[g]``.
    """
    g_count = len(mult)
    load = load or (1.0,) * g_count
    idio = idio or (1.0,) * g_count
    rng = np.random.default_rng(seed)
    path = _label_path(n + 1, g_count, rng)
    gov = path[:-1].astype(int)
    m = np.array(mult)[gov]
    f1 = SCALE * np.sqrt(m) * rng.standard_normal(n)
    bl = b * np.array(load)[gov]
    eps = SCALE * np.sqrt(s2 * np.array(idio)[gov])[:, None] * rng.standard_normal((n, N_ASSETS))
    r = bl[:, None] * f1[:, None] + eps
    index = pd.bdate_range("2014-01-02", periods=n + 1)
    factors = pd.DataFrame({"mkt": f1}, index=index[1:])
    if extra is not None:
        f2 = SCALE * np.sqrt(np.array(extra)[gov]) * rng.standard_normal(n)
        r = r + c * f2[:, None]
        factors["second"] = f2
    returns = pd.DataFrame(r, index=index[1:], columns=COLS)
    return returns, pd.Series(path, index=index), factors


def _rho(b2m: float, other: float, s2i: float) -> float:
    """Population correlation when every pair shares ``b2m + other`` of covariance."""
    return (b2m + other) / (b2m + other + s2i)


# ---- the factor model and its identity -----------------------------------------------------


def test_ols_identity_holds_exactly() -> None:
    rng = np.random.default_rng(0)
    f = rng.standard_normal((500, 2))
    x = f @ rng.uniform(0.5, 1.5, (6, 2)).T + rng.standard_normal((500, 6))
    fit = F.fit_factor_model(x, f)
    cov = np.cov(x, rowvar=False)
    implied = fit.loadings @ fit.factor_cov @ fit.loadings.T + fit.resid_cov
    np.testing.assert_allclose(implied, cov, atol=1e-12)
    assert fit.n_obs == 500
    np.testing.assert_allclose(fit.resid_cov, fit.resid_cov.T, atol=1e-15)


def test_ols_loadings_recover_the_truth_and_residuals_are_orthogonal() -> None:
    rng = np.random.default_rng(1)
    f = rng.standard_normal((100_000, 2))
    true = np.array([[1.0, 0.5], [0.2, -0.7], [0.0, 1.1]])
    x = 3.0 + f @ true.T + 0.5 * rng.standard_normal((100_000, 3))
    fit = F.fit_factor_model(x, f)
    np.testing.assert_allclose(fit.loadings, true, atol=0.02)
    np.testing.assert_allclose(np.diag(fit.resid_cov), 0.25, atol=0.01)


def test_implied_correlation_known_answer() -> None:
    b = np.array([[1.0], [1.0]])
    resid = np.eye(2)
    np.testing.assert_allclose(F.implied_correlation(b, np.array([[1.0]]), resid)[0, 1], 0.5)
    np.testing.assert_allclose(F.implied_correlation(b, np.array([[4.0]]), resid)[0, 1], 0.8)
    corr = F.implied_correlation(b, np.array([[4.0]]), resid)
    np.testing.assert_allclose(np.diag(corr), 1.0)
    np.testing.assert_allclose(corr, corr.T)


def test_implied_correlation_rejects_a_non_positive_variance() -> None:
    with pytest.raises(R.H2Error, match="non-positive variance"):
        F.implied_correlation(np.zeros((2, 1)), np.array([[1.0]]), np.zeros((2, 2)))


# ---- the eight mixtures and the Shapley attribution -----------------------------------------


def _fits_for_regimes() -> tuple[F.FactorFit, F.FactorFit, np.ndarray, pd.DataFrame]:
    returns, labels, factors = _simulate(6000, 3, (1.0, 3.0), load=(1.0, 1.2), idio=(1.0, 1.3))
    aligned = R.align_regime_labels(labels, returns.index, "predetermined").to_numpy()
    x, f = returns.to_numpy(), factors.to_numpy()
    ref = F.fit_factor_model(x[aligned == 0], f[aligned == 0])
    other = F.fit_factor_model(x[aligned == 1], f[aligned == 1])
    mask = R.pair_mask(list(SECTORS.values()), "cross")
    return ref, other, mask, returns


def test_the_corner_extremes_are_the_two_regimes_themselves() -> None:
    returns, labels, factors = _simulate(6000, 3, (1.0, 3.0), load=(1.0, 1.2), idio=(1.0, 1.3))
    aligned = R.align_regime_labels(labels, returns.index, "predetermined").to_numpy()
    x, f = returns.to_numpy(), factors.to_numpy()
    mask = R.pair_mask(list(SECTORS.values()), "cross")
    ref = F.fit_factor_model(x[aligned == 0], f[aligned == 0])
    other = F.fit_factor_model(x[aligned == 1], f[aligned == 1])
    corner = F.corner_averages(ref, other, mask)
    assert len(corner) == 8
    np.testing.assert_allclose(
        corner[(0, 0, 0)], np.corrcoef(x[aligned == 0], rowvar=False)[mask].mean(), atol=1e-12
    )
    np.testing.assert_allclose(
        corner[(1, 1, 1)], np.corrcoef(x[aligned == 1], rowvar=False)[mask].mean(), atol=1e-12
    )


def test_shapley_values_sum_to_the_total_change() -> None:
    ref, other, mask, _ = _fits_for_regimes()
    corner = F.corner_averages(ref, other, mask)
    shap = F.shapley_attribution(corner)
    assert math.isclose(sum(shap), corner[(1, 1, 1)] - corner[(0, 0, 0)], abs_tol=1e-12)


def test_shapley_known_answers_on_toy_value_functions() -> None:
    def table(fn: object) -> dict[tuple[int, int, int], float]:
        import itertools

        return {k: fn(*k) for k in itertools.product((0, 1), repeat=3)}  # type: ignore[operator]

    additive = table(lambda x, y, z: 2.0 * x - 1.0 * y + 0.5 * z)
    np.testing.assert_allclose(F.shapley_attribution(additive), (2.0, -1.0, 0.5), atol=1e-14)
    interaction = table(lambda x, y, z: float(x * y))
    np.testing.assert_allclose(F.shapley_attribution(interaction), (0.5, 0.5, 0.0), atol=1e-14)
    three_way = table(lambda x, y, z: float(x * y * z))
    np.testing.assert_allclose(F.shapley_attribution(three_way), (1 / 3, 1 / 3, 1 / 3), atol=1e-14)


def test_a_pure_factor_variance_change_is_all_factor_variance() -> None:
    ref, other, mask, _ = _fits_for_regimes()
    only_var = F.FactorFit(
        loadings=ref.loadings,
        factor_cov=other.factor_cov,
        resid_cov=ref.resid_cov,
        n_obs=other.n_obs,
    )
    corner = F.corner_averages(ref, only_var, mask)
    shap = F.shapley_attribution(corner)
    total = corner[(1, 1, 1)] - corner[(0, 0, 0)]
    assert total > 0.0
    np.testing.assert_allclose(shap, (total, 0.0, 0.0), atol=1e-12)


# ---- agreement with the 4.2 engine -----------------------------------------------------------


@pytest.mark.parametrize("alignment", ["predetermined", "contemporaneous"])
@pytest.mark.parametrize("pairs", ["cross", "within", "all"])
def test_avg_raw_equals_the_42_engine(alignment: str, pairs: str) -> None:
    returns, labels, factors = _simulate(3000, 5, (1.0, 2.0, 4.0), load=(1.0, 1.0, 1.1))
    ours = F.regime_factor_decomposition(
        returns,
        labels,
        SECTORS,
        factors,
        alignment=alignment,
        pairs=pairs,  # type: ignore[arg-type]
    ).table
    theirs = R.regime_correlations(
        returns,
        labels,
        SECTORS,
        alignment=alignment,
        pairs=pairs,  # type: ignore[arg-type]
    ).table
    np.testing.assert_allclose(ours["avg_raw"], theirs["avg_raw"], atol=1e-12)
    assert list(ours["n_obs"]) == list(theirs["n_obs"])


def test_the_reference_row_is_the_identity() -> None:
    returns, labels, factors = _simulate(3000, 6, (1.0, 2.0, 4.0))
    t = F.regime_factor_decomposition(returns, labels, SECTORS, factors).table
    ref = t.loc[0]
    assert math.isclose(ref["avg_var_only"], ref["avg_raw"], abs_tol=1e-12)
    for col in ("shap_factor_variance", "shap_loadings", "shap_residual_covariance"):
        assert math.isclose(ref[col], 0.0, abs_tol=1e-12)
    for col in ("sys_var_ratio", "idio_var_ratio", "loading_ratio"):
        assert math.isclose(ref[col], 1.0, abs_tol=1e-12)


def test_shapley_columns_sum_to_the_raw_change_in_the_table() -> None:
    returns, labels, factors = _simulate(3000, 7, (1.0, 2.0, 4.0), load=(1.0, 0.9, 1.3))
    t = F.regime_factor_decomposition(returns, labels, SECTORS, factors).table
    shap = t[["shap_factor_variance", "shap_loadings", "shap_residual_covariance"]].sum(axis=1)
    np.testing.assert_allclose(shap, t["avg_raw"] - t.loc[0, "avg_raw"], atol=1e-12)


# ---- planted truth -------------------------------------------------------------------------


def _pure_variance_truth() -> tuple[tuple[float, ...], float, float, float]:
    mult = (1.0, 2.0, 5.0)
    b, s2 = 1.0, 1.5
    rho0 = _rho(b * b * mult[0], 0.0, s2)
    rho2 = _rho(b * b * mult[2], 0.0, s2)
    delta2 = (b * b * mult[2] + s2) / (b * b * mult[0] + s2) - 1.0
    fr2 = rho2 / math.sqrt(1.0 + delta2 * (1.0 - rho2**2))
    return mult, rho0, rho2, (rho2 - fr2) / (rho2 - rho0)


def test_a_pure_volatility_change_is_recovered_where_the_per_asset_adjustment_undercorrects() -> (
    None
):
    mult, _, rho2, fr_share_truth = _pure_variance_truth()
    assert fr_share_truth < 0.6  # the per-asset FR share is far below the true 1.0 here
    returns, labels, factors = _simulate(60_000, 11, mult)
    ours = F.regime_factor_decomposition(returns, labels, SECTORS, factors).table
    raw_change = ours.loc[2, "avg_raw"] - ours.loc[0, "avg_raw"]
    share = (ours.loc[2, "avg_var_only"] - ours.loc[0, "avg_raw"]) / raw_change
    assert math.isclose(ours.loc[2, "avg_raw"], rho2, abs_tol=0.01)
    assert math.isclose(share, 1.0, abs_tol=0.03)
    assert abs(ours.loc[2, "shap_loadings"]) < 0.01
    assert abs(ours.loc[2, "shap_residual_covariance"]) < 0.01
    assert math.isclose(ours.loc[2, "sys_var_ratio"], mult[2], rel_tol=0.05)
    assert math.isclose(ours.loc[2, "idio_var_ratio"], 1.0, abs_tol=0.05)
    # the 4.2 engine, on the same data, lands where the analytic formula says: well below 1
    fr = R.regime_correlations(returns, labels, SECTORS).table
    fr_share = (
        (fr.loc[2, "avg_raw"] - fr.loc[0, "avg_raw"])
        - (fr.loc[2, "avg_adjusted"] - fr.loc[0, "avg_adjusted"])
    ) / (fr.loc[2, "avg_raw"] - fr.loc[0, "avg_raw"])
    assert math.isclose(fr_share, fr_share_truth, abs_tol=0.03)
    assert share - fr_share > 0.4


def test_a_loading_change_is_not_called_volatility() -> None:
    mult, load = (1.0, 2.0), (1.0, 1.5)
    b, s2 = 1.0, 1.5
    rho0 = _rho(b * b * mult[0], 0.0, s2)
    rho_g = _rho((b * load[1]) ** 2 * mult[1], 0.0, s2)
    rho_cf = _rho(b * b * mult[1], 0.0, s2)
    truth_share = (rho_cf - rho0) / (rho_g - rho0)
    returns, labels, factors = _simulate(60_000, 12, mult, load=load)
    t = F.regime_factor_decomposition(returns, labels, SECTORS, factors).table
    share = (t.loc[1, "avg_var_only"] - t.loc[0, "avg_raw"]) / (
        t.loc[1, "avg_raw"] - t.loc[0, "avg_raw"]
    )
    assert 0.0 < truth_share < 0.6
    assert math.isclose(share, truth_share, abs_tol=0.03)
    assert t.loc[1, "shap_loadings"] > 0.0
    assert math.isclose(t.loc[1, "loading_ratio"], load[1], rel_tol=0.03)


def test_a_rise_in_residual_variance_lowers_correlation_and_is_booked_as_residual() -> None:
    mult, idio = (1.0, 1.0), (1.0, 3.0)
    returns, labels, factors = _simulate(60_000, 13, mult, idio=idio)
    t = F.regime_factor_decomposition(returns, labels, SECTORS, factors).table
    assert t.loc[1, "avg_raw"] < t.loc[0, "avg_raw"]
    assert t.loc[1, "shap_residual_covariance"] < -0.05
    assert abs(t.loc[1, "shap_factor_variance"]) < 0.01
    assert math.isclose(t.loc[1, "idio_var_ratio"], 3.0, rel_tol=0.05)


def test_an_omitted_common_factor_is_booked_as_a_residual_change_until_it_is_supplied() -> None:
    """The documented limitation (module docstring): variance of a common factor that is not among
    the factors shows up as a change in residual covariance; supplying it restores the answer."""
    mult, extra = (1.0, 4.0), (1.0, 4.0)
    b, c, s2 = 1.0, 0.9, 1.5
    rho0 = _rho(b * b * mult[0], c * c * extra[0], s2)
    rho1 = _rho(b * b * mult[1], c * c * extra[1], s2)
    cf_market_only = _rho(b * b * mult[1], c * c * extra[0], s2)
    truth_k1 = (cf_market_only - rho0) / (rho1 - rho0)
    assert truth_k1 < 0.85
    returns, labels, factors = _simulate(60_000, 14, mult, extra=extra, b=b, c=c, s2=s2)

    k1 = F.regime_factor_decomposition(returns, labels, SECTORS, factors[["mkt"]]).table
    share_k1 = (k1.loc[1, "avg_var_only"] - k1.loc[0, "avg_raw"]) / (
        k1.loc[1, "avg_raw"] - k1.loc[0, "avg_raw"]
    )
    assert math.isclose(share_k1, truth_k1, abs_tol=0.03)
    assert k1.loc[1, "shap_residual_covariance"] > 0.02

    k2 = F.regime_factor_decomposition(returns, labels, SECTORS, factors).table
    share_k2 = (k2.loc[1, "avg_var_only"] - k2.loc[0, "avg_raw"]) / (
        k2.loc[1, "avg_raw"] - k2.loc[0, "avg_raw"]
    )
    assert math.isclose(share_k2, 1.0, abs_tol=0.03)


# ---- the latent-factor helper --------------------------------------------------------------


def test_latent_factors_recover_a_planted_second_factor() -> None:
    returns, _, factors = _simulate(8000, 15, (1.0, 1.0), extra=(1.0, 1.0))
    latent = F.latent_factors(returns, factors[["mkt"]], 2)
    assert list(latent.columns) == ["PC1", "PC2"]
    assert latent.index.equals(returns.index)
    assert np.corrcoef(latent["PC1"], factors["second"])[0, 1] > 0.9
    assert abs(np.corrcoef(latent["PC1"], latent["PC2"])[0, 1]) < 1e-8
    assert abs(np.corrcoef(latent["PC1"], factors["mkt"])[0, 1]) < 0.1
    again = F.latent_factors(returns, factors[["mkt"]], 2)
    pd.testing.assert_frame_equal(latent, again)


def test_latent_factors_leave_incomplete_rows_as_nan() -> None:
    returns, _, factors = _simulate(1000, 16, (1.0, 1.0), extra=(1.0, 1.0))
    broken = returns.copy()
    broken.iloc[10, 3] = np.nan
    latent = F.latent_factors(broken, factors[["mkt"]], 1)
    assert latent["PC1"].isna().sum() == 1
    assert np.isnan(latent["PC1"].iloc[10])


def test_latent_factors_reject_bad_requests() -> None:
    returns, _, factors = _simulate(500, 17, (1.0, 1.0))
    with pytest.raises(R.H2Error, match="n_extra"):
        F.latent_factors(returns, factors, 0)
    with pytest.raises(R.H2Error, match="too few complete rows"):
        F.latent_factors(returns.iloc[:5], factors, 1)


def test_latent_factor_signs_are_fixed_by_the_mean_loading_rule() -> None:
    """Each component's loadings on the standardised residuals sum to a non-negative number, so the
    component co-moves positively with the cross-sectional mean of those residuals (the sign of an
    eigenvector is otherwise arbitrary and platform dependent)."""
    for seed in range(12):
        returns, _, factors = _simulate(1500, 100 + seed, (1.0, 1.0), extra=(1.0, 1.0))
        latent = F.latent_factors(returns, factors[["mkt"]], 2)
        x = returns.to_numpy()
        b = factors[["mkt"]].to_numpy()
        xc, bc = x - x.mean(axis=0), b - b.mean(axis=0)
        resid = xc - bc @ np.linalg.lstsq(bc, xc, rcond=None)[0]
        mean_z = (resid / resid.std(axis=0, ddof=1)).mean(axis=1)
        for col in latent.columns:
            assert np.cov(latent[col].to_numpy(), mean_z)[0, 1] >= -1e-12


def test_loading_ratio_is_the_ratio_of_mean_first_factor_loadings() -> None:
    returns, labels, factors = _simulate(4000, 18, (1.0, 3.0), extra=(1.0, 2.0), c=0.4)
    t = F.regime_factor_decomposition(returns, labels, SECTORS, factors).table
    aligned = R.align_regime_labels(labels, returns.index, "predetermined").to_numpy()
    x, f = returns.to_numpy(), factors.to_numpy()
    ref = F.fit_factor_model(x[aligned == 0], f[aligned == 0])
    oth = F.fit_factor_model(x[aligned == 1], f[aligned == 1])
    expected = oth.loadings[:, 0].mean() / ref.loadings[:, 0].mean()
    assert math.isclose(t.loc[1, "loading_ratio"], expected, abs_tol=1e-12)


# ---- bootstrap -----------------------------------------------------------------------------


def _boot(**kw: object) -> F.FactorBootstrapResult:
    returns, labels, factors = _simulate(1500, 21, (1.0, 2.0, 4.0), load=(1.0, 1.0, 1.1))
    args: dict[str, object] = {"n_bootstrap": 60, "seed": 3}
    args.update(kw)
    return F.bootstrap_regime_factor(returns, labels, SECTORS, factors, **args)  # type: ignore[arg-type]


def test_bootstrap_is_deterministic_and_the_point_matches() -> None:
    a, b = _boot(), _boot()
    pd.testing.assert_frame_equal(a.frame(), b.frame())
    point = F.regime_factor_decomposition(*_inputs()).table
    assert math.isclose(a.stat("avg_raw", 2).point, point.loc[2, "avg_raw"], abs_tol=1e-12)
    assert math.isclose(
        a.stat("var_only", 2).point,
        point.loc[2, "avg_var_only"] - point.loc[0, "avg_raw"],
        abs_tol=1e-12,
    )
    assert a.stat("avg_raw", 1).replicates.shape == (60,)
    assert _boot(seed=4).frame()["lower"].tolist() != a.frame()["lower"].tolist()


def _inputs() -> tuple[pd.DataFrame, pd.Series, dict[str, str], pd.DataFrame]:
    returns, labels, factors = _simulate(1500, 21, (1.0, 2.0, 4.0), load=(1.0, 1.0, 1.1))
    return returns, labels, SECTORS, factors


def test_every_replicate_satisfies_the_identities() -> None:
    res = _boot()
    for g in (1, 2):
        total = (
            res.stat("shap_factor_variance", g).replicates
            + res.stat("shap_loadings", g).replicates
            + res.stat("shap_residual_covariance", g).replicates
        )
        np.testing.assert_allclose(total, res.stat("diff_raw", g).replicates, atol=1e-12)
        d = res.stat("diff_raw", g).replicates
        share = res.stat("var_only_share", g).replicates
        ok = np.abs(d) > 1e-12
        np.testing.assert_allclose(share[ok], res.stat("var_only", g).replicates[ok] / d[ok])


def test_the_share_interval_is_reported_only_when_the_raw_change_excludes_zero() -> None:
    res = _boot(n_bootstrap=100)
    for g in (1, 2):
        defined = not math.isnan(res.stat("var_only_share", g).lower)
        assert defined == res.stat("diff_raw", g).excludes_zero
    assert res.stat("diff_raw", 2).excludes_zero  # a large planted change is detected
    assert not math.isnan(res.stat("var_only_share", 2).lower)


def test_a_null_labelling_leaves_the_share_interval_undefined() -> None:
    returns, _, factors = _simulate(1500, 22, (1.0, 1.0, 1.0))
    rng = np.random.default_rng(99)
    labels = pd.Series(
        rng.integers(0, 3, size=len(returns) + 1).astype(float),
        index=pd.bdate_range("2014-01-02", periods=len(returns) + 1),
    )
    res = F.bootstrap_regime_factor(returns, labels, SECTORS, factors, n_bootstrap=100, seed=1)
    for g in (1, 2):
        assert (not res.stat("diff_raw", g).excludes_zero) == math.isnan(
            res.stat("var_only_share", g).lower
        )


def test_factors_are_resampled_with_the_returns() -> None:
    """Returns are an exact multiple of the factor: if the factor were not resampled with the
    same dates the loadings in a replicate would collapse and the loading ratio would wander."""
    rng = np.random.default_rng(31)
    n = 1500
    index = pd.bdate_range("2014-01-02", periods=n + 1)
    path = _label_path(n + 1, 2, rng)
    f = pd.Series(
        SCALE * np.where(path[:-1] == 1, 3.0, 1.0) * rng.standard_normal(n), index=index[1:]
    )
    returns = pd.DataFrame(
        np.outer(f.to_numpy(), np.linspace(0.8, 1.2, N_ASSETS))
        + 1e-5 * rng.standard_normal((n, N_ASSETS)),
        index=index[1:],
        columns=COLS,
    )
    labels = pd.Series(path, index=index)
    res = F.bootstrap_regime_factor(returns, labels, SECTORS, f, n_bootstrap=40, seed=2)
    reps = res.stat("loading_ratio", 1).replicates
    assert np.all(np.abs(reps - 1.0) < 0.02)


def test_bootstrap_rejects_bad_settings_and_sparse_replicates() -> None:
    returns, labels, factors = _simulate(800, 23, (1.0, 2.0))
    with pytest.raises(R.H2Error, match="n_bootstrap"):
        F.bootstrap_regime_factor(returns, labels, SECTORS, factors, n_bootstrap=1)
    with pytest.raises(R.H2Error, match="alpha"):
        F.bootstrap_regime_factor(returns, labels, SECTORS, factors, n_bootstrap=5, alpha=1.5)
    sparse = labels.copy()
    sparse.iloc[:] = 0.0
    sparse.iloc[100:112] = 1.0  # 12 days of regime 1, below the 30-day floor
    with pytest.raises(R.H2Error, match="need >="):
        F.regime_factor_decomposition(returns, sparse, SECTORS, factors)


# ---- reporting -----------------------------------------------------------------------------


def test_summary_lines_report_the_settings_and_the_caveat() -> None:
    lines = F.factor_summary_lines(_boot())
    text = "\n".join(lines)
    assert "factors=mkt" in text and "alignment=predetermined" in text
    assert "volatility-only counterfactual" in text and "Shapley split" in text
    assert "not part of the pre-registered H2" in text
    assert "for the three ratios: excludes 1" in text
    assert "high minus low" in text and "mid minus low" in text
    with pytest.raises(R.H2Error, match="regime names"):
        F.factor_summary_lines(_boot(), regime_names=["a"])


# ---- inputs --------------------------------------------------------------------------------


def test_inputs_are_not_mutated() -> None:
    returns, labels, factors = _simulate(800, 24, (1.0, 2.0))
    snapshot = (returns.copy(), labels.copy(), factors.copy())
    F.regime_factor_decomposition(returns, labels, SECTORS, factors)
    pd.testing.assert_frame_equal(returns, snapshot[0])
    pd.testing.assert_series_equal(labels, snapshot[1])
    pd.testing.assert_frame_equal(factors, snapshot[2])


def test_a_series_factor_is_accepted() -> None:
    returns, labels, factors = _simulate(800, 25, (1.0, 2.0))
    a = F.regime_factor_decomposition(returns, labels, SECTORS, factors["mkt"])
    b = F.regime_factor_decomposition(returns, labels, SECTORS, factors[["mkt"]])
    pd.testing.assert_frame_equal(a.table, b.table)
    assert a.factor_names == ("mkt",)


def test_rows_with_missing_values_are_dropped_not_filled() -> None:
    returns, labels, factors = _simulate(800, 26, (1.0, 2.0))
    base = F.regime_factor_decomposition(returns, labels, SECTORS, factors)
    holes = returns.copy()
    holes.iloc[5, 2] = np.nan
    fhole = factors.copy()
    fhole.iloc[9, 0] = np.nan
    res = F.regime_factor_decomposition(holes, labels, SECTORS, fhole)
    assert res.n_rows == base.n_rows - 2


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        ("unsorted", "strictly increasing"),
        ("dup_col", "unique"),
        ("one_col", "at least two columns"),
        ("no_sector", "no sector given"),
        ("inf_return", r"\+-inf"),
        ("inf_factor", r"\+-inf"),
        ("singular", "singular"),
        ("bad_labels", "non-negative integer"),
        ("absent_regime", "do not occur"),
        ("bad_reference", "reference regime"),
    ],
)
def test_invalid_inputs_raise(mutate: str, match: str) -> None:
    returns, labels, factors = _simulate(600, 27, (1.0, 2.0))
    sectors = dict(SECTORS)
    kwargs: dict[str, object] = {}
    if mutate == "unsorted":
        returns = returns.iloc[::-1]
    elif mutate == "dup_col":
        returns = returns.copy()
        returns.columns = [COLS[0], *COLS[:-1]]
    elif mutate == "one_col":
        returns = returns[[COLS[0]]]
    elif mutate == "no_sector":
        sectors.pop(COLS[0])
    elif mutate == "inf_return":
        returns = returns.copy()
        returns.iloc[3, 0] = np.inf
    elif mutate == "inf_factor":
        factors = factors.copy()
        factors.iloc[3, 0] = np.inf
    elif mutate == "singular":
        factors = factors.copy()
        factors["twin"] = factors["mkt"]
    elif mutate == "bad_labels":
        labels = labels - 0.5
    elif mutate == "absent_regime":
        kwargs["regimes"] = [0, 7]
    elif mutate == "bad_reference":
        kwargs["reference"] = 5
    with pytest.raises(R.H2Error, match=match):
        F.regime_factor_decomposition(returns, labels, sectors, factors, **kwargs)  # type: ignore[arg-type]


def test_a_ratio_is_starred_when_its_interval_excludes_one_not_zero() -> None:
    """Planted: the market factor variance rises about 4x while the residual variance is flat, so
    the systematic-variance ratio's interval excludes 1 and the residual-variance ratio's does not
    (a ratio interval never contains 0, so a star against 0 would mean nothing)."""
    returns, labels, factors = _simulate(2500, 41, (1.0, 4.0))
    res = F.bootstrap_regime_factor(returns, labels, SECTORS, factors, n_bootstrap=80, seed=0)
    text = "\n".join(F.factor_summary_lines(res, regime_names=["low", "high"]))
    sys_line = next(x for x in text.splitlines() if "ratios vs reference" in x)
    sys_part = sys_line.split("systematic variance")[1].split(";")[0]
    idio_part = sys_line.split("residual variance")[1].split(";")[0]
    assert sys_part.strip().endswith("*")
    assert not idio_part.strip().endswith("*")
    assert res.stat("sys_var_ratio", 1).lower > 1.0
    assert res.stat("idio_var_ratio", 1).lower < 1.0 < res.stat("idio_var_ratio", 1).upper
