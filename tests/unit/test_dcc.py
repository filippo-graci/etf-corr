"""Unit tests for ``etfcorr.correlation.dcc`` (step 4.4).

Strategy: every vectorised piece is compared with a slow, explicit reference written in the test
(per-row loops, ``np.linalg.inv``/``det``); the likelihoods are compared with scipy's densities;
analytic cases (constant correlation, ``a = b = 0``, identity ``Qbar``) have closed-form answers;
the estimators are checked on simulated data with a planted truth, with tolerances set from the
sampling spread measured over 20 seeds (about five standard deviations, so a green run is not luck
and a biased estimator is still caught). The causal driver is checked piece by piece; the
no-lookahead properties live in ``tests/leakage/test_dcc_no_lookahead.py``.
"""

import dataclasses
from functools import cache

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from etfcorr.correlation import dcc
from etfcorr.correlation.dcc import DccError

RB2 = np.array([[1.0, 0.4], [0.4, 1.0]])
RB3 = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
FAST = {"dist": "normal", "min_obs": 150, "refit_every": 100}


@cache
def _sim(
    n_obs: int = 700,
    rb: str = "3",
    a: float = 0.05,
    b: float = 0.90,
    g: float = 0.0,
    nu: float | None = None,
    seed: int = 1,
) -> dcc.DccSimulation:
    rbar = RB3 if rb == "3" else RB2
    return dcc.simulate_dcc(n_obs, rbar, a=a, b=b, g=g, nu=nu, seed=seed)


# ---- references ----------------------------------------------------------------------


def _ref_garch_variance(e: np.ndarray, omega: float, alpha: float, beta: float, h0: float):
    h = np.empty(len(e))
    prev_h, prev_e2 = h0, h0
    for t in range(len(e)):
        h[t] = omega + alpha * prev_e2 + beta * prev_h
        prev_h, prev_e2 = h[t], e[t] ** 2
    return h


def _ref_q_path(z, qbar, nbar, a, b, g):
    q = [qbar.copy()]
    for t in range(len(z)):
        n = np.minimum(z[t], 0.0)
        q.append(
            (1 - a - b) * qbar - g * nbar + a * np.outer(z[t], z[t]) + b * q[-1]
            + g * np.outer(n, n)
        )  # fmt: skip
    return np.array(q)


def _ref_loglik(z, qbar, nbar, a, b, g):
    q = _ref_q_path(z, qbar, nbar, a, b, g)
    total = 0.0
    for t in range(len(z)):
        d = np.sqrt(np.diag(q[t]))
        r = q[t] / np.outer(d, d)
        total += -0.5 * (np.log(np.linalg.det(r)) + z[t] @ np.linalg.inv(r) @ z[t] - z[t] @ z[t])
    return total


def _moments(z):
    qbar = (z[:, :, None] * z[:, None, :]).mean(axis=0)
    n = np.minimum(z, 0.0)
    nbar = (n[:, :, None] * n[:, None, :]).mean(axis=0)
    return qbar, nbar


def _random_z(t=60, n=4, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((t, n)) @ np.linalg.cholesky(0.5 * np.eye(n) + 0.5 * np.ones((n, n))).T
    )


# ---- GARCH: variance filter and likelihood -------------------------------------------


def test_backcast_known_value() -> None:
    e = np.array([1.0, 2.0, 3.0])
    w = 0.94 ** np.arange(3)
    expected = float(np.sum(w / w.sum() * e**2))
    assert dcc.garch_backcast(e) == pytest.approx(expected, rel=1e-12)


def test_backcast_uses_only_the_first_75_rows() -> None:
    rng = np.random.default_rng(0)
    e = rng.standard_normal(300)
    changed = e.copy()
    changed[75:] *= 50.0
    assert dcc.garch_backcast(e) == dcc.garch_backcast(changed)
    assert dcc.garch_backcast(e) != dcc.garch_backcast(np.concatenate([e[:74], [9.0]]))


def test_backcast_rejects_all_zero_and_bad_input() -> None:
    with pytest.raises(DccError, match="zero"):
        dcc.garch_backcast(np.zeros(10))
    with pytest.raises(DccError, match="finite"):
        dcc.garch_backcast(np.array([1.0, np.nan]))


def test_garch_variance_matches_the_explicit_recursion() -> None:
    rng = np.random.default_rng(1)
    e = rng.standard_normal(200) * 0.01
    h = dcc.garch11_variance(e, 2e-6, 0.07, 0.9, 1e-4)
    np.testing.assert_allclose(h, _ref_garch_variance(e, 2e-6, 0.07, 0.9, 1e-4), rtol=1e-12)


def test_garch_variance_first_value_and_a_hand_computed_second() -> None:
    e = np.array([0.02, 0.01, 0.03])
    h = dcc.garch11_variance(e, 1e-5, 0.1, 0.8, 4e-4)
    assert h[0] == pytest.approx(1e-5 + 0.1 * 4e-4 + 0.8 * 4e-4)
    assert h[1] == pytest.approx(1e-5 + 0.1 * 0.02**2 + 0.8 * h[0])


def test_garch_variance_depends_only_on_earlier_residuals() -> None:
    rng = np.random.default_rng(2)
    e = rng.standard_normal(100) * 0.01
    full = dcc.garch11_variance(e, 2e-6, 0.07, 0.9, 1e-4)
    for k in (1, 10, 60):
        np.testing.assert_array_equal(dcc.garch11_variance(e[:k], 2e-6, 0.07, 0.9, 1e-4), full[:k])
    changed = e.copy()
    changed[50:] = 0.5
    np.testing.assert_array_equal(
        dcc.garch11_variance(changed, 2e-6, 0.07, 0.9, 1e-4)[:51], full[:51]
    )


def test_garch_loglik_normal_matches_scipy() -> None:
    rng = np.random.default_rng(3)
    e = rng.standard_normal(150) * 0.01
    h = dcc.garch11_variance(e, 2e-6, 0.07, 0.9, 1e-4)
    expected = float(stats.norm.logpdf(e, scale=np.sqrt(h)).sum())
    assert dcc.garch11_loglik(e, 2e-6, 0.07, 0.9, 1e-4, "normal") == pytest.approx(expected)


def test_garch_loglik_t_matches_scipy_unit_variance_t() -> None:
    rng = np.random.default_rng(4)
    e = rng.standard_normal(150) * 0.01
    nu = 6.0
    h = dcc.garch11_variance(e, 2e-6, 0.07, 0.9, 1e-4)
    s = np.sqrt((nu - 2.0) / nu)  # scale of a unit-variance t
    expected = float((stats.t.logpdf(e / (np.sqrt(h) * s), nu) - np.log(np.sqrt(h) * s)).sum())
    assert dcc.garch11_loglik(e, 2e-6, 0.07, 0.9, 1e-4, "t", nu) == pytest.approx(expected)


def test_garch_loglik_t_needs_nu_above_two() -> None:
    e = np.array([0.01, -0.02, 0.01])
    with pytest.raises(DccError, match="nu"):
        dcc.garch11_loglik(e, 1e-5, 0.1, 0.8, 1e-4, "t", None)
    with pytest.raises(DccError, match="nu"):
        dcc.garch11_loglik(e, 1e-5, 0.1, 0.8, 1e-4, "t", 2.0)


def test_garch_t_tends_to_normal_for_large_nu() -> None:
    rng = np.random.default_rng(5)
    e = rng.standard_normal(200) * 0.01
    normal = dcc.garch11_loglik(e, 2e-6, 0.07, 0.9, 1e-4, "normal")
    t_large = dcc.garch11_loglik(e, 2e-6, 0.07, 0.9, 1e-4, "t", 1e6)
    assert t_large == pytest.approx(normal, rel=1e-4)


# ---- GARCH: estimation ---------------------------------------------------------------


def test_garch_normal_fit_recovers_planted_parameters() -> None:
    sim = dcc.simulate_dcc(6000, RB2, omega=2e-6, alpha=0.08, beta=0.90, a=0.04, b=0.93, seed=3)
    fit = dcc.fit_garch11(sim.returns.to_numpy()[:, 0], dist="normal")
    assert fit.converged and fit.nu is None
    assert fit.alpha == pytest.approx(0.08, abs=0.025)
    assert fit.beta == pytest.approx(0.90, abs=0.04)
    assert fit.unconditional_variance == pytest.approx(2e-6 / 0.02, rel=0.5)
    assert fit.mu == pytest.approx(0.0, abs=5e-4)


def test_garch_t_fit_recovers_the_degrees_of_freedom() -> None:
    sim = dcc.simulate_dcc(6000, RB2, omega=2e-6, alpha=0.08, beta=0.90, nu=6.0, seed=3)
    fit = dcc.fit_garch11(sim.returns.to_numpy()[:, 0], dist="t")
    assert fit.nu is not None
    assert fit.nu == pytest.approx(6.0, abs=2.0)
    assert fit.alpha == pytest.approx(0.08, abs=0.04)
    assert fit.beta == pytest.approx(0.90, abs=0.05)


def test_garch_fit_is_at_least_as_good_as_the_true_parameters_in_sample() -> None:
    sim = dcc.simulate_dcc(3000, RB2, omega=2e-6, alpha=0.08, beta=0.90, seed=4)
    y = sim.returns.to_numpy()[:, 0]
    fit = dcc.fit_garch11(y, dist="normal")
    e = y - fit.mu
    at_truth = dcc.garch11_loglik(e, 2e-6, 0.08, 0.90, fit.h0, "normal")
    assert fit.loglik >= at_truth - 1e-6


def test_garch_fit_reported_loglik_is_the_likelihood_at_the_reported_parameters() -> None:
    y = _sim(1500).returns.to_numpy()[:, 1]
    fit = dcc.fit_garch11(y, dist="t")
    again = dcc.garch11_loglik(y - fit.mu, fit.omega, fit.alpha, fit.beta, fit.h0, "t", fit.nu)
    assert fit.loglik == pytest.approx(again, rel=1e-12)


def test_garch_fit_is_scale_equivariant() -> None:
    y = _sim(1500).returns.to_numpy()[:, 0]
    base = dcc.fit_garch11(y, dist="normal")
    scaled = dcc.fit_garch11(100.0 * y, dist="normal")
    assert scaled.alpha == pytest.approx(base.alpha, abs=2e-3)
    assert scaled.beta == pytest.approx(base.beta, abs=2e-3)
    assert scaled.omega == pytest.approx(base.omega * 1e4, rel=0.05)
    assert scaled.loglik == pytest.approx(base.loglik - len(y) * np.log(100.0), abs=0.5)


def test_garch_fit_is_deterministic_and_respects_constraints() -> None:
    y = _sim(1200).returns.to_numpy()[:, 2]
    first = dcc.fit_garch11(y, dist="t")
    second = dcc.fit_garch11(y, dist="t")
    assert first == second
    assert first.omega > 0 and first.alpha >= 0 and first.beta >= 0
    assert first.persistence < 1.0
    assert dcc.NU_BOUNDS[0] <= (first.nu or 0.0) <= dcc.NU_BOUNDS[1]


def test_garch_fit_demean_flag_and_mean() -> None:
    y = _sim(1000).returns.to_numpy()[:, 0] + 0.002
    with_mean = dcc.fit_garch11(y, dist="normal")
    without = dcc.fit_garch11(y, dist="normal", demean=False)
    assert with_mean.mu == pytest.approx(float(y.mean()))
    assert without.mu == 0.0


def test_garch_warm_start_reaches_the_same_optimum() -> None:
    y = _sim(1500).returns.to_numpy()[:, 0]
    cold = dcc.fit_garch11(y, dist="normal")
    warm = dcc.fit_garch11(y[:-100], dist="normal")
    refit = dcc.fit_garch11(y, dist="normal", start=warm)
    assert refit.loglik == pytest.approx(cold.loglik, abs=1e-3)
    assert refit.alpha == pytest.approx(cold.alpha, abs=5e-3)


def test_garch_on_homoskedastic_noise_has_a_small_alpha() -> None:
    rng = np.random.default_rng(6)
    fit = dcc.fit_garch11(rng.standard_normal(3000) * 0.01, dist="normal")
    assert fit.alpha < 0.1


def test_garch_input_validation() -> None:
    with pytest.raises(DccError, match="1-D"):
        dcc.fit_garch11(np.zeros((200, 2)))
    with pytest.raises(DccError, match="at least"):
        dcc.fit_garch11(np.ones(10))
    with pytest.raises(DccError, match="finite"):
        dcc.fit_garch11(np.r_[np.nan, np.ones(200)])
    with pytest.raises(DccError, match="constant"):
        dcc.fit_garch11(np.full(200, 0.01))
    with pytest.raises(DccError, match="dist"):
        dcc.fit_garch11(np.arange(200.0), dist="laplace")  # type: ignore[arg-type]


def test_standardized_residuals_are_unit_variance_and_causal() -> None:
    sim = _sim(3000, seed=5)
    y = sim.returns.to_numpy()[:, 0]
    fit = dcc.fit_garch11(y, dist="normal")
    z, h = dcc.standardized_residuals(y, fit)
    assert z.std() == pytest.approx(1.0, abs=0.06)
    assert abs(z.mean()) < 0.08
    z_cut, h_cut = dcc.standardized_residuals(y[:500], fit)
    np.testing.assert_array_equal(z_cut, z[:500])
    np.testing.assert_array_equal(h_cut, h[:500])


def test_garch_start_grid_puts_the_warm_start_first() -> None:
    warm = dcc.GarchFit(
        omega=4e-6, alpha=0.1, beta=0.8, nu=7.0, mu=0.0, h0=1e-4, loglik=0.0, n_obs=100,
        converged=True, dist="t",
    )  # fmt: skip
    grid = dcc._garch_start_grid(warm, 0.01)
    assert grid[0] == pytest.approx((4e-6 / 1e-4, 0.1, 0.8))
    assert len(grid) == len(dcc._garch_start_grid(None, 0.01)) + 1
    assert dcc._garch_start_grid(None, 0.01)[0] != grid[0]


def test_garch_with_a_single_start_follows_the_warm_start() -> None:
    """With one start allowed, a warm start at the optimum stays there and a bad one does not."""
    y = _sim(2000, seed=6).returns.to_numpy()[:, 0]
    best = dcc.fit_garch11(y, dist="normal")
    stay = dcc.fit_garch11(y, dist="normal", start=best, n_starts=1)
    assert stay.loglik == pytest.approx(best.loglik, abs=1e-6)
    bad = dataclasses.replace(best, alpha=0.5, beta=0.01, omega=best.omega * 50)
    assert dcc.fit_garch11(y, dist="normal", start=bad, n_starts=1).loglik <= best.loglik + 1e-9


# ---- DCC: recursion and likelihood ---------------------------------------------------


@pytest.mark.parametrize(("a", "b", "g"), [(0.04, 0.9, 0.0), (0.02, 0.93, 0.02), (0.0, 0.0, 0.0)])
def test_q_path_matches_the_explicit_recursion(a: float, b: float, g: float) -> None:
    z = _random_z()
    qbar, nbar = _moments(z)
    np.testing.assert_allclose(
        dcc.dcc_q_path(z, qbar, nbar, a, b, g), _ref_q_path(z, qbar, nbar, a, b, g), rtol=1e-11
    )


def test_q_path_with_a_and_b_zero_is_constant_qbar() -> None:
    z = _random_z()
    qbar, _ = _moments(z)
    q = dcc.dcc_q_path(z, qbar, None, 0.0, 0.0)
    assert q.shape == (len(z) + 1, 4, 4)
    for t in range(len(z) + 1):
        np.testing.assert_allclose(q[t], qbar, atol=1e-14)


def test_q_path_rows_are_causal() -> None:
    z = _random_z(80)
    qbar, nbar = _moments(z)
    full = dcc.dcc_q_path(z, qbar, nbar, 0.04, 0.9, 0.01)
    for k in (1, 20, 50):
        np.testing.assert_array_equal(
            dcc.dcc_q_path(z[:k], qbar, nbar, 0.04, 0.9, 0.01), full[: k + 1]
        )


def test_asymmetric_term_needs_nbar() -> None:
    z = _random_z()
    qbar, _ = _moments(z)
    with pytest.raises(DccError, match="nbar"):
        dcc.dcc_q_path(z, qbar, None, 0.04, 0.9, 0.02)


def test_correlations_are_valid_matrices() -> None:
    z = _random_z(300)
    qbar, nbar = _moments(z)
    r = dcc.dcc_correlations(dcc.dcc_q_path(z, qbar, nbar, 0.04, 0.9, 0.02))
    np.testing.assert_allclose(r, np.swapaxes(r, 1, 2), atol=1e-14)
    np.testing.assert_array_equal(np.diagonal(r, axis1=1, axis2=2), np.ones((301, 4)))
    assert np.abs(r).max() <= 1.0 + 1e-12
    assert np.linalg.eigvalsh(r).min() > 0.0


@pytest.mark.parametrize(("a", "b", "g"), [(0.04, 0.9, 0.0), (0.03, 0.9, 0.03), (0.0, 0.0, 0.0)])
def test_loglik_matches_the_explicit_loop(a: float, b: float, g: float) -> None:
    z = _random_z(50)
    qbar, nbar = _moments(z)
    assert dcc.dcc_loglik(z, qbar, nbar, a, b, g) == pytest.approx(
        _ref_loglik(z, qbar, nbar, a, b, g), rel=1e-10
    )


def test_loglik_is_zero_for_identity_qbar_and_no_dynamics() -> None:
    z = _random_z(40)
    assert dcc.dcc_loglik(z, np.eye(4), None, 0.0, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_loglik_constant_correlation_closed_form() -> None:
    z = _random_z(70)
    r = 0.5 * np.eye(4) + 0.5 * np.ones((4, 4))
    expected = -0.5 * (
        len(z) * np.log(np.linalg.det(r))
        + np.einsum("ti,ij,tj->", z, np.linalg.inv(r), z)
        - (z**2).sum()
    )
    assert dcc.dcc_loglik(z, r, None, 0.0, 0.0) == pytest.approx(expected)


def test_loglik_is_minus_inf_when_some_correlation_matrix_is_not_positive_definite() -> None:
    z = _random_z(40)
    qbar, nbar = _moments(z)
    assert dcc.dcc_loglik(z, qbar, nbar, 0.5, 0.9, 5.0) == -float("inf")


def test_gaussian_negative_part_moment_special_values_and_monte_carlo() -> None:
    v = dcc.gaussian_negative_part_moment(np.array([0.0, 1.0, -1.0]))
    np.testing.assert_allclose(v, [1 / (2 * np.pi), 0.5, 0.0], atol=1e-12)
    rng = np.random.default_rng(0)
    rho = 0.5
    x = rng.standard_normal((2_000_000, 2)) @ np.linalg.cholesky([[1, rho], [rho, 1]]).T
    mc = float((np.minimum(x[:, 0], 0) * np.minimum(x[:, 1], 0)).mean())
    assert float(dcc.gaussian_negative_part_moment(np.array(rho))) == pytest.approx(mc, abs=2e-3)


# ---- DCC: estimation -----------------------------------------------------------------


def _z_of(sim: dcc.DccSimulation) -> np.ndarray:
    """Truth-standardised residuals (isolates step two from the GARCH step)."""
    return sim.z


def test_fit_dcc_recovers_planted_parameters() -> None:
    sim = dcc.simulate_dcc(4000, RB3, a=0.05, b=0.90, seed=3)
    fit = dcc.fit_dcc(_z_of(sim))
    assert fit.converged and not fit.asymmetric and fit.g == 0.0
    assert fit.a == pytest.approx(0.05, abs=0.02)
    assert fit.b == pytest.approx(0.90, abs=0.04)
    assert fit.persistence < 1.0


def test_fit_dcc_optimum_is_not_beaten_by_a_grid() -> None:
    z = _z_of(dcc.simulate_dcc(1500, RB3, a=0.05, b=0.90, seed=5))
    fit = dcc.fit_dcc(z)
    qbar, nbar = fit.qbar, fit.nbar
    best = max(
        dcc.dcc_loglik(z, qbar, nbar, a, b)
        for a in np.arange(0.0, 0.2, 0.01)
        for b in np.arange(0.5, 0.99, 0.02)
        if a + b < 0.999
    )
    assert fit.loglik >= best - 1e-6


def test_fit_dcc_loglik_is_the_likelihood_at_the_reported_parameters() -> None:
    z = _z_of(dcc.simulate_dcc(800, RB3, a=0.05, b=0.90, seed=6))
    fit = dcc.fit_dcc(z)
    assert fit.loglik == pytest.approx(dcc.dcc_loglik(z, fit.qbar, fit.nbar, fit.a, fit.b, fit.g))
    assert fit.n_obs == 800


def test_fit_dcc_on_constant_correlation_finds_almost_no_dynamics() -> None:
    from etfcorr.synthetic import make_factor_returns

    panel = make_factor_returns(2000, (2, 2), seed=1)
    model = dcc.fit_dcc_model(panel.returns.to_numpy(), dist="normal")
    assert model.dcc.a < 0.05


def test_fit_dcc_qbar_and_nbar_are_the_sample_moments() -> None:
    z = _random_z(500)
    fit = dcc.fit_dcc(z)
    qbar, nbar = _moments(z)
    np.testing.assert_allclose(fit.qbar, qbar, atol=1e-13)
    np.testing.assert_allclose(fit.nbar, nbar, atol=1e-13)


def test_delta_is_the_generalised_eigenvalue() -> None:
    q = 0.5 * np.eye(3) + 0.5 * np.ones((3, 3))
    assert dcc._delta(q, 2.5 * q) == pytest.approx(2.5)
    z = _random_z(400)
    qbar, nbar = _moments(z)
    root = np.linalg.cholesky(qbar)
    inv = np.linalg.inv(root)
    assert dcc._delta(qbar, nbar) == pytest.approx(np.linalg.eigvalsh(inv @ nbar @ inv.T).max())


def test_asymmetric_fit_nests_the_symmetric_one() -> None:
    z = _z_of(dcc.simulate_dcc(1500, RB3, a=0.05, b=0.90, seed=7))
    sym = dcc.fit_dcc(z)
    asym = dcc.fit_dcc(z, asymmetric=True)
    assert asym.asymmetric
    assert asym.loglik >= sym.loglik - 1e-6
    # under a symmetric truth g is at most a few hundredths (20 seeds: mean 0.010, max 0.032)
    assert asym.g < 0.06
    assert asym.loglik - sym.loglik < 4.0


def test_asymmetric_fit_detects_a_planted_asymmetry() -> None:
    sim = dcc.simulate_dcc(5000, RB3, a=0.02, b=0.90, g=0.05, seed=8)
    z = _z_of(sim)
    sym = dcc.fit_dcc(z)
    asym = dcc.fit_dcc(z, asymmetric=True)
    assert asym.g > 0.02
    assert asym.loglik > sym.loglik + 2.0
    assert asym.a + asym.b + asym.delta * asym.g < 1.0


def test_stationarity_slack_known_values() -> None:
    assert dcc._stationarity_slack(0.03, 0.95, 0.04, 0.5) == pytest.approx(
        dcc.MAX_PERSISTENCE - 0.03 - 0.95 - 0.02
    )
    assert dcc._stationarity_slack(0.03, 0.95, 0.0, 9.0) == pytest.approx(
        dcc.MAX_PERSISTENCE - 0.98
    )
    assert dcc._stationarity_slack(0.05, 0.95, 0.01, 0.6) < 0.0


def test_asymmetric_fit_is_held_at_the_stationarity_boundary() -> None:
    """A near-integrated truth (sum 0.9985) pushes the unconstrained optimum past the boundary
    for this seed (a + b + delta g = 1.0009 without the g term); the constraint must bind."""
    z = _z_of(dcc.simulate_dcc(2500, RB3, a=0.05, b=0.92, g=0.05, seed=0))
    fit = dcc.fit_dcc(z, asymmetric=True)
    total = fit.a + fit.b + fit.delta * fit.g
    assert total < 1.0
    assert total > 0.9999
    assert fit.g > 0.0


def test_fit_dcc_keeps_the_stationarity_constraint() -> None:
    z = _z_of(dcc.simulate_dcc(1500, RB3, a=0.04, b=0.93, seed=9))
    for asymmetric in (False, True):
        fit = dcc.fit_dcc(z, asymmetric=asymmetric)
        assert fit.a >= 0 and fit.b >= 0 and fit.g >= 0
        assert fit.a + fit.b + fit.delta * fit.g < 1.0


def test_fit_dcc_warm_start_reaches_the_same_optimum() -> None:
    z = _z_of(dcc.simulate_dcc(1500, RB3, a=0.05, b=0.90, seed=10))
    cold = dcc.fit_dcc(z)
    warm = dcc.fit_dcc(z, start=dcc.fit_dcc(z[:-200]))
    assert warm.loglik == pytest.approx(cold.loglik, abs=1e-4)


def test_fit_dcc_input_validation() -> None:
    with pytest.raises(DccError, match="more rows than series"):
        dcc.fit_dcc(np.random.default_rng(0).standard_normal((3, 5)))
    with pytest.raises(DccError, match="finite"):
        dcc.fit_dcc(np.array([[1.0, np.nan], [0.5, 0.2], [0.1, 0.3]]))
    with pytest.raises(DccError, match="2-D"):
        dcc.fit_dcc(np.ones(5))
    z = _random_z(100, 3)
    z[:, 2] = z[:, 0]
    with pytest.raises(DccError, match="not positive definite"):
        dcc.fit_dcc(z)


def test_fit_dcc_does_not_modify_its_input() -> None:
    z = _random_z(300)
    copy = z.copy()
    dcc.fit_dcc(z, asymmetric=True)
    np.testing.assert_array_equal(z, copy)


# ---- full model, filter, forecast ----------------------------------------------------


def test_fit_dcc_model_and_filter_compose_the_steps() -> None:
    sim = _sim(900, seed=2)
    x = sim.returns.to_numpy()
    model = dcc.fit_dcc_model(x, list(sim.returns.columns), dist="normal")
    flt = dcc.filter_dcc(model, x)
    z = np.column_stack([dcc.standardized_residuals(x[:, i], model.garch[i])[0] for i in range(3)])
    np.testing.assert_array_equal(flt.z, z)
    d = model.dcc
    np.testing.assert_array_equal(flt.q, dcc.dcc_q_path(z, d.qbar, d.nbar, d.a, d.b, d.g))
    assert flt.corr.shape == (901, 3, 3) and flt.h.shape == (900, 3)
    assert model.columns == ("S0", "S1", "S2")


def test_filter_prefix_property_under_a_fixed_model() -> None:
    sim = _sim(900, seed=2)
    x = sim.returns.to_numpy()
    model = dcc.fit_dcc_model(x[:500], dist="normal")
    full = dcc.filter_dcc(model, x)
    for k in (50, 400, 700):
        part = dcc.filter_dcc(model, x[:k])
        np.testing.assert_array_equal(part.corr, full.corr[: k + 1])
        np.testing.assert_array_equal(part.z, full.z[:k])


def test_filter_last_row_is_the_one_step_forecast() -> None:
    sim = _sim(700, seed=3)
    x = sim.returns.to_numpy()
    model = dcc.fit_dcc_model(x[:400], dist="normal")
    flt = dcc.filter_dcc(model, x)
    np.testing.assert_allclose(flt.corr[-1], dcc.dcc_correlations(flt.q[-1:])[0])
    z_last = flt.z[-1]
    d = model.dcc
    expected_q = (1 - d.a - d.b) * d.qbar + d.a * np.outer(z_last, z_last) + d.b * flt.q[-2]
    np.testing.assert_allclose(flt.q[-1], expected_q, rtol=1e-10)


def test_fit_dcc_model_t_distribution_and_validation() -> None:
    sim = _sim(900, nu=6.0, seed=4)
    model = dcc.fit_dcc_model(sim.returns, dist="t")  # type: ignore[arg-type]
    assert all(g.nu is not None and g.dist == "t" for g in model.garch)
    with pytest.raises(DccError, match="column names"):
        dcc.fit_dcc_model(sim.returns.to_numpy(), ["a", "b"], dist="normal")
    with pytest.raises(DccError, match="dist"):
        dcc.fit_dcc_model(sim.returns.to_numpy(), dist="x")  # type: ignore[arg-type]
    with pytest.raises(DccError, match="columns"):
        dcc.filter_dcc(model, sim.returns.to_numpy()[:, :2])


def test_forecast_horizon_one_is_the_normalised_q() -> None:
    z = _random_z(100)
    qbar, nbar = _moments(z)
    q = dcc.dcc_q_path(z, qbar, nbar, 0.04, 0.9)[-1]
    np.testing.assert_allclose(
        dcc.forecast_correlation(q, qbar, 0.04, 0.9, 1), dcc.dcc_correlations(q[None])[0]
    )


def test_forecast_follows_the_geometric_reversion_formula() -> None:
    z = _random_z(100)
    qbar, _ = _moments(z)
    q = dcc.dcc_q_path(z, qbar, None, 0.04, 0.9)[-1]
    for h in (2, 5, 21):
        expected = dcc.dcc_correlations((qbar + 0.94 ** (h - 1) * (q - qbar))[None])[0]
        np.testing.assert_allclose(dcc.forecast_correlation(q, qbar, 0.04, 0.9, h), expected)


def test_forecast_converges_to_the_targeted_correlation() -> None:
    z = _random_z(100)
    qbar, _ = _moments(z)
    q = dcc.dcc_q_path(z, qbar, None, 0.04, 0.9)[-1]
    far = dcc.forecast_correlation(q, qbar, 0.04, 0.9, 5000)
    np.testing.assert_allclose(far, dcc.dcc_correlations(qbar[None])[0], atol=1e-12)


def test_forecast_accepts_a_stack_and_rejects_bad_horizons() -> None:
    z = _random_z(100)
    qbar, _ = _moments(z)
    q = dcc.dcc_q_path(z, qbar, None, 0.04, 0.9)[-10:]
    assert dcc.forecast_correlation(q, qbar, 0.04, 0.9, 7).shape == (10, 4, 4)
    with pytest.raises(DccError, match="horizon"):
        dcc.forecast_correlation(q, qbar, 0.04, 0.9, 0)


def test_average_correlation_known_values() -> None:
    r = np.array([[1.0, 0.2, 0.4], [0.2, 1.0, 0.6], [0.4, 0.6, 1.0]])
    assert dcc.average_correlation(r) == pytest.approx(0.4)
    mask = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool)
    assert dcc.average_correlation(r, mask) == pytest.approx(0.2)
    stack = np.stack([r, r * 0 + np.eye(3)])
    np.testing.assert_allclose(dcc.average_correlation(stack), [0.4, 0.0])


def test_average_correlation_validates_the_mask() -> None:
    r = np.eye(3)
    with pytest.raises(DccError, match="shape"):
        dcc.average_correlation(r, np.ones((2, 2), dtype=bool))
    with pytest.raises(DccError, match="symmetric"):
        dcc.average_correlation(r, np.ones((3, 3), dtype=bool))
    with pytest.raises(DccError, match="symmetric"):
        dcc.average_correlation(r, np.triu(~np.eye(3, dtype=bool)))
    with pytest.raises(DccError, match="no pair"):
        dcc.average_correlation(r, np.zeros((3, 3), dtype=bool))


def test_cross_sector_mask() -> None:
    mask = dcc.cross_sector_mask(["A", "B", "C"], {"A": "x", "B": "x", "C": "y"})
    expected = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=bool)
    np.testing.assert_array_equal(mask, expected)
    with pytest.raises(DccError, match="no sector"):
        dcc.cross_sector_mask(["A", "Z"], {"A": "x"})


# ---- causal driver -------------------------------------------------------------------


def _panel(n_obs: int = 520, seed: int = 11) -> pd.DataFrame:
    return _sim(n_obs, seed=seed).returns


def test_rolling_dcc_shape_warmup_and_columns() -> None:
    returns = _panel()
    path = dcc.rolling_dcc(returns, **FAST)
    assert path.avg_corr.shape == (520, 1) and list(path.avg_corr.columns) == ["h1"]
    assert path.avg_corr.index.equals(returns.index)
    assert path.avg_corr.iloc[:149].isna().all().all()
    assert path.avg_corr.iloc[149:].notna().all().all()
    assert path.corr_next is not None and path.corr_next.shape == (520, 3, 3)
    assert np.isnan(path.corr_next[:149]).all() and np.isfinite(path.corr_next[149:]).all()
    assert list(path.fits.index) == [returns.index[i] for i in (149, 249, 349, 449)]
    assert path.columns == tuple(returns.columns) and path.last_model is not None


def test_rolling_dcc_values_are_valid_correlations() -> None:
    path = dcc.rolling_dcc(_panel(), **FAST)
    assert path.corr_next is not None
    r = path.corr_next[149:]
    np.testing.assert_allclose(r, np.swapaxes(r, 1, 2), atol=1e-13)
    np.testing.assert_array_equal(np.diagonal(r, axis1=1, axis2=2), np.ones((371, 3)))
    off = r[:, ~np.eye(3, dtype=bool)]
    assert np.abs(off).max() < 1.0
    assert np.linalg.eigvalsh(r).min() > 0.0
    assert path.avg_corr["h1"].dropna().between(-1, 1).all()


def test_rolling_dcc_uses_the_refit_parameters_of_each_segment() -> None:
    """Re-derive one segment's output independently: fit on the training rows, filter forward."""
    returns = _panel()
    path = dcc.rolling_dcc(returns, warm_start=False, **FAST)
    x = returns.to_numpy()
    for end in (149, 249, 349):
        model = dcc.fit_dcc_model(x[: end + 1], list(returns.columns), dist="normal")
        flt = dcc.filter_dcc(model, x[: end + 50])
        assert path.corr_next is not None
        np.testing.assert_allclose(path.corr_next[end], flt.corr[end + 1], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(path.corr_next[end + 49], flt.corr[end + 50], rtol=1e-12)
        expected = dcc.average_correlation(flt.corr[end + 1])
        assert path.avg_corr["h1"].iloc[end] == pytest.approx(float(expected), rel=1e-12)


def test_rolling_dcc_fits_table() -> None:
    path = dcc.rolling_dcc(_panel(), **FAST)
    f = path.fits
    assert len(f) == 4 and f.index.name == "fit_end"
    assert list(f["n_train"]) == [150.0, 250.0, 350.0, 450.0]
    assert (f["a_plus_b"] == f["a"] + f["b"]).all()
    assert (f["g"] == 0.0).all() and f["nu_median"].isna().all()
    assert set(f.columns) >= {"a", "b", "g", "dcc_loglik", "dcc_converged", "garch_not_converged"}


def test_rolling_dcc_is_deterministic() -> None:
    a = dcc.rolling_dcc(_panel(), **FAST)
    b = dcc.rolling_dcc(_panel(), **FAST)
    pd.testing.assert_frame_equal(a.avg_corr, b.avg_corr)
    pd.testing.assert_frame_equal(a.fits, b.fits)


def test_rolling_dcc_does_not_modify_its_input() -> None:
    returns = _panel()
    copy = returns.copy()
    dcc.rolling_dcc(returns, **FAST)
    pd.testing.assert_frame_equal(returns, copy)


def test_rolling_dcc_leading_missing_rows_are_skipped() -> None:
    returns = _panel().copy()
    returns.iloc[:30, 1] = np.nan
    path = dcc.rolling_dcc(returns, **FAST)
    assert path.avg_corr.iloc[: 30 + 149].isna().all().all()
    assert path.avg_corr.iloc[30 + 149 :].notna().all().all()
    assert path.fits["n_train"].iloc[0] == 150.0
    clean = dcc.rolling_dcc(returns.iloc[30:], **FAST)
    np.testing.assert_allclose(
        path.avg_corr["h1"].iloc[30:].to_numpy(), clean.avg_corr["h1"].to_numpy(), equal_nan=True
    )


def test_rolling_dcc_rejects_an_interior_gap() -> None:
    returns = _panel().copy()
    returns.iloc[200, 0] = np.nan
    with pytest.raises(DccError, match="after the first complete row"):
        dcc.rolling_dcc(returns, **FAST)
    with pytest.raises(DccError, match="at least two"):
        dcc.rolling_dcc(_panel().iloc[:, :1], **FAST)


def test_rolling_dcc_returns_all_nan_when_history_is_shorter_than_min_obs() -> None:
    for returns in (_panel().iloc[:100], _panel() * np.nan, _panel().iloc[:149]):
        path = dcc.rolling_dcc(returns, **FAST)
        assert path.avg_corr.isna().all().all() and path.avg_corr.index.equals(returns.index)
        assert path.corr_next is not None and np.isnan(path.corr_next).all()
        assert len(path.fits) == 0 and path.last_model is None
    first_fit = dcc.rolling_dcc(_panel().iloc[:150], **FAST)
    assert first_fit.avg_corr.iloc[149:].notna().all().all() and len(first_fit.fits) == 1


def test_rolling_dcc_parameter_validation() -> None:
    returns = _panel()
    with pytest.raises(DccError, match="min_obs"):
        dcc.rolling_dcc(returns, dist="normal", min_obs=20, refit_every=50)
    with pytest.raises(DccError, match="refit_every"):
        dcc.rolling_dcc(returns, dist="normal", min_obs=150, refit_every=0)
    with pytest.raises(DccError, match="horizons"):
        dcc.rolling_dcc(returns, horizons=(0,), **FAST)
    with pytest.raises(DccError, match="symmetric"):
        dcc.rolling_dcc(returns, horizons=(1, 5), asymmetric=True, **FAST)
    with pytest.raises(DccError, match="shape"):
        dcc.rolling_dcc(returns, pair_mask=np.ones((2, 2), dtype=bool), **FAST)


def test_rolling_dcc_horizons_are_forecasts_toward_the_target() -> None:
    returns = _panel()
    path = dcc.rolling_dcc(returns, horizons=(1, 21, 2000), **FAST)
    assert list(path.avg_corr.columns) == ["h1", "h21", "h2000"]
    seg = path.avg_corr.iloc[149:249]  # the first refit's segment: one Qbar, one target
    far = seg["h2000"]
    assert far.std() < 1e-9  # 0.9^1999 is zero: the very long forecast is the constant target
    assert (seg["h21"] - far).abs().mean() < (seg["h1"] - far).abs().mean()
    assert (seg["h21"] - far).abs().mean() > 0.0
    solo = dcc.rolling_dcc(returns, **FAST)  # h = 1 of a multi-horizon call is the plain call
    pd.testing.assert_series_equal(path.avg_corr["h1"], solo.avg_corr["h1"])


def test_rolling_dcc_cross_sector_average() -> None:
    returns = _panel()
    mask = dcc.cross_sector_mask(list(returns.columns), {"S0": "a", "S1": "a", "S2": "b"})
    path = dcc.rolling_dcc(returns, pair_mask=mask, **FAST)
    assert path.corr_next is not None
    manual = (path.corr_next[:, 0, 2] + path.corr_next[:, 1, 2]) / 2.0
    np.testing.assert_allclose(path.avg_corr["h1"].to_numpy(), manual, equal_nan=True, rtol=1e-12)


def test_rolling_dcc_keep_matrices_false() -> None:
    path = dcc.rolling_dcc(_panel(), keep_matrices=False, **FAST)
    assert path.corr_next is None
    pd.testing.assert_frame_equal(path.avg_corr, dcc.rolling_dcc(_panel(), **FAST).avg_corr)


def test_rolling_dcc_asymmetric_runs_and_reports_g() -> None:
    path = dcc.rolling_dcc(_panel(), asymmetric=True, **FAST)
    assert (path.fits["g"] >= 0.0).all()
    assert path.avg_corr["h1"].iloc[149:].notna().all()


def test_rolling_dcc_cold_and_warm_starts_agree_closely() -> None:
    returns = _panel()
    warm = dcc.rolling_dcc(returns, **FAST).avg_corr["h1"]
    cold = dcc.rolling_dcc(returns, warm_start=False, **FAST).avg_corr["h1"]
    assert (warm - cold).abs().max() < 0.02


def test_rolling_dcc_t_distribution_runs() -> None:
    path = dcc.rolling_dcc(_panel(), dist="t", min_obs=150, refit_every=200)
    assert path.fits["nu_median"].notna().all()
    assert path.avg_corr["h1"].iloc[149:].notna().all()


def test_rolling_dcc_follows_a_planted_time_varying_correlation() -> None:
    sim = dcc.simulate_dcc(1500, RB3, a=0.12, b=0.80, seed=21)
    path = dcc.rolling_dcc(sim.returns, dist="normal", min_obs=400, refit_every=200)
    truth = pd.Series(dcc.average_correlation(sim.corr), index=sim.returns.index)
    est = path.avg_corr["h1"]
    # est[t] forecasts the day-(t+1) matrix; truth[t+1] is that matrix
    aligned = pd.concat([est, truth.shift(-1)], axis=1).dropna()
    assert aligned.corr().iloc[0, 1] > 0.8
    assert est.dropna().std() > 0.05  # the path is not flat


def test_rolling_dcc_on_constant_correlation_stays_nearly_flat() -> None:
    from etfcorr.synthetic import make_factor_returns

    panel = make_factor_returns(1200, (2, 2), seed=3)
    path = dcc.rolling_dcc(panel.returns, dist="normal", min_obs=400, refit_every=200)
    truth = float(dcc.average_correlation(panel.true_corr))
    est = path.avg_corr["h1"].dropna()
    assert abs(est.mean() - truth) < 0.1
    assert est.std() < 0.08


def test_summary_lines() -> None:
    path = dcc.rolling_dcc(_panel(), **FAST)
    lines = dcc.dcc_summary_lines(path)
    assert lines[0].startswith("DCC refits: 4, training rows 150 to 450")
    assert any(line.startswith("a min/median/max") for line in lines)
    assert not any(line.startswith("g ") for line in lines)
    asym = dcc.dcc_summary_lines(dcc.rolling_dcc(_panel(), asymmetric=True, **FAST))
    assert any(line.startswith("g min/median/max") for line in asym)
    t_lines = dcc.dcc_summary_lines(
        dcc.rolling_dcc(_panel(), dist="t", min_obs=150, refit_every=300)
    )
    assert any("Student-t nu" in line for line in t_lines)
    empty = dcc.rolling_dcc(_panel().iloc[:50], **FAST)
    assert dcc.dcc_summary_lines(empty) == ["no DCC fit (fewer complete rows than min_obs)"]


def test_summary_lines_report_the_failure_counts() -> None:
    fits = pd.DataFrame(
        {
            "n_train": [100.0, 200.0],
            "a": [0.03, 0.04], "b": [0.9, 0.91], "g": [0.0, 0.0], "a_plus_b": [0.93, 0.95],
            "dcc_loglik": [1.0, 2.0], "dcc_converged": [True, False],
            "garch_not_converged": [1.0, 2.0],
            "garch_persistence_median": [0.97, 0.98], "garch_persistence_max": [0.99, 0.995],
            "nu_median": [np.nan, np.nan],
        },
        index=pd.Index(pd.to_datetime(["2020-01-01", "2020-06-01"]), name="fit_end"),
    )  # fmt: skip
    path = dcc.DccPath(
        avg_corr=pd.DataFrame(), corr_next=None, fits=fits, last_model=None, columns=()
    )
    lines = dcc.dcc_summary_lines(path)
    assert lines[0] == "DCC refits: 2, training rows 100 to 200"
    assert lines[-1] == "not converged: DCC step in 1 refits, GARCH fits 3 (summed over refits)"
    assert "largest single persistence 0.9950" in lines[-2]


# ---- simulation helper ---------------------------------------------------------------


def test_simulation_is_deterministic_and_shaped() -> None:
    one = dcc.simulate_dcc(200, RB3, seed=7)
    two = dcc.simulate_dcc(200, RB3, seed=7)
    pd.testing.assert_frame_equal(one.returns, two.returns)
    assert (
        one.corr.shape == (200, 3, 3) and one.sigma2.shape == (200, 3) and one.z.shape == (200, 3)
    )
    assert one.returns.index.name == "date"


def test_simulation_truth_is_consistent_with_the_returns() -> None:
    sim = dcc.simulate_dcc(500, RB3, seed=8)
    np.testing.assert_allclose(sim.returns.to_numpy(), np.sqrt(sim.sigma2) * sim.z, rtol=1e-12)
    assert np.linalg.eigvalsh(sim.corr).min() > 0


def test_simulated_shocks_have_the_target_correlation_and_unit_variance() -> None:
    sim = dcc.simulate_dcc(20000, RB3, a=0.0, b=0.0, seed=9)
    np.testing.assert_allclose(np.cov(sim.z.T), RB3, atol=0.03)
    t_sim = dcc.simulate_dcc(20000, RB3, a=0.0, b=0.0, nu=6.0, seed=9)
    np.testing.assert_allclose(np.cov(t_sim.z.T), RB3, atol=0.08)
    assert stats.kurtosis(t_sim.z[:, 0]) > 1.5  # fat tails (excess kurtosis 6 / (nu - 4) = 3)


def test_simulation_rejects_invalid_parameters() -> None:
    with pytest.raises(DccError, match="alpha"):
        dcc.simulate_dcc(50, RB3, alpha=0.5, beta=0.6)
    with pytest.raises(DccError, match="a, b, g"):
        dcc.simulate_dcc(50, RB3, a=0.5, b=0.6)
    with pytest.raises(DccError, match="delta"):
        dcc.simulate_dcc(50, RB3, a=0.05, b=0.9, g=0.9)
    with pytest.raises(DccError, match="nu"):
        dcc.simulate_dcc(50, RB3, nu=1.5)
    with pytest.raises(DccError, match="correlation matrix"):
        dcc.simulate_dcc(50, np.array([[1.0, 0.5], [0.4, 1.0]]))
