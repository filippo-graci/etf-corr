"""Unit tests for ``etfcorr.regime_hmm`` (step 4.3): inference, EM, ordering, the causal driver and
the observation builder. Known answers come from brute-force enumeration of state paths and from
planted-truth simulations; nothing here is a result about ETFs. Leakage tests are in
``tests/leakage/test_regime_hmm_no_lookahead.py``."""

from itertools import product

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal

from etfcorr.regime_hmm import (  # private: affine mapping and the separation diagnostic
    HmmParams,
    RegimeHmmError,
    _from_z,
    _min_separation,
    _to_z,
    baum_welch,
    fit_gaussian_hmm,
    forward_filtered_hmm,
    forward_probabilities,
    hmm_observations,
    hmm_summary_lines,
    order_states,
    refit_positions,
    smoothed_probabilities,
)

# ---- fixtures ------------------------------------------------------------------------

TRUE_A = np.array([[0.97, 0.03], [0.04, 0.96]])
TRUE_MU = np.array([[0.0, 0.0], [2.5, 1.5]])
TRUE_COV = np.stack([0.5 * np.eye(2), np.array([[1.0, 0.4], [0.4, 0.8]])])


def _true_params() -> HmmParams:
    return HmmParams(np.array([0.5, 0.5]), TRUE_A, TRUE_MU, TRUE_COV)


def _simulate(
    n: int, seed: int = 0, params: HmmParams | None = None
) -> tuple[np.ndarray, np.ndarray]:
    p = _true_params() if params is None else params
    rng = np.random.default_rng(seed)
    k = p.n_states
    states = np.empty(n, dtype=int)
    states[0] = rng.choice(k, p=p.startprob)
    for t in range(1, n):
        states[t] = rng.choice(k, p=p.transmat[states[t - 1]])
    x = np.stack([rng.multivariate_normal(p.means[s], p.covars[s]) for s in states])
    return x, states


def _frame(x: np.ndarray, start: str = "2018-01-01") -> pd.DataFrame:
    index = pd.bdate_range(start, periods=len(x), name="date")
    return pd.DataFrame(x, index=index, columns=["log_vol", "z_cross_corr"])


def _brute_force(x: np.ndarray, params: HmmParams) -> tuple[np.ndarray, np.ndarray, float]:
    """Filtered, smoothed posteriors and log-likelihood by enumerating every state path."""
    n, k = len(x), params.n_states
    dens = np.stack(
        [multivariate_normal(params.means[j], params.covars[j]).pdf(x) for j in range(k)], axis=1
    ).reshape(n, k)

    def joint(path: tuple[int, ...]) -> float:
        p = params.startprob[path[0]] * dens[0, path[0]]
        for t in range(1, len(path)):
            p *= params.transmat[path[t - 1], path[t]] * dens[t, path[t]]
        return float(p)

    filtered = np.zeros((n, k))
    for t in range(n):
        for path in product(range(k), repeat=t + 1):
            filtered[t, path[-1]] += joint(path)
    filtered /= filtered.sum(axis=1, keepdims=True)
    smoothed = np.zeros((n, k))
    total = 0.0
    for path in product(range(k), repeat=n):
        w = joint(path)
        total += w
        for t, s in enumerate(path):
            smoothed[t, s] += w
    smoothed /= total
    return filtered, smoothed, float(np.log(total))


# ---- inference primitives ------------------------------------------------------------


def test_filter_smoother_and_loglik_match_brute_force_two_states() -> None:
    x, _ = _simulate(6, seed=3)
    f_true, s_true, ll_true = _brute_force(x, _true_params())
    f, ll = forward_probabilities(x, _true_params())
    s = smoothed_probabilities(x, _true_params())
    np.testing.assert_allclose(f, f_true, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(s, s_true, rtol=1e-10, atol=1e-12)
    assert ll == pytest.approx(ll_true, rel=1e-12)


def test_filter_smoother_and_loglik_match_brute_force_three_states_one_feature() -> None:
    params = HmmParams(
        startprob=np.array([0.2, 0.5, 0.3]),
        transmat=np.array([[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]]),
        means=np.array([[-1.0], [0.0], [2.0]]),
        covars=np.array([[[0.5]], [[1.0]], [[2.0]]]),
    )
    x = np.array([[0.3], [-0.8], [1.9], [2.4], [0.1]])
    f_true, s_true, ll_true = _brute_force(x, params)
    f, ll = forward_probabilities(x, params)
    s = smoothed_probabilities(x, params)
    np.testing.assert_allclose(f, f_true, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(s, s_true, rtol=1e-10, atol=1e-12)
    assert ll == pytest.approx(ll_true, rel=1e-12)


def test_filtered_rows_are_distributions_and_the_prefix_property_holds() -> None:
    x, _ = _simulate(400, seed=1)
    f, _ = forward_probabilities(x, _true_params())
    assert np.all(f >= 0.0)
    np.testing.assert_allclose(f.sum(axis=1), 1.0, atol=1e-12)
    for t in (1, 50, 399):
        prefix, _ = forward_probabilities(x[:t], _true_params())
        np.testing.assert_allclose(prefix, f[:t], rtol=1e-10, atol=1e-12)


def test_smoothing_uses_the_future_and_filtering_does_not() -> None:
    """The smoothed posterior at t changes when later rows change; the filtered one cannot."""
    x, _ = _simulate(300, seed=2)
    altered = x.copy()
    altered[200:] = altered[200:][::-1] + 1.0
    s0 = smoothed_probabilities(x, _true_params())
    s1 = smoothed_probabilities(altered, _true_params())
    f0, _ = forward_probabilities(x, _true_params())
    f1, _ = forward_probabilities(altered, _true_params())
    np.testing.assert_allclose(f0[:200], f1[:200], rtol=1e-12, atol=1e-14)
    assert float(np.abs(s0[:200] - s1[:200]).max()) > 1e-3


def test_last_smoothed_row_equals_last_filtered_row() -> None:
    x, _ = _simulate(250, seed=4)
    f, _ = forward_probabilities(x, _true_params())
    s = smoothed_probabilities(x, _true_params())
    np.testing.assert_allclose(s[-1], f[-1], rtol=1e-10, atol=1e-12)


def test_smoothed_differs_from_filtered_on_average() -> None:
    x, _ = _simulate(600, seed=5)
    f, _ = forward_probabilities(x, _true_params())
    s = smoothed_probabilities(x, _true_params())
    assert float(np.abs(f - s).mean()) > 1e-3


def test_log_density_matches_scipy_multivariate_normal() -> None:
    """One state, so the HMM log-likelihood is the sum of Gaussian log densities."""
    rng = np.random.default_rng(0)
    cov = np.array([[1.0, 0.3], [0.3, 0.7]])
    mean = np.array([0.5, -0.2])
    x = rng.multivariate_normal(mean, cov, size=40)
    two = HmmParams(
        startprob=np.array([0.5, 0.5]),
        transmat=np.array([[0.5, 0.5], [0.5, 0.5]]),
        means=np.stack([mean, mean]),
        covars=np.stack([cov, cov]),
    )
    _, ll = forward_probabilities(x, two)
    assert ll == pytest.approx(float(multivariate_normal(mean, cov).logpdf(x).sum()), rel=1e-12)


def test_extreme_observation_does_not_underflow() -> None:
    x, _ = _simulate(50, seed=6)
    x[10] = [60.0, 60.0]
    f, ll = forward_probabilities(x, _true_params())
    assert np.isfinite(f).all()
    assert np.isfinite(ll)
    assert f[10].argmax() == 1


# ---- Baum-Welch ----------------------------------------------------------------------


def test_em_loglik_is_non_decreasing_without_regularisation() -> None:
    x, _ = _simulate(1500, seed=7)
    start = HmmParams(
        np.array([0.5, 0.5]),
        np.array([[0.8, 0.2], [0.2, 0.8]]),
        np.array([[-0.5, -0.5], [1.0, 1.0]]),
        np.stack([np.eye(2), np.eye(2)]),
    )
    out = baum_welch(x, start, max_iter=30, tol=0.0, min_covar=0.0)
    assert out is not None
    trace = np.asarray(out.loglik_trace)
    assert len(trace) >= 5  # tol=0 stops only when float noise makes a gain negative
    assert np.all(np.diff(trace) > -1e-8 * np.abs(trace[:-1]))
    assert trace[-1] > trace[0]


def test_em_recovers_planted_parameters() -> None:
    x, _ = _simulate(6000, seed=8)
    fit = fit_gaussian_hmm(x, 2, seed=0)
    p = fit.params
    np.testing.assert_allclose(p.means, TRUE_MU, atol=0.15)
    np.testing.assert_allclose(np.diag(p.transmat), np.diag(TRUE_A), atol=0.02)
    np.testing.assert_allclose(p.covars, TRUE_COV, atol=0.15)
    assert fit.converged
    assert float(fit.state_share.sum()) == pytest.approx(1.0)


def test_fitted_loglik_beats_the_true_parameters_in_sample() -> None:
    """ML property: the fit cannot do worse than the generating parameters (small slack for
    the covariance floor and the stopping tolerance)."""
    x, _ = _simulate(3000, seed=9)
    fit = fit_gaussian_hmm(x, 2, seed=0, min_covar=1e-6)
    _, ll_true = forward_probabilities(x, _true_params())
    assert fit.loglik >= ll_true - 0.5


def test_fit_orders_states_by_the_chosen_column_whatever_the_generating_order() -> None:
    swapped = HmmParams(
        np.array([0.5, 0.5]),
        TRUE_A[::-1, ::-1],
        TRUE_MU[::-1],
        TRUE_COV[::-1],
    )
    x, _ = _simulate(4000, seed=10, params=swapped)
    fit = fit_gaussian_hmm(x, 2, seed=0)
    assert fit.params.means[0, 0] < fit.params.means[1, 0]
    by1 = fit_gaussian_hmm(x, 2, seed=0, order_by=1)
    assert by1.params.means[0, 1] < by1.params.means[1, 1]


def test_fit_is_deterministic_and_loglik_matches_the_filter() -> None:
    x, _ = _simulate(1200, seed=11)
    a = fit_gaussian_hmm(x, 2, seed=3)
    b = fit_gaussian_hmm(x, 2, seed=3)
    np.testing.assert_array_equal(a.params.means, b.params.means)
    np.testing.assert_array_equal(a.params.transmat, b.params.transmat)
    assert a.loglik == forward_probabilities(x, a.params)[1]


def test_warm_start_from_the_truth_needs_no_more_iterations_than_a_cold_start() -> None:
    x, _ = _simulate(2500, seed=12)
    cold = fit_gaussian_hmm(x, 2, seed=0)
    warm = fit_gaussian_hmm(x, 2, init=_true_params(), seed=0)
    assert warm.n_iter <= cold.n_iter
    assert warm.loglik == pytest.approx(cold.loglik, abs=1e-2)


def test_degenerate_start_returns_none_not_a_bad_fit() -> None:
    x, _ = _simulate(500, seed=13)
    far = HmmParams(
        np.array([0.5, 0.5]),
        np.array([[0.9, 0.1], [0.1, 0.9]]),
        np.array([[0.0, 0.0], [1e4, 1e4]]),
        np.stack([np.eye(2), np.eye(2)]),
    )
    assert baum_welch(x, far) is None


def test_degenerate_warm_start_falls_back_to_cold_starts() -> None:
    x, _ = _simulate(800, seed=14)
    far = HmmParams(
        np.array([0.5, 0.5]),
        np.array([[0.9, 0.1], [0.1, 0.9]]),
        np.array([[0.0, 0.0], [1e4, 1e4]]),
        np.stack([np.eye(2), np.eye(2)]),
    )
    fit = fit_gaussian_hmm(x, 2, init=far, seed=0)
    assert np.isfinite(fit.loglik)
    assert fit.params.means[1, 0] > fit.params.means[0, 0]


def test_fit_standardises_with_its_own_rows_only() -> None:
    """An affine map of the data (a unit change) must map the fit by the same affine map."""
    x, _ = _simulate(1500, seed=15)
    scale = np.array([100.0, 0.01])
    shift = np.array([5.0, -3.0])
    a = fit_gaussian_hmm(x, 2, seed=0, min_covar=0.0)
    b = fit_gaussian_hmm(x * scale + shift, 2, seed=0, min_covar=0.0)
    np.testing.assert_allclose(b.params.means, a.params.means * scale + shift, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(b.params.transmat, a.params.transmat, rtol=1e-6, atol=1e-8)
    assert b.loglik - a.loglik == pytest.approx(-np.log(scale).sum(), abs=1e-5)


def test_order_states_permutes_everything_consistently() -> None:
    p = _true_params()
    flipped = HmmParams(p.startprob[::-1], p.transmat[::-1, ::-1], p.means[::-1], p.covars[::-1])
    out = order_states(flipped)
    np.testing.assert_array_equal(out.means, p.means)
    np.testing.assert_array_equal(out.transmat, p.transmat)
    np.testing.assert_array_equal(out.covars, p.covars)
    with pytest.raises(RegimeHmmError):
        order_states(p, by=5)


# ---- validation ----------------------------------------------------------------------


def test_input_validation_for_primitives() -> None:
    x, _ = _simulate(100, seed=16)
    with pytest.raises(RegimeHmmError, match="2-D"):
        forward_probabilities(x[:, 0], _true_params())
    bad = x.copy()
    bad[3, 0] = np.nan
    with pytest.raises(RegimeHmmError, match="finite"):
        forward_probabilities(bad, _true_params())
    with pytest.raises(RegimeHmmError, match="empty"):
        forward_probabilities(np.empty((0, 2)), _true_params())
    with pytest.raises(RegimeHmmError, match="features"):
        forward_probabilities(x[:, :1], _true_params())
    broken = HmmParams(
        _true_params().startprob, np.array([[0.5, 0.4], [0.5, 0.5]]), TRUE_MU, TRUE_COV
    )
    with pytest.raises(RegimeHmmError, match="transmat"):
        forward_probabilities(x, broken)
    with pytest.raises(RegimeHmmError, match="n_states"):
        fit_gaussian_hmm(x, 1)
    with pytest.raises(RegimeHmmError, match="at least"):
        fit_gaussian_hmm(x[:30], 2)
    with pytest.raises(RegimeHmmError, match="n_init"):
        fit_gaussian_hmm(x, 2, n_init=0)
    with pytest.raises(RegimeHmmError, match="order_by"):
        fit_gaussian_hmm(x, 2, order_by=2)
    const = x.copy()
    const[:, 1] = 1.0
    with pytest.raises(RegimeHmmError, match="constant"):
        fit_gaussian_hmm(const, 2)
    with pytest.raises(RegimeHmmError, match="states"):
        fit_gaussian_hmm(x, 3, init=_true_params())


# ---- the causal driver ---------------------------------------------------------------


def test_refit_positions_known_answer() -> None:
    assert refit_positions(10, 4, 3) == (3, 6, 9)
    assert refit_positions(9, 4, 3) == (3, 6)
    assert refit_positions(3, 4, 3) == ()
    with pytest.raises(RegimeHmmError):
        refit_positions(10, 0, 3)


def test_driver_shapes_warmup_and_label_conventions() -> None:
    x, _ = _simulate(700, seed=17)
    obs = _frame(x)
    out = forward_filtered_hmm(obs, 2, min_obs=200, refit_every=100, n_init=2)
    assert list(out.probs.columns) == ["p_0", "p_1"]
    assert out.probs.index.equals(obs.index)
    assert out.labels.index.equals(obs.index)
    assert out.probs.iloc[:199].isna().all().all()
    assert out.probs.iloc[199:].notna().all().all()
    np.testing.assert_allclose(out.probs.iloc[199:].sum(axis=1), 1.0, atol=1e-12)
    expected = out.probs.iloc[199:].to_numpy().argmax(axis=1).astype(float)
    np.testing.assert_array_equal(out.labels.iloc[199:].to_numpy(), expected)
    assert out.labels.iloc[:199].isna().all()
    assert len(out.fits) == len(refit_positions(700, 200, 100)) == 6
    assert out.fits.index[0] == obs.index[199]
    assert out.fits["n_fit_obs"].tolist() == [200, 300, 400, 500, 600, 700]
    assert out.last_params is not None


def test_driver_uses_the_parameters_of_the_last_refit_in_each_segment() -> None:
    """Row t of a segment equals the plain forward filter run with that refit's fitted
    parameters, which are re-derived here from the same prefix and the same warm-start chain."""
    x, _ = _simulate(500, seed=18)
    obs = _frame(x)
    kwargs = {"n_init": 2, "max_iter": 50, "tol": 1e-6, "min_covar": 1e-3, "order_by": 0}
    out = forward_filtered_hmm(obs, 2, min_obs=200, refit_every=150, seed=0, **kwargs)
    positions = refit_positions(500, 200, 150)
    prev: HmmParams | None = None
    for k, pos in enumerate(positions):
        seed_k = int(np.random.default_rng([0, k]).integers(0, 2**31 - 1))
        fit = fit_gaussian_hmm(x[: pos + 1], 2, init=prev, seed=seed_k, **kwargs)
        prev = fit.params
        end = positions[k + 1] - 1 if k + 1 < len(positions) else 499
        expected, _ = forward_probabilities(x[: end + 1], fit.params)
        np.testing.assert_allclose(
            out.probs.to_numpy()[pos : end + 1], expected[pos : end + 1], rtol=1e-9, atol=1e-12
        )


def test_driver_handles_a_leading_nan_block_and_rejects_interior_nan() -> None:
    x, _ = _simulate(600, seed=19)
    obs = _frame(x)
    obs.iloc[:40] = np.nan
    out = forward_filtered_hmm(obs, 2, min_obs=200, refit_every=200, n_init=2)
    assert out.probs.iloc[: 40 + 199].isna().all().all()
    assert out.probs.iloc[40 + 199 :].notna().all().all()
    broken = obs.copy()
    broken.iloc[300, 0] = np.nan
    with pytest.raises(RegimeHmmError, match="only leading NaN"):
        forward_filtered_hmm(broken, 2, min_obs=200, refit_every=200)


def test_driver_returns_all_nan_when_history_is_shorter_than_min_obs() -> None:
    x, _ = _simulate(150, seed=20)
    out = forward_filtered_hmm(_frame(x), 2, min_obs=200)
    assert out.probs.isna().all().all()
    assert out.labels.isna().all()
    assert out.fits.empty
    assert out.last_params is None
    assert hmm_summary_lines(out) == ["no HMM fit (fewer valid rows than min_obs)"]


def test_driver_recovers_planted_regimes_causally() -> None:
    x, states = _simulate(2500, seed=21)
    out = forward_filtered_hmm(_frame(x), 2, min_obs=300, refit_every=200, n_init=3)
    labelled = out.labels.notna().to_numpy()
    accuracy = float((out.labels.to_numpy()[labelled] == states[labelled]).mean())
    assert accuracy > 0.95
    # filtering is never better than smoothing with the true parameters, and both are good
    smoothed = smoothed_probabilities(x, _true_params()).argmax(axis=1)
    assert float((smoothed == states).mean()) >= accuracy - 0.02
    last = out.last_params
    assert last is not None
    np.testing.assert_allclose(last.means, TRUE_MU, atol=0.25)


def test_driver_is_deterministic_and_three_states_run() -> None:
    mu3 = np.array([[0.0, 0.0], [2.5, 1.5], [5.0, 3.0]])
    p3 = HmmParams(
        np.full(3, 1 / 3),
        np.array([[0.96, 0.04, 0.0], [0.03, 0.94, 0.03], [0.0, 0.05, 0.95]]) + 1e-9,
        mu3,
        np.stack([0.5 * np.eye(2)] * 3),
    )
    p3 = HmmParams(p3.startprob, p3.transmat / p3.transmat.sum(1, keepdims=True), mu3, p3.covars)
    x, _ = _simulate(1500, seed=22, params=p3)
    obs = _frame(x)
    a = forward_filtered_hmm(obs, 3, min_obs=400, refit_every=300, n_init=3, seed=1)
    b = forward_filtered_hmm(obs, 3, min_obs=400, refit_every=300, n_init=3, seed=1)
    pd.testing.assert_frame_equal(a.probs, b.probs)
    assert list(a.probs.columns) == ["p_0", "p_1", "p_2"]
    assert set(a.labels.dropna().unique()) <= {0.0, 1.0, 2.0}
    cold = forward_filtered_hmm(
        obs, 3, min_obs=400, refit_every=300, n_init=3, seed=1, warm_start=False
    )
    assert cold.probs.notna().sum().sum() == a.probs.notna().sum().sum()


def test_driver_input_validation() -> None:
    x, _ = _simulate(300, seed=23)
    obs = _frame(x)
    with pytest.raises(RegimeHmmError, match="DataFrame"):
        forward_filtered_hmm(obs["log_vol"])  # type: ignore[arg-type]
    with pytest.raises(RegimeHmmError, match="increasing"):
        forward_filtered_hmm(obs.iloc[::-1])
    with pytest.raises(RegimeHmmError, match="n_states"):
        forward_filtered_hmm(obs, 1)
    with pytest.raises(RegimeHmmError, match="refit_every"):
        forward_filtered_hmm(obs, 2, refit_every=0)
    with pytest.raises(RegimeHmmError, match="min_obs"):
        forward_filtered_hmm(obs, 2, min_obs=20)
    with pytest.raises(RegimeHmmError, match="order_by"):
        forward_filtered_hmm(obs, 2, min_obs=100, order_by=7)
    inf = obs.copy()
    inf.iloc[5, 0] = np.inf
    with pytest.raises(RegimeHmmError, match="inf"):
        forward_filtered_hmm(inf, 2, min_obs=100)


def test_driver_does_not_mutate_its_input() -> None:
    x, _ = _simulate(400, seed=24)
    obs = _frame(x)
    before = obs.copy()
    forward_filtered_hmm(obs, 2, min_obs=200, refit_every=100, n_init=2)
    pd.testing.assert_frame_equal(obs, before)


def test_summary_lines_report_the_fits() -> None:
    x, _ = _simulate(700, seed=25)
    out = forward_filtered_hmm(_frame(x), 2, min_obs=200, refit_every=250, n_init=2)
    lines = hmm_summary_lines(out)
    assert lines[0].startswith("refits: 3")
    assert any(line.startswith("state 1:") for line in lines)


# ---- observation builder -------------------------------------------------------------


def _panel(n: int = 120, seed: int = 0) -> tuple[pd.DataFrame, dict[str, str]]:
    rng = np.random.default_rng(seed)
    cols = ["A1", "A2", "B1", "B2", "C1"]
    sectors = {"A1": "a", "A2": "a", "B1": "b", "B2": "b", "C1": "c"}
    common = rng.standard_normal((n, 1))
    data = 0.6 * common + rng.standard_normal((n, len(cols)))
    index = pd.bdate_range("2020-01-01", periods=n, name="date")
    return pd.DataFrame(data * 0.01, index=index, columns=cols), sectors


def test_observations_match_an_independent_computation_on_one_window() -> None:
    returns, sectors = _panel()
    window = 30
    obs = hmm_observations(returns, sectors, window=window)
    t = 77
    block = returns.iloc[t - window + 1 : t + 1].to_numpy()
    expected_vol = np.log(block.std(axis=0, ddof=1).mean())
    corr = np.corrcoef(block, rowvar=False)
    cross = [
        (i, j)
        for i in range(5)
        for j in range(i + 1, 5)
        if sectors[returns.columns[i]] != sectors[returns.columns[j]]
    ]
    assert len(cross) == 8  # 10 pairs minus the two within-sector ones
    expected_z = np.mean([np.arctanh(corr[i, j]) for i, j in cross])
    assert obs["log_vol"].iloc[t] == pytest.approx(expected_vol, rel=1e-12)
    assert obs["z_cross_corr"].iloc[t] == pytest.approx(expected_z, rel=1e-12)
    assert list(obs.columns) == ["log_vol", "z_cross_corr"]
    assert obs.index.equals(returns.index)


def test_observations_warmup_and_nan_propagation() -> None:
    returns, sectors = _panel()
    obs = hmm_observations(returns, sectors, window=30)
    assert obs.iloc[:29].isna().all().all()
    assert obs.iloc[29:].notna().all().all()
    holey = returns.copy()
    holey.iloc[50, 0] = np.nan
    out = hmm_observations(holey, sectors, window=30)
    # every window containing row 50 is NaN: rows 50..79
    assert out.iloc[50:80].isna().all().all()
    assert out.iloc[80:].notna().all().all()
    assert out.iloc[29:50].notna().all().all()
    flat = returns.copy()
    flat.iloc[:, 1] = 0.0
    assert hmm_observations(flat, sectors, window=30).isna().all().all()


def test_observations_rise_with_a_common_factor_variance_and_correlation() -> None:
    rng = np.random.default_rng(3)
    n, m = 400, 6
    sectors = {f"T{i}": ("a" if i < 3 else "b") for i in range(m)}
    load = np.where(np.arange(400) < 200, 0.2, 2.0)[:, None]
    data = load * rng.standard_normal((n, 1)) + rng.standard_normal((n, m))
    returns = pd.DataFrame(
        data * 0.01, index=pd.bdate_range("2020-01-01", periods=n), columns=list(sectors)
    )
    obs = hmm_observations(returns, sectors, window=60)
    calm = obs.iloc[100:190].mean()
    stress = obs.iloc[300:390].mean()
    assert stress["log_vol"] > calm["log_vol"] + 0.3
    assert stress["z_cross_corr"] > calm["z_cross_corr"] + 0.5


def test_observations_validation() -> None:
    returns, sectors = _panel()
    with pytest.raises(RegimeHmmError, match="DataFrame"):
        hmm_observations(returns["A1"], sectors)  # type: ignore[arg-type]
    with pytest.raises(RegimeHmmError, match="increasing"):
        hmm_observations(returns.iloc[::-1], sectors)
    with pytest.raises(RegimeHmmError, match="window"):
        hmm_observations(returns, sectors, window=5)
    with pytest.raises(RegimeHmmError, match="no sector"):
        hmm_observations(returns, {"A1": "a"})
    with pytest.raises(RegimeHmmError, match="two sectors"):
        hmm_observations(returns, dict.fromkeys(returns.columns, "x"))
    bad = returns.copy()
    bad.iloc[3, 0] = np.inf
    with pytest.raises(RegimeHmmError, match="inf"):
        hmm_observations(bad, sectors)


# ---- further fit properties ----------------------------------------------------------


def test_state_share_follows_the_state_ordering() -> None:
    """``state_share`` is the mean smoothed probability at the returned parameters, reported in
    the final (ordered) state order, on data where the two states have unequal occupancy."""
    skewed = HmmParams(
        np.array([0.5, 0.5]),
        np.array([[0.99, 0.01], [0.05, 0.95]]),
        TRUE_MU[::-1],
        TRUE_COV[::-1],
    )
    x, _ = _simulate(5000, seed=26, params=skewed)
    # warm start in the generating (descending) order, so the internal EM order is the reverse of
    # the final ordering and the permutation of state_share is actually exercised
    fit = fit_gaussian_hmm(x, 2, init=skewed, seed=0)
    assert fit.params.means[0, 0] < fit.params.means[1, 0]
    expected = smoothed_probabilities(x, fit.params).mean(axis=0)
    np.testing.assert_allclose(fit.state_share, expected, atol=1e-6)
    assert abs(float(expected[0]) - float(expected[1])) > 0.2


def test_min_covar_is_a_floor_in_the_standardised_space() -> None:
    x, _ = _simulate(1500, seed=27)
    floor = 0.4
    fit = fit_gaussian_hmm(x, 2, seed=0, min_covar=floor)
    for j in range(2):
        assert np.all(np.diag(fit.params.covars[j]) >= floor * x.var(axis=0) - 1e-9)
    loose = fit_gaussian_hmm(x, 2, seed=0, min_covar=0.0)
    assert float(np.diag(loose.params.covars[0]).min()) < float(np.diag(fit.params.covars[0]).min())


def test_affine_parameter_mapping_round_trips_and_moves_the_density_correctly() -> None:
    mean = np.array([5.0, -3.0])
    std = np.array([100.0, 0.01])
    raw = _true_params()
    z = _to_z(raw, mean, std)
    back = _from_z(z, mean, std)
    np.testing.assert_allclose(back.means, raw.means, rtol=1e-12)
    np.testing.assert_allclose(back.covars, raw.covars, rtol=1e-12)
    # the log-likelihood in z-space equals the raw one plus the Jacobian term
    x, _ = _simulate(200, seed=28)
    _, ll_raw = forward_probabilities(x, raw)
    _, ll_z = forward_probabilities((x - mean) / std, z)
    assert ll_z - ll_raw == pytest.approx(len(x) * np.log(std).sum(), rel=1e-9)


def test_each_refit_gets_its_own_deterministic_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    import etfcorr.regime_hmm as module

    seen: list[int] = []
    real = module.fit_gaussian_hmm

    def spy(*args: object, **kwargs: object) -> object:
        seen.append(int(kwargs["seed"]))  # type: ignore[call-overload]
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(module, "fit_gaussian_hmm", spy)
    x, _ = _simulate(500, seed=29)
    forward_filtered_hmm(_frame(x), 2, min_obs=200, refit_every=100, n_init=2, seed=7)
    first = list(seen)
    seen.clear()
    forward_filtered_hmm(_frame(x), 2, min_obs=200, refit_every=100, n_init=2, seed=7)
    assert seen == first
    assert len(first) == 4
    assert len(set(first)) == 4


def test_min_separation_known_answer_and_it_is_reported_per_refit() -> None:
    params = HmmParams(
        np.array([0.5, 0.5]),
        TRUE_A,
        np.array([[0.0, 5.0], [2.0, 5.0]]),
        np.array([[[1.0, 0.0], [0.0, 1.0]], [[3.0, 0.0], [0.0, 1.0]]]),
    )
    assert _min_separation(params, 0) == pytest.approx(2.0 / np.sqrt(2.0))
    assert _min_separation(params, 1) == pytest.approx(0.0)
    x, _ = _simulate(700, seed=30)
    out = forward_filtered_hmm(_frame(x), 2, min_obs=200, refit_every=250, n_init=2)
    assert "min_separation" in out.fits.columns
    assert float(out.fits["min_separation"].min()) > 2.0  # planted gap of 2.5 on column 0
