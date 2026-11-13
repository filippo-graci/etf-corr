"""Step 4.3: forward-filtered Gaussian hidden Markov regimes (causal, expanding-window refits).

Question served: Phase 4 regime definitions. H2 (``config/experiment.yaml``) is tested on VIX
terciles (step 4.1, no model fitted). This module supplies the *model-based* regime of
``project_plan.md`` section 6.5 -- a Gaussian HMM on cross-sector volatility and correlation -- as
(a) an exploratory alternative regime definition for the H2 engine and (b) a causal regime-state
feature for Phases 5-7. It is **not** part of the pre-registered H2 test and does not change it.

**What "forward-filtered" means, and why it is the only mode here.** A hidden Markov model gives
two different posteriors for the unobserved state ``S_t``:

* the *filtered* probability ``P(S_t | x_1..x_t)`` uses observations up to ``t`` only;
* the *smoothed* probability ``P(S_t | x_1..x_T)`` uses the whole sample, i.e. the future.

Library posteriors are typically the smoothed ones, which is the look-ahead pitfall flagged in
``EFT_project_v2.md`` (hmmlearn's documentation does not say which one ``predict_proba`` returns;
the cross-check test in ``tests/unit/test_regime_hmm_hmmlearn.py`` settles it empirically, and
``predict`` is a whole-sequence decoding by construction). Here the state probability at ``t`` is
computed by the forward recursion

    alpha_t(j)  is proportional to  b_j(x_t) * sum_i alpha_{t-1}(i) * A_ij

with ``b_j`` the Gaussian emission density and ``A`` the transition matrix, **and the parameters
used at ``t`` were estimated on data dated <= the last refit date <= t**. Parameters are
re-estimated on an expanding window: the first fit uses the first ``min_obs`` valid observations,
then every ``refit_every`` rows the model is refit on everything seen so far. The refit schedule
counts rows from the first valid observation, so it does not depend on how much *later* data
exists; the output at ``t`` is therefore a pure function ``f(data_up_to_t) -> value_at_t`` (checked
with the shared detectors in ``tests/leakage/test_regime_hmm_no_lookahead.py``, which also prove
they catch two leaky twins: parameters fitted once on the full sample, and smoothed posteriors).
:func:`smoothed_probabilities` is exported **only** for diagnostics (how much does smoothing
change the labels) and for those leaky twins; never use it as a feature.

**Estimation.** ``fit_gaussian_hmm`` is an own, numpy-only Baum-Welch (EM) fit
(Rabiner 1989) with scaled forward-backward recursions. It is own code rather than hmmlearn because
(i) hmmlearn does not expose a forward-only filter, so the filter had to be written anyway,
(ii) the sandbox the code is gate-checked in cannot install hmmlearn, and (iii) the inference
functions are cross-checked against hmmlearn's own ``score``/``predict_proba`` in
``tests/unit/test_regime_hmm_hmmlearn.py`` (skipped when hmmlearn is absent). Observations are
standardised with the **training window's** mean and standard deviation (never full-sample), the
covariance is regularised by ``min_covar`` times the identity in the standardised space (a floor,
like hmmlearn's ``min_covar``; with ``min_covar > 0`` the M-step is not the exact ML maximiser, so
the log-likelihood is only approximately monotone), and fits are initialised by quantile
splitting of the first observation column plus ``n_init - 1`` random-point starts, or, after the
first fit and with ``warm_start=True``, by the previous refit's parameters (cold starts are the
fallback if the warm start degenerates). Randomness is a function of ``(seed, refit number)`` only.

**Label switching is fixed by ordering.** HMM state indices are arbitrary. After every fit the
states are sorted by the fitted mean of observation column ``order_by`` (default column 0, which
:func:`hmm_observations` makes the log volatility), so state ``0`` is always the lowest-volatility
("calm") state and the highest state the highest ("stress"). This depends on the fitted parameters
only, hence is causal. It does **not** guarantee that a state means the same thing at every refit:
the means move as data accumulate.

**Known failure modes (state them wherever a result is used).** Local optima of EM (mitigated, not
removed, by restarts); weak identifiability of the states in a short first window; the plug-in
filter ignores parameter uncertainty; the observations from :func:`hmm_observations` are trailing
window statistics, so consecutive observations overlap and are strongly autocorrelated, which
violates the HMM's conditional independence and makes the probabilities overconfident and the
states more persistent than the data-generating process (**[Inference]** for the mechanism;
**measured** consequence, ``scripts/hmm_calibration_4_3.py``, one data-generating process, five
panels of independent-in-time returns with N=25 and T=3201: the fitted two-state HMM still reports
persistent states (median stay probability 0.98-0.99) separated by 0.02-3.3 pooled standard
deviations on the log-volatility column, against 3.1-4.1 on a planted two-regime panel; so neither
persistence nor separation alone is evidence that regimes exist, and a real-data fit must be read
against a null made of the same pipeline run on shuffled returns); the hard label is the argmax
of the filtered probability, which is not the Viterbi path (Viterbi uses the whole sequence); a
two-state default is a parsimony choice fixed before any result, three
states are a sensitivity check. Daily returns have ``NaN`` where an asset is missing, and the
window statistics are ``NaN`` there; interior ``NaN`` in the HMM input are rejected, not filled.

**Citations (checked by search/fetch on 2026-10-02; the papers' *results* are not used).**
Hamilton, J. D. (1989), *A New Approach to the Economic Analysis of Nonstationary Time Series and
the Business Cycle*, Econometrica 57(2), 357-384 (title, author, venue, pages checked; the abstract
supports "regime changes modelled by a discrete Markov process with probabilistic inference about
the unobserved regime", which is all this docstring relies on it for). Rabiner, L. R. (1989),
*A tutorial on hidden Markov models and selected applications in speech recognition*,
Proceedings of the IEEE 77(2), 257-286, doi 10.1109/5.18626 (title, author, venue, pages, doi
checked; the listing names the forward-backward procedure and the re-estimation formulas, the
recursions themselves are textbook material **[Established]**). Ang, A. and Timmermann, A. (2012),
*Regime Changes and Financial Markets*, Annual Review of Financial Economics 4, 313-337 (checked;
motivation only: regime-switching models are used for persistence, heteroskedasticity and
time-varying correlations in asset returns). Library reference: hmmlearn ``GaussianHMM`` (its
documentation does not state whether ``predict_proba`` is filtered or smoothed; the local
cross-check test decides it empirically).

Not adapted from third-party code (no licence flag): Baum-Welch and the forward filter are
implemented from the textbook recursions. Pure functions of their inputs: no I/O, no global state,
no dependency on ``etfcorr.config`` or ``etfcorr.data``.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import log, pi

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]

DEFAULT_N_STATES = 2
DEFAULT_MIN_OBS = 504  # two years of daily rows before the first fit
DEFAULT_REFIT_EVERY = 63  # one quarter
DEFAULT_WINDOW = 63  # trailing window of hmm_observations
PROB_FLOOR = 1e-6  # floor on start and transition probabilities (keeps the recursion alive)
_CORR_CLIP = 0.999  # before Fisher's z


class RegimeHmmError(ValueError):
    """Invalid input to, or a numerical failure of, an HMM regime call."""


# --------------------------------------------------------------------------------------
# Parameter containers
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HmmParams:
    """Parameters of a Gaussian HMM in the space of the observations they are applied to.

    ``startprob`` (K,), ``transmat`` (K, K, rows sum to 1), ``means`` (K, D), ``covars``
    (K, D, D, full covariance per state).
    """

    startprob: FloatArray
    transmat: FloatArray
    means: FloatArray
    covars: FloatArray

    @property
    def n_states(self) -> int:
        return int(self.means.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.means.shape[1])


@dataclass(frozen=True)
class EmResult:
    """Outcome of one Baum-Welch run: parameters, log-likelihood trace and occupancy."""

    params: HmmParams
    loglik_trace: tuple[float, ...]
    converged: bool
    state_share: FloatArray  # mean smoothed state probability at the returned parameters

    @property
    def loglik(self) -> float:
        return self.loglik_trace[-1]

    @property
    def n_iter(self) -> int:
        return len(self.loglik_trace) - 1


@dataclass(frozen=True)
class HmmFit:
    """A fit on raw (unstandardised) observations with states ordered by ``order_by``."""

    params: HmmParams
    loglik: float
    n_iter: int
    converged: bool
    state_share: FloatArray


@dataclass(frozen=True)
class HmmRegimes:
    """Output of :func:`forward_filtered_hmm`.

    ``probs``: filtered state probabilities, columns ``p_0 .. p_{K-1}`` (state 0 = lowest mean of
    the ordering column), ``NaN`` before the first fit. ``labels``: argmax of ``probs`` as float
    codes ``0.0 .. K-1`` (``NaN`` where ``probs`` is ``NaN``); the same convention as
    ``etfcorr.regimes`` so ``regime_summary`` and the H2 engine accept it. ``fits``: one row per
    refit (index = date of the last observation used by the fit; columns include the fit
    diagnostics ``converged``, ``min_state_share``, ``p_stay_*``, ``order_mean_*`` and
    ``min_separation``). ``last_params``: the final fit
    (``None`` if there was no fit).
    """

    probs: pd.DataFrame
    labels: pd.Series
    fits: pd.DataFrame
    last_params: HmmParams | None


# --------------------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------------------


def _check_matrix(x: FloatArray, name: str = "x") -> FloatArray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise RegimeHmmError(f"{name} must be 2-D (rows x features), got shape {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise RegimeHmmError(f"{name} is empty")
    if not np.isfinite(arr).all():
        raise RegimeHmmError(f"{name} must be finite (no NaN or +-inf)")
    return arr


def _check_params(params: HmmParams, n_features: int | None = None) -> None:
    k = params.n_states
    d = params.n_features
    if n_features is not None and d != n_features:
        raise RegimeHmmError(f"params have {d} features, data has {n_features}")
    if params.startprob.shape != (k,) or params.transmat.shape != (k, k):
        raise RegimeHmmError("startprob/transmat shapes do not match the number of states")
    if params.covars.shape != (k, d, d):
        raise RegimeHmmError("covars must have shape (n_states, n_features, n_features)")
    for name, arr in (
        ("startprob", params.startprob),
        ("transmat", params.transmat),
        ("means", params.means),
        ("covars", params.covars),
    ):
        if not np.isfinite(arr).all():
            raise RegimeHmmError(f"{name} contains non-finite values")
    if (params.startprob < 0).any() or not np.isclose(params.startprob.sum(), 1.0):
        raise RegimeHmmError("startprob must be non-negative and sum to 1")
    if (params.transmat < 0).any() or not np.allclose(params.transmat.sum(axis=1), 1.0):
        raise RegimeHmmError("every transmat row must be non-negative and sum to 1")


# --------------------------------------------------------------------------------------
# Inference primitives
# --------------------------------------------------------------------------------------


def _log_emission(x: FloatArray, means: FloatArray, covars: FloatArray) -> FloatArray:
    """Gaussian log densities, shape (rows, states), via a Cholesky factor per state."""
    n, d = x.shape
    k = means.shape[0]
    out = np.empty((n, k))
    for j in range(k):
        try:
            chol = np.linalg.cholesky(covars[j])
        except np.linalg.LinAlgError as exc:
            raise RegimeHmmError(f"covariance of state {j} is not positive definite") from exc
        z = np.linalg.solve(chol, (x - means[j]).T)
        log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
        out[:, j] = -0.5 * (d * log(2.0 * pi) + log_det + np.sum(z * z, axis=0))
    return out


def _forward(
    log_b: FloatArray, startprob: FloatArray, transmat: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    """Scaled forward recursion: (filtered alpha, scale, shifted emissions, log-likelihood)."""
    n, k = log_b.shape
    shift = log_b.max(axis=1)
    b = np.exp(log_b - shift[:, None])
    alpha = np.empty((n, k))
    scale = np.empty(n)
    a = startprob * b[0]
    s = float(a.sum())
    if not s > 0.0:
        raise RegimeHmmError("forward recursion underflowed at the first row")
    alpha[0] = a / s
    scale[0] = s
    for t in range(1, n):
        a = (alpha[t - 1] @ transmat) * b[t]
        s = float(a.sum())
        if not s > 0.0:
            raise RegimeHmmError(f"forward recursion underflowed at row {t}")
        alpha[t] = a / s
        scale[t] = s
    loglik = float(np.sum(np.log(scale)) + shift.sum())
    return alpha, scale, b, loglik


def _backward(b: FloatArray, scale: FloatArray, transmat: FloatArray) -> FloatArray:
    """Scaled backward recursion matching :func:`_forward`'s scaling."""
    n, k = b.shape
    beta = np.empty((n, k))
    beta[-1] = 1.0
    for t in range(n - 2, -1, -1):
        beta[t] = transmat @ (b[t + 1] * beta[t + 1]) / scale[t + 1]
    return beta


def forward_probabilities(x: FloatArray, params: HmmParams) -> tuple[FloatArray, float]:
    """Filtered state probabilities ``P(S_t | x_1..x_t)`` (rows sum to 1) and the log-likelihood.

    Row ``t`` of the result depends on ``x[: t + 1]`` and ``params`` only.
    """
    arr = _check_matrix(x)
    _check_params(params, arr.shape[1])
    log_b = _log_emission(arr, params.means, params.covars)
    alpha, _, _, loglik = _forward(log_b, params.startprob, params.transmat)
    return alpha, loglik


def smoothed_probabilities(x: FloatArray, params: HmmParams) -> FloatArray:
    """Smoothed state probabilities ``P(S_t | x_1..x_T)``: **uses the future, diagnostics only**.

    Exported to measure how far smoothing moves the labels and to build the leaky twin in the
    leakage tests; the last row equals the filtered probability of the last row.
    """
    arr = _check_matrix(x)
    _check_params(params, arr.shape[1])
    log_b = _log_emission(arr, params.means, params.covars)
    alpha, scale, b, _ = _forward(log_b, params.startprob, params.transmat)
    beta = _backward(b, scale, params.transmat)
    gamma = alpha * beta
    return gamma / gamma.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------------------
# Baum-Welch
# --------------------------------------------------------------------------------------


def _floor_rows(p: FloatArray) -> FloatArray:
    q = np.maximum(p, PROB_FLOOR)
    return np.asarray(q / q.sum(axis=-1, keepdims=True), dtype=float)


def _m_step(
    x: FloatArray,
    gamma: FloatArray,
    xi_sum: FloatArray,
    min_covar: float,
) -> HmmParams | None:
    d = x.shape[1]
    k = gamma.shape[1]
    weight = gamma.sum(axis=0)
    if (weight < 1.0).any() or not np.isfinite(weight).all():
        return None  # a state with under one effective observation: degenerate fit
    means = (gamma.T @ x) / weight[:, None]
    covars = np.empty((k, d, d))
    for j in range(k):
        dev = x - means[j]
        covars[j] = (gamma[:, j, None] * dev).T @ dev / weight[j] + min_covar * np.eye(d)
    row = xi_sum.sum(axis=1, keepdims=True)
    if not (row > 0).all():
        return None
    out = HmmParams(
        startprob=_floor_rows(gamma[0]),
        transmat=_floor_rows(xi_sum / row),
        means=means,
        covars=covars,
    )
    finite = all(np.isfinite(a).all() for a in (out.means, out.covars, out.transmat))
    return out if finite else None


def baum_welch(
    x: FloatArray,
    init: HmmParams,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    min_covar: float = 1e-3,
) -> EmResult | None:
    """EM for a Gaussian HMM from ``init`` on rows ``x`` (used as given, no standardisation).

    Stops when the log-likelihood gain of an iteration is below ``tol * n_rows`` or after
    ``max_iter`` M-steps; the returned parameters are always the ones the last trace entry was
    evaluated at. Returns ``None`` if the run degenerates (a state with under one effective
    observation, a non-positive-definite covariance, an underflow): the caller tries another start.
    """
    arr = _check_matrix(x)
    _check_params(init, arr.shape[1])
    if max_iter < 0 or tol < 0 or min_covar < 0:
        raise RegimeHmmError("max_iter, tol and min_covar must be non-negative")
    n = arr.shape[0]
    params = init
    trace: list[float] = []
    converged = False
    share = np.full(init.n_states, np.nan)
    for it in range(max_iter + 1):
        try:
            log_b = _log_emission(arr, params.means, params.covars)
            alpha, scale, b, loglik = _forward(log_b, params.startprob, params.transmat)
        except RegimeHmmError:
            return None
        beta = _backward(b, scale, params.transmat)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        trace.append(loglik)
        share = gamma.mean(axis=0)
        if it > 0 and loglik - trace[-2] < tol * n:
            converged = True
            break
        if it == max_iter:
            break
        xi_sum = params.transmat * (alpha[:-1].T @ (b[1:] * beta[1:] / scale[1:, None]))
        updated = _m_step(arr, gamma, xi_sum, min_covar)
        if updated is None:
            return None
        params = updated
    return EmResult(params, tuple(trace), converged, np.asarray(share, dtype=float))


def _quantile_init(z: FloatArray, k: int, min_covar: float) -> HmmParams:
    """Deterministic start: split the rows into ``k`` groups by quantiles of column 0."""
    d = z.shape[1]
    order = np.argsort(z[:, 0], kind="stable")
    groups = np.array_split(order, k)
    means = np.stack([z[g].mean(axis=0) for g in groups])
    covars = np.stack(
        [np.cov(z[g], rowvar=False).reshape(d, d) + min_covar * np.eye(d) for g in groups]
    )
    return HmmParams(
        startprob=np.full(k, 1.0 / k),
        transmat=_persistent_transmat(k),
        means=means,
        covars=covars,
    )


def _random_init(z: FloatArray, k: int, rng: np.random.Generator, min_covar: float) -> HmmParams:
    n, d = z.shape
    rows = rng.choice(n, size=k, replace=False)
    base = np.cov(z, rowvar=False).reshape(d, d) + min_covar * np.eye(d)
    return HmmParams(
        startprob=np.full(k, 1.0 / k),
        transmat=_persistent_transmat(k),
        means=z[rows].copy(),
        covars=np.stack([base for _ in range(k)]),
    )


def _persistent_transmat(k: int) -> FloatArray:
    t = np.full((k, k), 0.1 / (k - 1))
    np.fill_diagonal(t, 0.9)
    return np.asarray(t, dtype=float)


def order_states(params: HmmParams, by: int = 0) -> HmmParams:
    """Reindex the states so ``means[:, by]`` is ascending (ties broken by original index)."""
    if not 0 <= by < params.n_features:
        raise RegimeHmmError(f"order_by must be in [0, {params.n_features}), got {by}")
    perm = np.argsort(params.means[:, by], kind="stable")
    return HmmParams(
        startprob=params.startprob[perm],
        transmat=params.transmat[np.ix_(perm, perm)],
        means=params.means[perm],
        covars=params.covars[perm],
    )


def _to_z(params: HmmParams, mean: FloatArray, std: FloatArray) -> HmmParams:
    return HmmParams(
        startprob=params.startprob,
        transmat=params.transmat,
        means=(params.means - mean) / std,
        covars=params.covars / np.outer(std, std),
    )


def _from_z(params: HmmParams, mean: FloatArray, std: FloatArray) -> HmmParams:
    return HmmParams(
        startprob=params.startprob,
        transmat=params.transmat,
        means=params.means * std + mean,
        covars=params.covars * np.outer(std, std),
    )


def fit_gaussian_hmm(
    x: FloatArray,
    n_states: int = DEFAULT_N_STATES,
    *,
    init: HmmParams | None = None,
    n_init: int = 5,
    max_iter: int = 100,
    tol: float = 1e-6,
    min_covar: float = 1e-3,
    order_by: int = 0,
    seed: int = 0,
) -> HmmFit:
    """Fit a Gaussian HMM (full covariances) by Baum-Welch on the rows of ``x``.

    ``x`` is standardised with its own mean and standard deviation (so everything the fit uses is
    inside ``x``), EM runs in that space and the result is mapped back to the raw space of ``x``
    and ordered by ``means[:, order_by]`` (:func:`order_states`). ``init`` (raw space, K states)
    is a warm start; if it is absent or degenerates, ``n_init`` cold starts are used (one
    quantile split of column 0, then ``n_init - 1`` random-point starts drawn from
    ``default_rng(seed)``) and the best log-likelihood wins. Raises :class:`RegimeHmmError` if
    every start degenerates.
    """
    arr = _check_matrix(x)
    n, d = arr.shape
    if n_states < 2:
        raise RegimeHmmError(f"n_states must be >= 2, got {n_states}")
    if n < 20 * n_states:
        raise RegimeHmmError(f"need at least {20 * n_states} rows for {n_states} states, got {n}")
    if n_init < 1:
        raise RegimeHmmError(f"n_init must be >= 1, got {n_init}")
    if not 0 <= order_by < d:
        raise RegimeHmmError(f"order_by must be in [0, {d}), got {order_by}")
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    if not (std > 0).all() or not np.isfinite(std).all():
        raise RegimeHmmError("a column of x is constant: cannot standardise")
    z = (arr - mean) / std

    results: list[EmResult] = []
    if init is not None:
        _check_params(init, d)
        if init.n_states != n_states:
            raise RegimeHmmError(f"init has {init.n_states} states, expected {n_states}")
        warm = baum_welch(
            z, _to_z(init, mean, std), max_iter=max_iter, tol=tol, min_covar=min_covar
        )
        if warm is not None:
            results.append(warm)
    if not results:
        rng = np.random.default_rng(seed)
        starts = [_quantile_init(z, n_states, min_covar)]
        starts += [_random_init(z, n_states, rng, min_covar) for _ in range(n_init - 1)]
        for start in starts:
            run = baum_welch(z, start, max_iter=max_iter, tol=tol, min_covar=min_covar)
            if run is not None:
                results.append(run)
    if not results:
        raise RegimeHmmError("EM degenerated from every starting point")
    best = max(results, key=lambda r: r.loglik)

    raw = _from_z(best.params, mean, std)
    perm = np.argsort(raw.means[:, order_by], kind="stable")
    ordered = order_states(raw, order_by)
    # The log-likelihood in z-space differs from the raw-space one by a constant Jacobian term,
    # so report the raw-space value (the one forward_probabilities gives for the same data).
    loglik = float(forward_probabilities(arr, ordered)[1])
    return HmmFit(
        params=ordered,
        loglik=loglik,
        n_iter=best.n_iter,
        converged=best.converged,
        state_share=best.state_share[perm],
    )


# --------------------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------------------


def hmm_observations(
    returns: pd.DataFrame,
    sectors: Mapping[str, str],
    *,
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Causal two-column observation frame for the HMM from daily log returns.

    Row ``t`` uses the ``window`` rows ending at ``t`` (trailing, inclusive): ``log_vol`` is the
    log of the cross-sectional mean of the per-asset sample standard deviations (``ddof=1``) and
    ``z_cross_corr`` is the mean Fisher z, ``arctanh`` of the pair correlation clipped to
    +-0.999, over all **cross-sector** pairs. ``log_vol`` is column 0 so that :func:`order_states`
    ranks states by volatility. ``NaN`` for the first ``window - 1`` rows, for any window that
    contains a ``NaN`` return, and for a window in which an asset is constant. ``sectors`` must
    map every column of ``returns`` to a sector label and contain at least two sectors.
    Overlapping windows make consecutive rows strongly autocorrelated (see the module docstring).
    """
    if not isinstance(returns, pd.DataFrame):
        raise RegimeHmmError(f"returns must be a DataFrame, got {type(returns)}")
    if not returns.index.is_monotonic_increasing or not returns.index.is_unique:
        raise RegimeHmmError("returns index must be strictly increasing and unique")
    if window < 10:
        raise RegimeHmmError(f"window must be >= 10, got {window}")
    columns = [str(c) for c in returns.columns]
    missing = [c for c in columns if c not in sectors]
    if missing:
        raise RegimeHmmError(f"no sector given for columns: {missing}")
    values = returns.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise RegimeHmmError("returns contain +-inf")
    n, m = values.shape
    names = np.array([sectors[c] for c in columns])
    if len(set(names.tolist())) < 2:
        raise RegimeHmmError("need at least two sectors for cross-sector pairs")
    iu = np.triu_indices(m, 1)
    cross = (names[:, None] != names[None, :])[iu]

    log_vol = np.full(n, np.nan)
    z_corr = np.full(n, np.nan)
    for t in range(window - 1, n):
        block = values[t - window + 1 : t + 1]
        if np.isnan(block).any():
            continue
        sd = block.std(axis=0, ddof=1)
        if not (sd > 0).all():
            continue
        log_vol[t] = log(float(sd.mean()))
        corr = np.corrcoef(block, rowvar=False)[iu][cross]
        z_corr[t] = float(np.arctanh(np.clip(corr, -_CORR_CLIP, _CORR_CLIP)).mean())
    return pd.DataFrame(
        {"log_vol": log_vol, "z_cross_corr": z_corr}, index=returns.index, dtype=float
    )


# --------------------------------------------------------------------------------------
# The causal driver
# --------------------------------------------------------------------------------------


def _min_separation(params: HmmParams, order_by: int) -> float:
    """Smallest gap between neighbouring (ordered) state means on the ordering column, in units
    of the pooled within-state standard deviation: ``|mu_{j+1} - mu_j| / sqrt((v_j + v_{j+1})/2)``.

    A diagnostic, not a test: values near zero mean the "states" are not distinguishable on the
    column they are ordered by, but large values are **not** evidence of real regimes (the module
    docstring gives the measured values on regime-free noise).
    """
    mu = params.means[:, order_by]
    var = params.covars[:, order_by, order_by]
    gaps = np.abs(np.diff(mu)) / np.sqrt(0.5 * (var[:-1] + var[1:]))
    return float(gaps.min())


def refit_positions(n_valid: int, min_obs: int, refit_every: int) -> tuple[int, ...]:
    """Zero-based positions (among the valid rows) at which the HMM is (re)fitted.

    The first fit uses rows ``0 .. min_obs - 1``; later fits follow every ``refit_every`` rows.
    The schedule depends on ``min_obs`` and ``refit_every`` only, never on how many rows exist
    after a position, which is what makes the output causal.
    """
    if min_obs < 1 or refit_every < 1:
        raise RegimeHmmError("min_obs and refit_every must be >= 1")
    return tuple(range(min_obs - 1, n_valid, refit_every))


def forward_filtered_hmm(
    obs: pd.DataFrame,
    n_states: int = DEFAULT_N_STATES,
    *,
    min_obs: int = DEFAULT_MIN_OBS,
    refit_every: int = DEFAULT_REFIT_EVERY,
    n_init: int = 5,
    max_iter: int = 100,
    tol: float = 1e-6,
    min_covar: float = 1e-3,
    order_by: int = 0,
    seed: int = 0,
    warm_start: bool = True,
) -> HmmRegimes:
    """Causal HMM regimes: filtered state probabilities with expanding-window refits.

    ``obs`` is a frame of causal observations on an increasing, duplicate-free index; ``NaN`` may
    appear only in a leading block (warm-up), never later. With ``V`` valid rows counted from the
    first complete one, the model is fitted on the first ``min_obs`` valid rows and refitted every
    ``refit_every`` rows on all valid rows seen so far (:func:`refit_positions`). The probability
    at row ``t`` is the forward-filtered ``P(S_t | x_1..x_t)`` under the parameters of the last
    refit at or before ``t`` (the filter is re-run from the first valid row with those parameters,
    so the probabilities inside a segment are mutually consistent). Rows before the first fit are
    ``NaN``; if there are fewer than ``min_obs`` valid rows the whole output is ``NaN`` and
    ``fits`` is empty. ``warm_start`` starts each refit from the previous parameters. See the
    module docstring for the label ordering and the failure modes.
    """
    if not isinstance(obs, pd.DataFrame):
        raise RegimeHmmError(f"obs must be a DataFrame, got {type(obs)}")
    if not obs.index.is_monotonic_increasing or not obs.index.is_unique:
        raise RegimeHmmError("obs index must be strictly increasing and unique")
    if n_states < 2:
        raise RegimeHmmError(f"n_states must be >= 2, got {n_states}")
    if refit_every < 1:
        raise RegimeHmmError(f"refit_every must be >= 1, got {refit_every}")
    if min_obs < 30 * n_states:
        raise RegimeHmmError(f"min_obs must be >= {30 * n_states} for {n_states} states")
    values = obs.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise RegimeHmmError("obs contains +-inf")
    if not 0 <= order_by < values.shape[1]:
        raise RegimeHmmError(f"order_by must be in [0, {values.shape[1]}), got {order_by}")
    complete = ~np.isnan(values).any(axis=1)
    where = np.flatnonzero(complete)
    first = int(where[0]) if where.size else len(values)
    if not complete[first:].all():
        raise RegimeHmmError("obs has NaN after its first complete row; only leading NaN allowed")

    cols = [f"p_{k}" for k in range(n_states)]
    probs = np.full((len(values), n_states), np.nan)
    x = values[first:]
    n_valid = len(x)
    fit_rows: list[dict[str, float | int | bool]] = []
    fit_index: list[pd.Timestamp] = []
    last: HmmParams | None = None

    positions = refit_positions(n_valid, min_obs, refit_every)
    for k, pos in enumerate(positions):
        fit = fit_gaussian_hmm(
            x[: pos + 1],
            n_states,
            init=last if (warm_start and last is not None) else None,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            min_covar=min_covar,
            order_by=order_by,
            seed=int(np.random.default_rng([seed, k]).integers(0, 2**31 - 1)),
        )
        last = fit.params
        seg_end = (positions[k + 1] - 1) if k + 1 < len(positions) else n_valid - 1
        filtered, _ = forward_probabilities(x[: seg_end + 1], fit.params)
        probs[first + pos : first + seg_end + 1] = filtered[pos : seg_end + 1]

        row: dict[str, float | int | bool] = {
            "n_fit_obs": pos + 1,
            "loglik": fit.loglik,
            "n_iter": fit.n_iter,
            "converged": fit.converged,
            "min_state_share": float(fit.state_share.min()),
        }
        for j in range(n_states):
            row[f"p_stay_{j}"] = float(fit.params.transmat[j, j])
            row[f"order_mean_{j}"] = float(fit.params.means[j, order_by])
        row["min_separation"] = _min_separation(fit.params, order_by)
        fit_rows.append(row)
        fit_index.append(pd.Timestamp(obs.index[first + pos]))

    prob_frame = pd.DataFrame(probs, index=obs.index, columns=cols)
    codes = np.full(len(values), np.nan)
    have = ~np.isnan(probs[:, 0])
    codes[have] = np.argmax(probs[have], axis=1).astype(float)
    labels = pd.Series(codes, index=obs.index, name="hmm_state", dtype=float)
    fits = pd.DataFrame(fit_rows, index=pd.Index(fit_index, name=obs.index.name))
    return HmmRegimes(prob_frame, labels, fits, last)


def hmm_summary_lines(result: HmmRegimes) -> list[str]:
    """Plain-text diagnostics of the refits: how many, convergence, persistence, state means."""
    fits = result.fits
    if fits.empty:
        return ["no HMM fit (fewer valid rows than min_obs)"]
    n_states = result.probs.shape[1]
    first, last = pd.Timestamp(fits.index[0]), pd.Timestamp(fits.index[-1])
    converged = fits["converged"].to_numpy(dtype=bool)
    iterations = fits["n_iter"].to_numpy(dtype=float)
    shares = fits["min_state_share"].to_numpy(dtype=float)
    sep = fits["min_separation"].to_numpy(dtype=float)
    lines = [
        f"refits: {len(fits)}, first {first.date()}, last {last.date()}, "
        f"converged {int(converged.sum())}/{len(fits)}, "
        f"mean iterations {iterations.mean():.1f}",
        f"smallest state share in any fit: {shares.min():.3f}",
        f"neighbouring-state separation on the ordering column (pooled-sd units), "
        f"min/median/max over refits: {sep.min():.2f}/{np.median(sep):.2f}/{sep.max():.2f}",
    ]
    for j in range(n_states):
        stay = fits[f"p_stay_{j}"].to_numpy(dtype=float)
        mean_last = fits[f"order_mean_{j}"].to_numpy(dtype=float)[-1]
        lines.append(
            f"state {j}: p_stay min/median/max {stay.min():.3f}/{np.median(stay):.3f}/"
            f"{stay.max():.3f}; ordering-column mean at last fit {mean_last:.3f}"
        )
    return lines
