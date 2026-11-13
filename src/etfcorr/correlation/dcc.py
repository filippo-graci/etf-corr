"""Step 4.4: dynamic conditional correlation (DCC, asymmetric DCC), causal expanding windows.

Question served: Phase 4 time-varying correlation. H2 (``config/experiment.yaml``) is tested on VIX
terciles (steps 4.1/4.2, no model fitted); this module supplies the *model-based* time-varying
correlation of ``project_plan.md`` section 6.4 for (a) an exploratory cross-check of the H2 reading,
(b) the DCC baseline of the Phase 5 model ladder (target T1, next-month average cross-sector
correlation) and (c) the DCC covariance forecast of H5 (Phase 8). It is **not** part of the
pre-registered H2 test and does not change it.

**Model (Engle 2002; asymmetric term Cappiello-Engle-Sheppard 2006).** Each return series follows a
univariate GARCH(1,1) with a constant mean,

    r_it = mu_i + e_it,  e_it = sqrt(h_it) z_it,
    h_it = omega_i + alpha_i e_i,t-1^2 + beta_i h_i,t-1,

with Gaussian or unit-variance Student-t innovations. The standardised residuals ``z_t`` have
conditional correlation ``R_t = diag(Q_t)^(-1/2) Q_t diag(Q_t)^(-1/2)`` where

    Q_{t+1} = (1 - a - b) Qbar - g Nbar + a z_t z_t' + b Q_t + g n_t n_t',   n_t = min(z_t, 0),

with ``g = 0`` for the symmetric DCC. ``Qbar = mean(z z')`` and ``Nbar = mean(n n')`` are sample
moments (correlation targeting), ``Q_0 = Qbar``. ``Q_t`` is known at ``t - 1``: ``R_t`` is the
correlation forecast for day ``t`` made at the close of ``t - 1``, and ``R_{t+1|t}``
(``corr[T]`` of a filter over ``T`` rows) is the one-step-ahead forecast made at the last row.
Stationarity/positive definiteness: ``a, b, g >= 0`` and ``a + b + delta g < 1`` with ``delta`` the
largest generalised eigenvalue of ``(Nbar, Qbar)``.

**Estimation: two-step quasi-maximum likelihood (Engle and Sheppard 2001, Engle 2002).** Step one
fits each GARCH(1,1) separately; step two maximises the correlation part of the Gaussian
likelihood,

    L2(a, b, g) = -1/2 sum_t [ log|R_t| + z_t' R_t^{-1} z_t - z_t' z_t ],

over ``(a, b, g)`` with ``z`` held fixed (a generated-regressor problem: no standard errors are
reported here; inference in this project is by out-of-sample comparison and the stationary
bootstrap, not by the two-step sandwich). The ``Q`` recursion is linear in ``(z z', n n')``, so it
is evaluated for the whole sample at once with ``scipy.signal.lfilter``, and ``R_t`` and its
log-determinant are batched Cholesky/solve calls; one likelihood evaluation costs a few tens of
milliseconds at N=25, T=3000. Optimisation is SLSQP (bounds plus the stationarity inequality) from
a few fixed starts, deterministic; no randomness anywhere in estimation.

**Why own code, and what is cross-checked.** ``project_plan.md`` 6.4 planned ``arch`` marginals plus
``pymgarch`` and "a ~100-line own DCC used only in tests". This step inverts that role, a stated
deviation: the causal driver needs fits whose parameters are frozen at each refit and a filter that
can be run forward on later rows with those parameters; the library API for doing that was neither
installable in the gate sandbox (PyPI unreachable) nor, for ``pymgarch``, documented in the parts
read (parameter attribute names are not specified). Own code is validated offline by brute-force
reference implementations, analytic cases, planted-truth recovery on simulated data and an
independent minimal DCC (``tests/unit/test_dcc_crosscheck.py``: plain loops, a different
parametrisation and optimiser, same likelihood and optimum), and, at Gate 2 on the user's machine,
against ``arch`` for the marginals (``tests/unit/test_dcc_arch.py``,
``scripts/dcc_crosscheck_4_4.py``). ``pymgarch`` is not used: not installable in the gate sandbox
and its interface was not read.

**Causality.** ``rolling_dcc`` fits on an expanding window: the first fit uses the first
``min_obs`` rows, then the model is refit every ``refit_every`` rows on everything seen so far. For
a row ``t`` in segment ``k`` the output is the one-step-ahead correlation forecast made at ``t`` by
the model of refit ``k`` (its GARCH parameters, mean, variance backcast, ``Qbar``, ``Nbar``,
``a, b, g`` all estimated on rows dated <= the refit date <= ``t``), with the filter re-run from
the first valid row. The refit schedule counts rows from the first valid observation, so it does
not depend on how much later data exists: the output at ``t`` is a pure function
``f(data_up_to_t) -> value_at_t`` (checked with the shared detectors in
``tests/leakage/test_dcc_no_lookahead.py``, which also prove they catch six leaky twins). The
variance backcast is the RiskMetrics-style exponentially weighted mean (decay 0.94) of the first 75
squared demeaned returns of the fitting window, a function of the first rows only.

**Forecasts.** ``forecast_correlation`` gives the standard h-step approximation
``E_T[Q_{T+h}] ~ Qbar + (a + b)^(h-1) (Q_{T+1} - Qbar)`` followed by normalisation, obtained by
replacing ``E[z z']`` with ``Q``; it is an approximation (Jensen), not the exact conditional mean of
``R_{T+h}`` (**[Inference]**; the size of the error is measured in
``scripts/dcc_calibration_4_4.py``). Only the symmetric model has a multi-step forecast here; for
the asymmetric model only ``h = 1`` (exact given the data) is supported.

**Known failure modes (state them wherever a result is used).** Local optima of the likelihood
(mitigated by restarts, not removed); boundary solutions (``alpha`` near 0 or ``alpha + beta`` near
1) are common in GARCH fits and legitimate; with a constant true correlation ``b`` is not
identified and ``a`` is close to 0; Engle's two-step estimator of ``Qbar`` is not consistent for
the correlation targeting in general (Aielli 2013, cDCC, **not implemented**); the plug-in forecast
ignores parameter uncertainty; large N makes the single (a, b) pair a strong restriction (all
pairs share one persistence); a fixed sample mean for ``mu``; no leverage term in the marginals
(a GJR/EGARCH marginal is a possible extension); returns with missing values inside the sample are
rejected, not filled.

**Citations (checked by search/fetch on 2026-10-02; the papers' *results* are not used).**
Engle, R. (2002), *Dynamic Conditional Correlation: A Simple Class of Multivariate Generalized
Autoregressive Conditional Heteroskedasticity Models*, Journal of Business & Economic Statistics
20(3), 339-350: existence and title confirmed by search listings; volume, pages and DOI come from
the standard bibliographic record and were **not** confirmed by a fetch (the RePEc and DOI pages
returned 404/403), the page range was CHECKED on Crossref on 2026-11-02. Engle, R. and Sheppard, K.
(2001), *Theoretical and Empirical Properties of Dynamic Conditional Correlation Multivariate
GARCH*, NBER Working Paper 8554 (October 2001): title, authors, date and number checked; the
abstract describes the two-step procedure (univariate GARCH, then a correlation estimator from the
transformed residuals) and applications up to 100 assets. Cappiello, L., Engle, R., Sheppard, K.
(2006), *Asymmetric Dynamics in the Correlations of Global Equity and Bond Returns*, Journal of
Financial Econometrics 4(4), 537-572, doi 10.1093/jjfinec/nbl005: checked (the abstract introduces
the AG-DCC model; the intercept ``-g Nbar`` and the ``delta`` stationarity condition used here
follow the standard statement of the model and are **not** verified against the paper's text).
Aielli, G. P. (2013), *Dynamic Conditional Correlation: On Properties and Estimation*, Journal of
Business & Economic Statistics 31(3), 282-299, doi 10.1080/07350015.2013.771027: checked; the
abstract says the original DCC large-system estimator can be inconsistent and proposes the corrected
cDCC (not implemented). The Gaussian QML property of GARCH (Bollerslev and Wooldridge 1992) is cited
from memory (record CHECKED on Crossref 2026-11-02: Econometric Reviews 11(2), 143-172), used only
as the reason Gaussian likelihood step one is acceptable. Library reference: ``arch``
(``arch_model``), used only in the Gate 2 cross-check.

Not adapted from third-party code (no licence flag): every recursion is implemented from the
equations above. Pure functions of their inputs: no I/O, no global state, no dependency on
``etfcorr.config`` or ``etfcorr.data``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log, pi
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln

FloatArray = npt.NDArray[np.float64]
Dist = Literal["normal", "t"]

DEFAULT_MIN_OBS = 504  # two years of daily rows before the first fit
DEFAULT_REFIT_EVERY = 63  # one quarter
BACKCAST_OBS = 75  # rows used for the variance backcast
BACKCAST_DECAY = 0.94  # RiskMetrics decay of the backcast weights
MIN_GARCH_OBS = 100
NU_BOUNDS = (2.05, 100.0)
MAX_PERSISTENCE = 1.0 - 1e-6  # alpha + beta (and a + b + delta g) stay below this
_PENALTY = 1e10  # objective value for numerically invalid parameter points


class DccError(ValueError):
    """Invalid input to, or a numerical failure of, a DCC / GARCH call."""


# --------------------------------------------------------------------------------------
# Containers
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GarchFit:
    """A fitted constant-mean GARCH(1,1) in the units of the returns it was fitted on.

    ``h0`` is the variance backcast used for the first conditional variance;
    ``nu`` is ``None`` for Gaussian innovations.
    """

    omega: float
    alpha: float
    beta: float
    nu: float | None
    mu: float
    h0: float
    loglik: float
    n_obs: int
    converged: bool
    dist: Dist

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def unconditional_variance(self) -> float:
        p = self.persistence
        return self.omega / (1.0 - p) if p < 1.0 else float("nan")


@dataclass(frozen=True)
class DccFit:
    """Second-step estimates: ``a, b, g`` and the sample moments they are tied to.

    ``loglik`` is the correlation part ``L2`` of the Gaussian likelihood at the optimum (not the
    full likelihood); ``delta`` is the largest generalised eigenvalue of ``(nbar, qbar)``.
    """

    a: float
    b: float
    g: float
    qbar: FloatArray
    nbar: FloatArray
    delta: float
    loglik: float
    n_obs: int
    converged: bool
    asymmetric: bool

    @property
    def persistence(self) -> float:
        """``a + b``: the geometric rate at which ``Q`` reverts to ``Qbar`` (symmetric model)."""
        return self.a + self.b


@dataclass(frozen=True)
class DccModel:
    """A full two-step model: one GARCH per column and the DCC second step."""

    columns: tuple[str, ...]
    garch: tuple[GarchFit, ...]
    dcc: DccFit
    dist: Dist


@dataclass(frozen=True)
class DccFiltered:
    """Filtered quantities for ``T`` rows under a fixed model.

    ``z`` and ``h`` are (T, N); ``q`` and ``corr`` are (T + 1, N, N): ``corr[t]`` is the
    correlation of day ``t`` conditional on rows before ``t`` and ``corr[T]`` is the one-step
    forecast for the day after the last row.
    """

    z: FloatArray
    h: FloatArray
    q: FloatArray
    corr: FloatArray


@dataclass(frozen=True)
class DccPath:
    """Output of :func:`rolling_dcc`.

    ``avg_corr``: DataFrame indexed like the input, one column ``h<k>`` per forecast horizon,
    the average correlation over ``pair_mask`` of ``R_{t+k|t}`` forecast at row ``t`` (``NaN``
    before the first fit). ``corr_next``: (T, N, N) one-step forecasts ``R_{t+1|t}`` (``NaN``
    before the first fit; ``None`` if ``keep_matrices=False``). ``fits``: one row per refit,
    indexed by the date of the last observation used. ``last_model``: the final model.
    ``var_next``: (T, N) one-step GARCH variance forecasts ``h_{t+1|t}`` under the same fits
    (kept with the matrices; with ``corr_next`` it gives the full DCC covariance forecast).
    """

    avg_corr: pd.DataFrame
    corr_next: FloatArray | None
    fits: pd.DataFrame
    last_model: DccModel | None
    columns: tuple[str, ...]
    var_next: FloatArray | None = None


@dataclass(frozen=True)
class DccSimulation:
    """Simulated returns together with the truth they were drawn from (tests and calibration).

    ``corr[t]`` is the true conditional correlation of row ``t`` given earlier rows; ``sigma2``
    the true conditional variances; ``z`` the true standardised shocks.
    """

    returns: pd.DataFrame
    corr: FloatArray
    sigma2: FloatArray
    z: FloatArray


# --------------------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------------------


def _check_vector(x: object, name: str, min_len: int = 2) -> FloatArray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise DccError(f"{name} must be 1-D, got shape {arr.shape}")
    if len(arr) < min_len:
        raise DccError(f"{name} needs at least {min_len} observations, got {len(arr)}")
    if not np.isfinite(arr).all():
        raise DccError(f"{name} must be finite (no NaN or +-inf)")
    return arr


def _check_matrix(x: object, name: str, min_rows: int = 2) -> FloatArray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise DccError(f"{name} must be 2-D (rows x series), got shape {arr.shape}")
    if arr.shape[0] < min_rows or arr.shape[1] < 1:
        raise DccError(f"{name} needs at least {min_rows} rows and 1 column, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise DccError(f"{name} must be finite (no NaN or +-inf)")
    return arr


def _check_dist(dist: str) -> Dist:
    if dist == "normal":
        return "normal"
    if dist == "t":
        return "t"
    raise DccError(f"dist must be 'normal' or 't', got {dist!r}")


# --------------------------------------------------------------------------------------
# Univariate GARCH(1,1)
# --------------------------------------------------------------------------------------


def garch_backcast(resid: FloatArray) -> float:
    """Exponentially weighted mean (decay 0.94) of the first 75 squared residuals.

    A function of the first ``min(75, n)`` rows only, so it is the same number whatever later rows
    exist. It is the RiskMetrics-style convention that ``arch`` documents as its default backcast
    (CHECKED 2026-11-02 in the installed ``arch`` source: ``tau = min(75, n)``, ``0.94**k``).
    """
    e = _check_vector(resid, "resid", min_len=1)
    tau = min(BACKCAST_OBS, len(e))
    w = BACKCAST_DECAY ** np.arange(tau)
    w = w / w.sum()
    value = float(np.sum(w * e[:tau] ** 2))
    if value <= 0.0:
        raise DccError("variance backcast is zero (the first rows of the residuals are all zero)")
    return value


def garch11_variance(
    resid: FloatArray, omega: float, alpha: float, beta: float, h0: float
) -> FloatArray:
    """Conditional variances ``h_1..h_T``: ``h_1 = omega + (alpha + beta) h0``,
    ``h_t = omega + alpha e_{t-1}^2 + beta h_{t-1}``. ``h_t`` uses residuals before ``t`` only."""
    e = _check_vector(resid, "resid", min_len=1)
    lagged = np.concatenate(([h0], e[:-1] ** 2))
    h, _ = lfilter([1.0], [1.0, -beta], omega + alpha * lagged, zi=[beta * h0])
    return np.asarray(h, dtype=float)


def garch11_loglik(
    resid: FloatArray,
    omega: float,
    alpha: float,
    beta: float,
    h0: float,
    dist: Dist = "normal",
    nu: float | None = None,
) -> float:
    """Log-likelihood of the residuals (all constants included) at the given parameters.

    ``dist='t'`` uses the unit-variance Student-t with ``nu`` degrees of freedom (``nu > 2``).
    """
    e = _check_vector(resid, "resid", min_len=1)
    h = garch11_variance(e, omega, alpha, beta, h0)
    if not np.all(h > 0.0):
        return -float("inf")
    if dist == "normal":
        return float(-0.5 * np.sum(log(2.0 * pi) + np.log(h) + e**2 / h))
    if nu is None or nu <= 2.0:
        raise DccError("a Student-t likelihood needs nu > 2")
    const = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * log(pi * (nu - 2.0))
    return float(
        len(e) * const
        - 0.5 * np.sum(np.log(h))
        - 0.5 * (nu + 1.0) * np.sum(np.log1p(e**2 / (h * (nu - 2.0))))
    )


def _garch_start_grid(warm: GarchFit | None, scale: float) -> list[tuple[float, float, float]]:
    """Starting (omega, alpha, beta) in the scaled units (unit sample variance)."""
    starts: list[tuple[float, float, float]] = []
    if warm is not None:
        starts.append((max(warm.omega / scale**2, 1e-8), warm.alpha, warm.beta))
    for alpha, beta in ((0.05, 0.90), (0.10, 0.80), (0.02, 0.97)):
        starts.append((1.0 - alpha - beta, alpha, beta))
    return starts


def fit_garch11(
    returns: FloatArray,
    *,
    dist: Dist = "t",
    demean: bool = True,
    start: GarchFit | None = None,
    n_starts: int | None = None,
) -> GarchFit:
    """Gaussian or Student-t quasi-maximum-likelihood fit of a constant-mean GARCH(1,1).

    The series is rescaled to unit variance before optimisation and the parameters are mapped
    back. Several fixed starting points are tried (plus ``start`` if given, which is the
    warm start of the causal driver); the best finite likelihood wins, preferring runs the
    optimiser reports as converged. ``n_starts`` caps how many starting points are used (default
    all). Constraints: ``omega > 0``, ``alpha, beta >= 0``, ``alpha + beta < 1``,
    ``2.05 <= nu <= 100``. Deterministic.
    """
    dist = _check_dist(dist)
    r = _check_vector(returns, "returns", min_len=MIN_GARCH_OBS)
    mu = float(r.mean()) if demean else 0.0
    e = r - mu
    scale = float(e.std())
    if scale <= 0.0:
        raise DccError("returns are constant: a GARCH model is not identified")
    y = e / scale
    n = len(y)
    h0_scaled = garch_backcast(y)
    use_t = dist == "t"

    def objective(theta: FloatArray) -> float:
        omega, alpha, beta = float(theta[0]), float(theta[1]), float(theta[2])
        nu = float(theta[3]) if use_t else None
        ll = garch11_loglik(y, omega, alpha, beta, h0_scaled, dist, nu)
        return _PENALTY if not np.isfinite(ll) else -ll / n

    bounds = [(1e-10, 50.0), (0.0, 1.0), (0.0, 1.0)] + ([NU_BOUNDS] if use_t else [])
    constraints = [{"type": "ineq", "fun": lambda th: MAX_PERSISTENCE - th[1] - th[2]}]
    grid = _garch_start_grid(start, scale)
    if n_starts is not None:
        grid = grid[: max(n_starts, 1)]

    best_ok: tuple[float, FloatArray] | None = None
    best_any: tuple[float, FloatArray] | None = None
    for omega0, alpha0, beta0 in grid:
        nu0 = min(max(start.nu if start is not None and start.nu else 8.0, NU_BOUNDS[0]), 100.0)
        x0 = np.array([omega0, alpha0, beta0] + ([nu0] if use_t else []), dtype=float)
        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-10},
        )
        if not np.isfinite(res.fun) or res.fun >= _PENALTY:
            continue
        item = (float(res.fun), np.asarray(res.x, dtype=float))
        if best_any is None or item[0] < best_any[0]:
            best_any = item
        if res.success and (best_ok is None or item[0] < best_ok[0]):
            best_ok = item
    chosen = best_ok if best_ok is not None else best_any
    if chosen is None:
        raise DccError("GARCH(1,1) optimisation failed from every starting point")
    theta = chosen[1]
    omega_s, alpha, beta = float(theta[0]), float(theta[1]), float(theta[2])
    nu_hat = float(theta[3]) if use_t else None
    omega = omega_s * scale**2
    h0 = h0_scaled * scale**2
    loglik = garch11_loglik(e, omega, alpha, beta, h0, dist, nu_hat)
    return GarchFit(
        omega=omega,
        alpha=alpha,
        beta=beta,
        nu=nu_hat,
        mu=mu,
        h0=h0,
        loglik=loglik,
        n_obs=n,
        converged=best_ok is not None,
        dist=dist,
    )


def standardized_residuals(returns: FloatArray, fit: GarchFit) -> tuple[FloatArray, FloatArray]:
    """``(z, h)``: standardised residuals and conditional variances of ``returns`` under ``fit``.

    Row ``t`` uses the parameters of ``fit`` and returns up to ``t - 1`` for ``h_t``.
    """
    r = _check_vector(returns, "returns", min_len=1)
    e = r - fit.mu
    h = garch11_variance(e, fit.omega, fit.alpha, fit.beta, fit.h0)
    return e / np.sqrt(h), h


# --------------------------------------------------------------------------------------
# DCC second step
# --------------------------------------------------------------------------------------


def _outer_rows(x: FloatArray) -> FloatArray:
    """Row-wise outer products of a (T, N) array, flattened to (T, N * N)."""
    t, n = x.shape
    return (x[:, :, None] * x[:, None, :]).reshape(t, n * n)


def _normalise(q: FloatArray) -> FloatArray:
    """``diag(Q)^(-1/2) Q diag(Q)^(-1/2)`` for a stack (..., N, N), diagonal set to exactly 1."""
    d = np.sqrt(np.diagonal(q, axis1=-2, axis2=-1))
    r = q / (d[..., :, None] * d[..., None, :])
    n = r.shape[-1]
    idx = np.arange(n)
    r[..., idx, idx] = 1.0
    return np.asarray(r, dtype=float)


def _q_flat(
    zz: FloatArray,
    nn: FloatArray | None,
    qbar_flat: FloatArray,
    nbar_flat: FloatArray | None,
    a: float,
    b: float,
    g: float,
) -> FloatArray:
    """``Q_0 .. Q_T`` as a (T + 1, N * N) array from the precomputed outer products."""
    drive = (1.0 - a - b) * qbar_flat[None, :] + a * zz
    if g != 0.0:
        if nn is None or nbar_flat is None:
            raise DccError("an asymmetric model (g != 0) needs nbar")
        drive = drive - g * nbar_flat[None, :] + g * nn
    path, _ = lfilter([1.0], [1.0, -b], drive, axis=0, zi=(b * qbar_flat)[None, :])
    return np.concatenate([qbar_flat[None, :], np.asarray(path, dtype=float)], axis=0)


def dcc_q_path(
    z: FloatArray,
    qbar: FloatArray,
    nbar: FloatArray | None,
    a: float,
    b: float,
    g: float = 0.0,
) -> FloatArray:
    """The matrices ``Q_0 .. Q_T`` (shape (T + 1, N, N)) for standardised residuals ``z``.

    ``Q_0 = Qbar`` and ``Q_{t+1} = (1 - a - b) Qbar - g Nbar + a z_t z_t' + b Q_t + g n_t n_t'``.
    """
    z = _check_matrix(z, "z", min_rows=1)
    t, n = z.shape
    nn = _outer_rows(np.minimum(z, 0.0)) if g != 0.0 else None
    nbar_flat = None if nbar is None else np.asarray(nbar, dtype=float).reshape(n * n)
    q = _q_flat(
        _outer_rows(z), nn, np.asarray(qbar, dtype=float).reshape(n * n), nbar_flat, a, b, g
    )
    return q.reshape(t + 1, n, n)


def dcc_correlations(q: FloatArray) -> FloatArray:
    """Normalise a stack of ``Q`` matrices to correlation matrices (diagonal exactly 1)."""
    return _normalise(np.asarray(q, dtype=float).copy())


def _loglik_flat(
    z: FloatArray,
    zz: FloatArray,
    nn: FloatArray | None,
    qbar_flat: FloatArray,
    nbar_flat: FloatArray | None,
    a: float,
    b: float,
    g: float,
) -> float:
    t, n = z.shape
    q = _q_flat(zz, nn, qbar_flat, nbar_flat, a, b, g)[:-1].reshape(t, n, n)
    d = np.diagonal(q, axis1=-2, axis2=-1)
    if not np.all(np.isfinite(q)) or np.any(d <= 0.0):
        return -float("inf")
    r = _normalise(q)
    sign, logdet = np.linalg.slogdet(r)
    if np.any(sign <= 0.0) or not np.all(np.isfinite(logdet)):
        return -float("inf")
    sol = np.linalg.solve(r, z[:, :, None])[:, :, 0]
    quad = np.sum(z * sol, axis=1)
    value = -0.5 * float(np.sum(logdet + quad - np.sum(z * z, axis=1)))
    return value if np.isfinite(value) else -float("inf")


def dcc_loglik(
    z: FloatArray,
    qbar: FloatArray,
    nbar: FloatArray | None,
    a: float,
    b: float,
    g: float = 0.0,
) -> float:
    """Correlation part ``L2`` of the Gaussian log-likelihood at ``(a, b, g)``.

    ``-1/2 sum_t [log|R_t| + z_t' R_t^{-1} z_t - z_t' z_t]`` over the ``T`` rows of ``z``. Returns
    ``-inf`` where some ``R_t`` is not positive definite or the values are not finite.
    """
    z = _check_matrix(z, "z", min_rows=1)
    n = z.shape[1]
    nn = _outer_rows(np.minimum(z, 0.0)) if g != 0.0 else None
    nbar_flat = None if nbar is None else np.asarray(nbar, dtype=float).reshape(n * n)
    qbar_flat = np.asarray(qbar, dtype=float).reshape(n * n)
    return _loglik_flat(z, _outer_rows(z), nn, qbar_flat, nbar_flat, a, b, g)


def _delta(qbar: FloatArray, nbar: FloatArray) -> float:
    """Largest generalised eigenvalue of ``(nbar, qbar)`` (the asymmetric stationarity scale)."""
    vals = eigh(nbar, qbar, eigvals_only=True)
    return float(np.max(vals))


def _stationarity_slack(a: float, b: float, g: float, delta: float) -> float:
    """Distance of ``a + b + delta g`` below the persistence cap (positive when feasible)."""
    return MAX_PERSISTENCE - a - b - delta * g


def fit_dcc(
    z: FloatArray,
    *,
    asymmetric: bool = False,
    start: DccFit | None = None,
) -> DccFit:
    """Second-step QML estimate of ``(a, b)`` (and ``g`` if ``asymmetric``) for residuals ``z``.

    ``Qbar`` and ``Nbar`` are the sample moments of ``z z'`` and ``n n'`` (``n = min(z, 0)``);
    ``T > N`` is required so that ``Qbar`` is positive definite. SLSQP with ``a, b, g >= 0`` and
    ``a + b + delta g < 1``, from fixed starts (plus ``start``'s estimates as a warm start).
    ``converged`` is the optimiser's success flag for the reported point.
    """
    z = _check_matrix(z, "z", min_rows=2)
    t, n = z.shape
    if t <= n:
        raise DccError(f"need more rows than series for a non-singular Qbar (T={t}, N={n})")
    qbar = _outer_rows(z).mean(axis=0).reshape(n, n)
    qbar = 0.5 * (qbar + qbar.T)
    if np.linalg.eigvalsh(qbar).min() <= 1e-10:
        raise DccError("Qbar is not positive definite (collinear series)")
    nbar = _outer_rows(np.minimum(z, 0.0)).mean(axis=0).reshape(n, n)
    nbar = 0.5 * (nbar + nbar.T)
    delta = _delta(qbar, nbar)

    k = 3 if asymmetric else 2

    def split(x: FloatArray) -> tuple[float, float, float]:
        return float(x[0]), float(x[1]), float(x[2]) if asymmetric else 0.0

    zz = _outer_rows(z)
    nn = _outer_rows(np.minimum(z, 0.0))
    qbar_flat = qbar.reshape(n * n)
    nbar_flat = nbar.reshape(n * n)

    def objective(x: FloatArray) -> float:
        a, b, g = split(x)
        ll = _loglik_flat(z, zz, nn, qbar_flat, nbar_flat, a, b, g)
        return _PENALTY if not np.isfinite(ll) else -ll / t

    def slack(x: FloatArray) -> float:
        a, b, g = split(x)
        return _stationarity_slack(a, b, g, delta)

    g_cap = max(MAX_PERSISTENCE / delta, 1e-3) if delta > 0.0 else 1.0
    bounds = [(0.0, 1.0), (0.0, 1.0)] + ([(0.0, min(g_cap, 1.0))] if asymmetric else [])
    constraints = [{"type": "ineq", "fun": slack}]
    starts: list[list[float]] = []
    if start is not None and start.asymmetric == asymmetric:
        starts.append([start.a, start.b] + ([start.g] if asymmetric else []))
    for a0, b0 in ((0.03, 0.94), (0.05, 0.90), (0.01, 0.98)):
        starts.append([a0, b0] + ([0.01] if asymmetric else []))

    best_ok: tuple[float, FloatArray] | None = None
    best_any: tuple[float, FloatArray] | None = None
    for x0 in starts:
        x0_arr = np.array(x0[:k], dtype=float)
        if slack(x0_arr) <= 0.0:
            continue
        res = minimize(
            objective,
            x0_arr,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-10},
        )
        x_res = np.asarray(res.x, dtype=float)
        if not np.isfinite(res.fun) or res.fun >= _PENALTY or slack(x_res) < -1e-9:
            continue
        item = (float(res.fun), x_res)
        if best_any is None or item[0] < best_any[0]:
            best_any = item
        if res.success and (best_ok is None or item[0] < best_ok[0]):
            best_ok = item
    chosen = best_ok if best_ok is not None else best_any
    if chosen is None:
        raise DccError("DCC optimisation failed from every starting point")
    a, b, g = split(chosen[1])
    return DccFit(
        a=a,
        b=b,
        g=g,
        qbar=qbar,
        nbar=nbar,
        delta=delta,
        loglik=dcc_loglik(z, qbar, nbar, a, b, g),
        n_obs=t,
        converged=best_ok is not None,
        asymmetric=asymmetric,
    )


# --------------------------------------------------------------------------------------
# Full two-step model, filtering, forecasting
# --------------------------------------------------------------------------------------


def fit_dcc_model(
    returns: FloatArray,
    columns: Sequence[str] | None = None,
    *,
    dist: Dist = "t",
    asymmetric: bool = False,
    start: DccModel | None = None,
    n_starts: int | None = None,
) -> DccModel:
    """Two-step fit on a (T, N) array of returns with no missing values."""
    dist = _check_dist(dist)
    x = _check_matrix(returns, "returns", min_rows=MIN_GARCH_OBS)
    n = x.shape[1]
    names = tuple(f"x{i}" for i in range(n)) if columns is None else tuple(str(c) for c in columns)
    if len(names) != n:
        raise DccError(f"{len(names)} column names for {n} series")
    garch = tuple(
        fit_garch11(
            x[:, i], dist=dist, start=None if start is None else start.garch[i], n_starts=n_starts
        )
        for i in range(n)
    )
    z = np.column_stack([standardized_residuals(x[:, i], garch[i])[0] for i in range(n)])
    dcc = fit_dcc(z, asymmetric=asymmetric, start=None if start is None else start.dcc)
    return DccModel(columns=names, garch=garch, dcc=dcc, dist=dist)


def filter_dcc(model: DccModel, returns: FloatArray) -> DccFiltered:
    """Run the fixed ``model`` forward over (T, N) ``returns``; nothing is re-estimated."""
    x = _check_matrix(returns, "returns", min_rows=1)
    if x.shape[1] != len(model.columns):
        raise DccError(f"returns have {x.shape[1]} columns, the model {len(model.columns)}")
    pairs = [standardized_residuals(x[:, i], model.garch[i]) for i in range(x.shape[1])]
    z = np.column_stack([p[0] for p in pairs])
    h = np.column_stack([p[1] for p in pairs])
    d = model.dcc
    q = dcc_q_path(z, d.qbar, d.nbar, d.a, d.b, d.g)
    return DccFiltered(z=z, h=h, q=q, corr=dcc_correlations(q))


def forecast_correlation(
    q_next: FloatArray, qbar: FloatArray, a: float, b: float, horizon: int
) -> FloatArray:
    """Approximate ``R_{T+h|T}`` from ``Q_{T+1}`` (symmetric DCC).

    ``E_T[Q_{T+h}] ~ Qbar + (a + b)^(h-1) (Q_{T+1} - Qbar)``, then normalised. ``q_next`` may be a
    stack (..., N, N). ``horizon = 1`` returns the normalised ``Q_{T+1}`` exactly.
    """
    if horizon < 1:
        raise DccError(f"horizon must be >= 1, got {horizon}")
    q1 = np.asarray(q_next, dtype=float)
    qb = np.asarray(qbar, dtype=float)
    q_h = qb + (a + b) ** (horizon - 1) * (q1 - qb)
    return _normalise(np.array(q_h, dtype=float))


def average_correlation(corr: FloatArray, mask: npt.NDArray[np.bool_] | None = None) -> FloatArray:
    """Mean of the off-diagonal entries of ``corr`` (..., N, N) selected by ``mask`` (N, N).

    ``mask`` must be symmetric with a false diagonal; ``None`` selects every off-diagonal pair.
    """
    c = np.asarray(corr, dtype=float)
    n = c.shape[-1]
    if c.shape[-2] != n:
        raise DccError("corr must be square in its last two dimensions")
    if mask is None:
        m = ~np.eye(n, dtype=bool)
    else:
        m = np.asarray(mask, dtype=bool)
        if m.shape != (n, n):
            raise DccError(f"mask has shape {m.shape}, expected {(n, n)}")
        if m.diagonal().any() or not np.array_equal(m, m.T):
            raise DccError("mask must be symmetric with a false diagonal")
        if not m.any():
            raise DccError("mask selects no pair")
    return np.asarray(c[..., m].mean(axis=-1), dtype=float)


def cross_sector_mask(columns: Sequence[str], sectors: Mapping[str, str]) -> npt.NDArray[np.bool_]:
    """(N, N) mask of pairs whose two members are in different sectors."""
    missing = [c for c in columns if c not in sectors]
    if missing:
        raise DccError(f"no sector for: {missing}")
    labels = np.array([sectors[c] for c in columns])
    return np.asarray(labels[:, None] != labels[None, :], dtype=bool)


# --------------------------------------------------------------------------------------
# Causal expanding-window driver
# --------------------------------------------------------------------------------------


def _fit_row(model: DccModel, n_train: int) -> dict[str, float | bool]:
    garch = model.garch
    nu = [g.nu for g in garch if g.nu is not None]
    return {
        "n_train": float(n_train),
        "a": model.dcc.a,
        "b": model.dcc.b,
        "g": model.dcc.g,
        "a_plus_b": model.dcc.a + model.dcc.b,
        "dcc_loglik": model.dcc.loglik,
        "dcc_converged": model.dcc.converged,
        "garch_not_converged": float(sum(not g.converged for g in garch)),
        "garch_persistence_median": float(np.median([g.persistence for g in garch])),
        "garch_persistence_max": float(np.max([g.persistence for g in garch])),
        "nu_median": float(np.median(nu)) if nu else float("nan"),
    }


def rolling_dcc(
    returns: pd.DataFrame,
    *,
    dist: Dist = "t",
    asymmetric: bool = False,
    min_obs: int = DEFAULT_MIN_OBS,
    refit_every: int = DEFAULT_REFIT_EVERY,
    pair_mask: npt.NDArray[np.bool_] | None = None,
    horizons: Sequence[int] = (1,),
    warm_start: bool = True,
    keep_matrices: bool = True,
) -> DccPath:
    """Causal DCC: expanding-window refits, one-step (and h-step) correlation forecasts.

    The value dated ``t`` is the forecast of the correlation matrix of day ``t + h`` made with
    information dated <= ``t`` only (see the module docstring). Leading rows with a missing value
    are skipped (output ``NaN``); a missing value after the first complete row is an error. With
    fewer than ``min_obs`` complete rows the whole output is ``NaN`` (no fit yet), no fits table
    rows and ``last_model=None``, so the function is defined at every sample length.
    ``horizons`` other than ``(1,)`` need the symmetric model.
    """
    dist = _check_dist(dist)
    if min_obs < MIN_GARCH_OBS or refit_every < 1:
        raise DccError(f"need min_obs >= {MIN_GARCH_OBS} and refit_every >= 1")
    hs = tuple(int(h) for h in horizons)
    if not hs or any(h < 1 for h in hs):
        raise DccError(f"horizons must be positive integers, got {tuple(horizons)}")
    if asymmetric and any(h > 1 for h in hs):
        raise DccError("multi-step forecasts are implemented for the symmetric DCC only")
    if returns.shape[1] < 2:
        raise DccError("need at least two series")
    x_all = returns.to_numpy(dtype=float)
    names = tuple(str(c) for c in returns.columns)
    n = len(names)
    if pair_mask is not None:
        average_correlation(np.eye(n), pair_mask)  # validates the mask
    total = len(returns)
    columns = [f"h{h}" for h in hs]
    complete = np.isfinite(x_all).all(axis=1)
    first = int(np.argmax(complete)) if complete.any() else total
    if not complete[first:].all():
        raise DccError("missing values after the first complete row are not supported")
    n_rows = total - first
    if n_rows < min_obs:  # no fit is possible yet: an all-NaN path, like the HMM driver
        empty = pd.DataFrame(
            np.full((total, len(hs)), np.nan), index=returns.index, columns=columns
        )
        mats0 = np.full((total, n, n), np.nan) if keep_matrices else None
        vars0 = np.full((total, n), np.nan) if keep_matrices else None
        no_fits = pd.DataFrame({"n_train": []}, index=pd.Index([], name="fit_end"))
        return DccPath(
            avg_corr=empty,
            corr_next=mats0,
            fits=no_fits,
            last_model=None,
            columns=names,
            var_next=vars0,
        )
    x = x_all[first:]

    ends = list(range(min_obs - 1, n_rows, refit_every))
    out = np.full((total, len(hs)), np.nan)
    mats = np.full((total, n, n), np.nan) if keep_matrices else None
    variances = np.full((total, n), np.nan) if keep_matrices else None
    fit_rows: list[dict[str, float | bool]] = []
    fit_dates: list[pd.Timestamp] = []
    model: DccModel | None = None
    for k, end in enumerate(ends):
        seg_end = (ends[k + 1] - 1) if k + 1 < len(ends) else n_rows - 1
        model = fit_dcc_model(
            x[: end + 1],
            names,
            dist=dist,
            asymmetric=asymmetric,
            start=model if warm_start else None,
            n_starts=2 if (warm_start and model is not None) else None,
        )
        flt = filter_dcc(model, x[: seg_end + 1])
        seg = slice(end, seg_end + 1)
        q_next = flt.q[end + 1 : seg_end + 2]
        for j, h in enumerate(hs):
            if h == 1:
                corr_h = flt.corr[end + 1 : seg_end + 2]
            else:
                corr_h = forecast_correlation(q_next, model.dcc.qbar, model.dcc.a, model.dcc.b, h)
            out[first + seg.start : first + seg.stop, j] = average_correlation(corr_h, pair_mask)
        if mats is not None:
            mats[first + seg.start : first + seg.stop] = flt.corr[end + 1 : seg_end + 2]
        if variances is not None:  # h_{t+1} = omega + alpha e_t^2 + beta h_t, per asset
            g = model.garch
            e2 = (x[seg] - np.array([f.mu for f in g])) ** 2
            variances[first + seg.start : first + seg.stop] = (
                np.array([f.omega for f in g])
                + np.array([f.alpha for f in g]) * e2
                + np.array([f.beta for f in g]) * flt.h[seg]
            )
        date = pd.Timestamp(returns.index[first + end])
        fit_dates.append(date)
        fit_rows.append(_fit_row(model, end + 1))
    avg = pd.DataFrame(out, index=returns.index, columns=columns)
    fits = pd.DataFrame(fit_rows, index=pd.Index(fit_dates, name="fit_end"))
    return DccPath(
        avg_corr=avg,
        corr_next=mats,
        fits=fits,
        last_model=model,
        columns=names,
        var_next=variances,
    )


def dcc_summary_lines(path: DccPath) -> list[str]:
    """Plain-text diagnostics of a :func:`rolling_dcc` run (fit counts and parameter ranges)."""
    f = path.fits
    if len(f) == 0:
        return ["no DCC fit (fewer complete rows than min_obs)"]

    def rng(col: str) -> str:
        v = f[col].to_numpy(dtype=float)
        return f"min/median/max {v.min():.4f}/{np.median(v):.4f}/{v.max():.4f}"

    lines = [
        f"DCC refits: {len(f)}, training rows "
        f"{int(f['n_train'].min())} to {int(f['n_train'].max())}",
        f"a {rng('a')}; b {rng('b')}; a+b {rng('a_plus_b')}",
    ]
    if bool((f["g"] > 0).any()):
        lines.append(f"g {rng('g')}")
    nu = f["nu_median"].to_numpy(dtype=float)
    if np.isfinite(nu).any():
        lines.append(
            f"median Student-t nu per refit: min/max {np.nanmin(nu):.2f}/{np.nanmax(nu):.2f}"
        )
    lines.append(
        f"GARCH persistence (per-refit median) {rng('garch_persistence_median')}; largest single "
        f"persistence {float(f['garch_persistence_max'].max()):.4f}"
    )
    lines.append(
        f"not converged: DCC step in {int((~f['dcc_converged'].astype(bool)).sum())} refits, "
        f"GARCH fits {int(f['garch_not_converged'].sum())} (summed over refits)"
    )
    return lines


# --------------------------------------------------------------------------------------
# Simulation (tests and calibration only)
# --------------------------------------------------------------------------------------


def gaussian_negative_part_moment(rho: FloatArray) -> FloatArray:
    """``E[min(z1, 0) min(z2, 0)]`` for a standard bivariate normal with correlation ``rho``.

    Closed form ``(sqrt(1 - rho^2) + rho (pi - arccos rho)) / (2 pi)`` (orthant probabilities);
    equals 1/(2 pi) at 0, 1/2 at 1 and 0 at -1. Used to give a simulated asymmetric process a
    self-consistent intercept; the numerical check against Monte Carlo is in the tests.
    """
    r = np.clip(np.asarray(rho, dtype=float), -1.0, 1.0)
    return np.asarray((np.sqrt(1.0 - r**2) + r * (pi - np.arccos(r))) / (2.0 * pi), dtype=float)


def simulate_dcc(
    n_obs: int,
    rbar: FloatArray,
    *,
    omega: float | Sequence[float] = 2e-6,
    alpha: float | Sequence[float] = 0.06,
    beta: float | Sequence[float] = 0.92,
    a: float = 0.03,
    b: float = 0.95,
    g: float = 0.0,
    nu: float | None = None,
    seed: int = 0,
    burn: int = 300,
    q0: FloatArray | None = None,
    h0: FloatArray | None = None,
    start: str = "2016-01-04",
) -> DccSimulation:
    """Simulate ``n_obs`` rows from a DCC-GARCH with the intercept targeted at ``rbar``.

    Innovations are Gaussian, or unit-variance multivariate Student-t with ``nu`` degrees of
    freedom. For ``g > 0`` the intercept uses ``Nbar`` from the Gaussian closed form
    (:func:`gaussian_negative_part_moment`), so the process is only approximately
    mean-consistent with the estimator's sample-moment targeting. ``q0``/``h0`` set the initial
    state (default ``rbar`` and the unconditional variances) and ``burn=0`` keeps the first row.
    """
    rb = _check_matrix(rbar, "rbar", min_rows=2)
    n = rb.shape[0]
    if rb.shape != (n, n) or not np.allclose(rb, rb.T):
        raise DccError("rbar must be a symmetric correlation matrix")
    if n_obs < 1 or burn < 0:
        raise DccError("n_obs must be >= 1 and burn >= 0")
    om = np.broadcast_to(np.asarray(omega, dtype=float), (n,))
    al = np.broadcast_to(np.asarray(alpha, dtype=float), (n,))
    be = np.broadcast_to(np.asarray(beta, dtype=float), (n,))
    if np.any(al + be >= 1.0) or np.any(om <= 0.0):
        raise DccError("need omega > 0 and alpha + beta < 1 for every series")
    if a < 0 or b < 0 or g < 0 or a + b >= 1.0:
        raise DccError("need a, b, g >= 0 and a + b < 1")
    nbar = gaussian_negative_part_moment(rb)
    if g > 0.0 and a + b + g * _delta(rb, nbar) >= 1.0:
        raise DccError("a + b + delta g must be < 1")
    if nu is not None and nu <= 2.0:
        raise DccError("nu must exceed 2")
    rng = np.random.default_rng(seed)
    total = n_obs + burn
    w = rng.standard_normal((total, n))
    mix = np.sqrt((nu - 2.0) / rng.chisquare(nu, size=total)) if nu is not None else np.ones(total)
    q = rb.copy() if q0 is None else np.asarray(q0, dtype=float).copy()
    h_prev = om / (1.0 - al - be) if h0 is None else np.asarray(h0, dtype=float).copy()
    r_prev2 = h_prev.copy()
    corr = np.empty((total, n, n))
    sigma2 = np.empty((total, n))
    zs = np.empty((total, n))
    ret = np.empty((total, n))
    intercept = (1.0 - a - b) * rb - g * nbar
    for t in range(total):
        r_t = _normalise(q[None, :, :].copy())[0]
        chol = np.linalg.cholesky(r_t)
        z = (chol @ w[t]) * (mix[t] if nu is not None else 1.0)
        h = om + al * r_prev2 + be * h_prev
        ret[t] = np.sqrt(h) * z
        corr[t], sigma2[t], zs[t] = r_t, h, z
        n_t = np.minimum(z, 0.0)
        q = intercept + a * np.outer(z, z) + b * q + g * np.outer(n_t, n_t)
        h_prev, r_prev2 = h, ret[t] ** 2
    index = pd.bdate_range(start, periods=n_obs, name="date")
    cols = [f"S{i}" for i in range(n)]
    return DccSimulation(
        returns=pd.DataFrame(ret[burn:], index=index, columns=cols),
        corr=corr[burn:],
        sigma2=sigma2[burn:],
        z=zs[burn:],
    )
