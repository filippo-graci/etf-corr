"""Step 8.1: covariance estimators into minimum-variance and HRP portfolios (H5).

**Question served.** H5: do shrinkage, RMT-cleaned and DCC-forecast covariance estimates give
minimum-variance portfolios with lower out-of-sample realised volatility than the raw sample
covariance? The design and the verdict rule are in ``config/experiment.yaml`` (``economics``),
declared before any portfolio was computed.

**Walk-forward.** On the first session of each month in the evaluated window, a covariance is
estimated from the trailing window of returns dated **strictly before** that session (daily,
weekly, monthly or quarterly returns, see ``estimation``), a long-only fully invested portfolio
is formed and held to the next rebalance while its weights drift with prices. The daily
portfolio returns are simple returns; realised volatility is their standard deviation times
``sqrt(252)``. Nothing dated at or after a rebalance enters its weights
(``tests/unit/test_portfolio.py``).

**Estimators.** ``sample``; ``ledoit_wolf`` (scikit-learn, shrinkage to a scaled identity);
``ledoit_wolf_corr`` (the same on standardised returns, with the sample volatilities put back);
``rmt`` (sample volatilities with the sample correlation denoised at the Marchenko-Pastur edge,
``correlation.rmt.denoise_corr``); ``ewma`` (RiskMetrics recursion over the window); ``dcc`` (the
one-step DCC correlation forecast made the day before, with EWMA variances; daily only).

**Portfolios.** Minimum variance by SLSQP (``scipy``) and hierarchical risk parity (López de
Prado 2016: single-linkage ordering, recursive bisection with inverse-variance weights). Both
are written here because skfolio does not import in this environment (``cvxpy`` missing): a
deviation from the plan. The tests check the optimiser against the closed form where the
long-only constraint does not bind, and HRP against hand-computed cases.

Assumptions and limits: long-only, no leverage, no weight cap; the DCC cell mixes a DCC
correlation with EWMA variances because ``rolling_dcc`` does not export GARCH variance
forecasts, so it is not a full DCC covariance; monthly rebalancing only; costs are a flat
number of basis points on traded notional, with no spread, impact, tax or borrow; close-to-close
vendor prices (asynchronous closes, premium/discount to NAV and retroactive adjustments are not
modelled); the Sharpe ratio uses a zero risk-free rate and is reported only because the plan
asks for it. **Realised volatility is the result; nothing here is a claim of tradeable alpha.**

Reference (CHECKED on Crossref 2026-11-02): López de Prado (2016), *Building
Diversified Portfolios that Outperform Out of Sample*, Journal of Portfolio Management 42(4), 59-69.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import purgedcv
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

from etfcorr.config import CovEstimator, DateWindow, Economics
from etfcorr.correlation.horizon_analysis import stationary_bootstrap_indices
from etfcorr.correlation.rmt import denoise_corr
from etfcorr.features import FloatArray

TRADING_DAYS = 252
PORTFOLIOS: tuple[str, ...] = ("minvar", "hrp")
# Exploratory (docs/FOLLOW_UPS.md item 3): accepted by ``weights``, not part of ``run_all``.
EXPLORATORY_PORTFOLIOS: tuple[str, ...] = ("minvar_ls",)


class PortfolioError(ValueError):
    """Invalid portfolio request."""


# --------------------------------------------------------------------------------------
# Covariances and weights
# --------------------------------------------------------------------------------------


def ewma_covariance(x: FloatArray, lam: float) -> FloatArray:
    """RiskMetrics covariance of the rows of ``x`` (zero mean, ``S_0 = x_0 x_0'``)."""
    s = np.outer(x[0], x[0])
    for row in x[1:]:
        s = lam * s + (1.0 - lam) * np.outer(row, row)
    return np.asarray(s)


def nonlinear_shrinkage(x: FloatArray) -> FloatArray:
    """Analytical nonlinear shrinkage of the sample covariance of the rows of ``x``.

    Ledoit and Wolf (2020), *Analytical nonlinear shrinkage of large-dimensional covariance
    matrices*, Annals of Statistics 48(5) (checked against the authors' copy, 2026-11-01:
    equations 4.8-4.9 and the summary in Section 4.7; ``n`` is the sample size minus one for
    demeaned data, Remark 2.1): the sample eigenvectors are kept and each sample eigenvalue is
    replaced using a kernel estimate (Epanechnikov, bandwidth ``lambda_j * n^(-1/3)``) of the
    spectral density and of its Hilbert transform. Needs more rows than columns.
    """
    n, p = x.shape[0] - 1, x.shape[1]  # one degree of freedom goes to the mean
    if p >= n:
        raise PortfolioError("nonlinear shrinkage is implemented for fewer columns than rows")
    lam, u = np.linalg.eigh(np.cov(x, rowvar=False))
    lam = np.clip(lam, 1e-12 * lam[-1], None)
    c = p / n
    h = n ** (-1.0 / 3.0) * lam[None, :]  # bandwidth of each kernel, proportional to its centre
    z = (lam[:, None] - lam[None, :]) / h
    density = (0.75 / np.sqrt(5.0)) * (np.clip(1.0 - z**2 / 5.0, 0.0, None) / h).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.log(np.abs((np.sqrt(5.0) - z) / (np.sqrt(5.0) + z)))
    hilbert = -0.3 / np.pi * z + (0.75 / np.sqrt(5.0) / np.pi) * (1.0 - z**2 / 5.0) * log_term
    edge = np.isclose(np.abs(z), np.sqrt(5.0))
    hilbert[edge] = -0.3 / np.pi * z[edge]
    hilbert = (hilbert / h).mean(axis=1)
    shrunk = lam / ((np.pi * c * lam * density) ** 2 + (1.0 - c - np.pi * c * lam * hilbert) ** 2)
    return np.asarray((u * shrunk) @ u.T)


def covariance(
    window: pd.DataFrame,
    estimator: CovEstimator,
    *,
    lam: float = 0.94,
    dcc_corr: FloatArray | None = None,
    dcc_var: FloatArray | None = None,
) -> FloatArray:
    """Covariance matrix of the columns of ``window`` (complete rows) by one estimator."""
    x = window.to_numpy(dtype=float)
    if x.ndim != 2 or x.shape[0] < 3 or not np.isfinite(x).all():
        raise PortfolioError("the estimation window needs at least 3 complete rows")
    if estimator == "sample":
        return np.asarray(np.cov(x, rowvar=False))
    if estimator == "ledoit_wolf":
        return np.asarray(LedoitWolf().fit(x).covariance_)
    if estimator == "ledoit_wolf_corr":
        sd = x.std(axis=0, ddof=1)
        shrunk = LedoitWolf().fit((x - x.mean(axis=0)) / sd).covariance_
        d = np.sqrt(np.diag(shrunk))
        return np.asarray(shrunk / np.outer(d, d) * np.outer(sd, sd))
    if estimator == "rmt":
        sd = x.std(axis=0, ddof=1)
        corr = pd.DataFrame(
            np.corrcoef(x, rowvar=False), index=window.columns, columns=window.columns
        )
        clean = denoise_corr(corr, x.shape[0]).to_numpy(dtype=float)
        return np.asarray(clean * np.outer(sd, sd))
    if estimator == "ewma":
        return ewma_covariance(x, lam)
    if estimator == "dcc":
        if dcc_corr is None or not np.isfinite(dcc_corr).all():
            raise PortfolioError("the dcc estimator needs a finite correlation forecast")
        sd = np.sqrt(np.diag(ewma_covariance(x, lam)))
        return np.asarray(dcc_corr * np.outer(sd, sd))
    if estimator == "nonlinear":
        return nonlinear_shrinkage(x)
    if estimator == "nonlinear_corr":  # as ledoit_wolf_corr: clean the correlation, keep the vols
        sd = x.std(axis=0, ddof=1)
        shrunk = nonlinear_shrinkage((x - x.mean(axis=0)) / sd)
        d = np.sqrt(np.diag(shrunk))
        return np.asarray(shrunk / np.outer(d, d) * np.outer(sd, sd))
    if estimator == "dcc_garch":  # the full DCC covariance: its correlation and GARCH variances
        if dcc_corr is None or dcc_var is None:
            raise PortfolioError("the dcc_garch estimator needs dcc_corr and dcc_var")
        if not (np.isfinite(dcc_corr).all() and np.isfinite(dcc_var).all() and (dcc_var > 0).all()):
            raise PortfolioError("the dcc_garch estimator needs finite, positive forecasts")
        sd = np.sqrt(dcc_var)
        return np.asarray(dcc_corr * np.outer(sd, sd))
    raise PortfolioError(f"unknown estimator {estimator!r}")


def min_variance_weights(cov: FloatArray) -> FloatArray:
    """Long-only, fully invested minimum-variance weights (SLSQP from equal weights)."""
    n = cov.shape[0]
    scaled = cov / np.trace(cov) * n  # weights do not depend on the scale; SLSQP's tolerance does
    done = minimize(
        lambda w: float(w @ scaled @ w),
        np.full(n, 1.0 / n),
        jac=lambda w: 2.0 * scaled @ w,
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not done.success:
        raise PortfolioError(f"minimum-variance optimisation failed: {done.message}")
    w = np.clip(done.x, 0.0, None)
    return np.asarray(w / w.sum())


def min_variance_ls_weights(cov: FloatArray) -> FloatArray:
    """Fully invested minimum variance with no sign constraint: ``S^-1 1 / (1' S^-1 1)``."""
    raw = np.linalg.solve(cov, np.ones(cov.shape[0]))
    return np.asarray(raw / raw.sum())


def hrp_weights(cov: FloatArray, *, optimal_ordering: bool = False) -> FloatArray:
    """Hierarchical risk parity weights (single linkage on the correlation distance).

    The leaves are in dendrogram order, as in López de Prado (2016). ``optimal_ordering`` reorders
    them to minimise the distance between neighbours, which is what skfolio does; the two give
    different weights whenever a cluster is not split at the same place (checked 2026-10-12: with
    it the weights equal skfolio's, without it they differ by up to 0.08 on a 6-asset panel).
    """
    n = cov.shape[0]
    sd = np.sqrt(np.diag(cov))
    corr = np.clip(cov / np.outer(sd, sd), -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    tree = linkage(condensed, method="single", optimal_ordering=optimal_ordering)
    order = leaves_list(tree).tolist()

    def cluster_variance(items: list[int]) -> float:
        sub = cov[np.ix_(items, items)]
        ivp = 1.0 / np.diag(sub)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)

    w = np.ones(n)
    clusters = [order]
    while clusters:
        clusters = [c[k:m] for c in clusters for k, m in ((0, len(c) // 2), (len(c) // 2, len(c)))]
        clusters = [c for c in clusters if c]
        for left, right in zip(clusters[0::2], clusters[1::2], strict=False):
            v_left, v_right = cluster_variance(left), cluster_variance(right)
            alpha = 1.0 - v_left / (v_left + v_right)
            w[left] *= alpha
            w[right] *= 1.0 - alpha
        clusters = [c for c in clusters if len(c) > 1]
    return np.asarray(w / w.sum())


def weights(cov: FloatArray, portfolio: str) -> FloatArray:
    if portfolio == "minvar":
        return min_variance_weights(cov)
    if portfolio == "hrp":
        return hrp_weights(cov)
    if portfolio == "minvar_ls":
        return min_variance_ls_weights(cov)
    raise PortfolioError(
        f"unknown portfolio {portfolio!r}; expected one of {PORTFOLIOS + EXPLORATORY_PORTFOLIOS}"
    )


# --------------------------------------------------------------------------------------
# Walk-forward
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioRun:
    """Daily simple returns of one portfolio and the turnover at each rebalance."""

    returns: pd.Series
    turnover: pd.Series
    weights: pd.DataFrame  # target weights at each rebalance


def rebalance_dates(index: pd.DatetimeIndex, window: DateWindow) -> pd.DatetimeIndex:
    """The first session of each calendar month inside ``window``."""
    inside = index[(index >= pd.Timestamp(window.start)) & (index <= pd.Timestamp(window.end))]
    if len(inside) == 0:
        raise PortfolioError(f"no session inside {window.start}..{window.end}")
    first = pd.Series(inside, index=inside).groupby(inside.to_period("M")).first()
    return pd.DatetimeIndex(first.to_numpy())


def hold(daily: pd.DataFrame, targets: pd.DataFrame, end: pd.Timestamp) -> PortfolioRun:
    """Hold the ``targets`` (rows: rebalance dates) with drifting weights until ``end``.

    ``daily`` holds log returns. The weights set on a rebalance date earn that date's return.
    Turnover is the sum of absolute weight changes against the drifted portfolio (1 at the first
    rebalance: the portfolio is bought from cash).
    """
    held = daily.loc[targets.index[0] : end]
    simple = pd.DataFrame(np.expm1(held.to_numpy(dtype=float)), held.index, held.columns)
    out = np.empty(len(simple))
    turnover: dict[pd.Timestamp, float] = {}
    w = np.zeros(daily.shape[1])
    for k, (date, row) in enumerate(zip(simple.index, simple.to_numpy(), strict=True)):
        if date in targets.index:
            target = targets.loc[date].to_numpy(dtype=float)
            turnover[date] = float(np.abs(target - w).sum())
            w = target
        out[k] = float(w @ row)
        w = w * (1.0 + row) / (1.0 + out[k])
    return PortfolioRun(
        pd.Series(out, index=simple.index, name="return"),
        pd.Series(turnover, name="turnover"),
        targets,
    )


def run_cell(
    daily: pd.DataFrame,
    source: pd.DataFrame,
    n_rows: int,
    estimator: CovEstimator,
    portfolio: str,
    window: DateWindow,
    *,
    lam: float = 0.94,
    dcc_corr: FloatArray | None = None,
    dcc_var: FloatArray | None = None,
) -> PortfolioRun:
    """One (frequency, estimator, portfolio) cell over the evaluated ``window``.

    ``daily`` are the daily log returns that are traded; ``source`` are the returns the
    covariance is estimated from (``daily`` itself, or the weekly, monthly or quarterly panel),
    of which the last ``n_rows`` rows dated strictly before each rebalance are used.
    ``dcc_corr[t]`` (and ``dcc_var[t]``, for ``dcc_garch``) is the forecast for the day after
    row ``t`` of ``daily``.
    """
    index = pd.DatetimeIndex(daily.index)
    targets = {}
    for date in rebalance_dates(index, window):
        history = source.loc[source.index < date].iloc[-n_rows:]
        if len(history) < n_rows:
            raise PortfolioError(f"fewer than {n_rows} rows before {date.date()}")
        forecast = variances = None
        if estimator in ("dcc", "dcc_garch"):
            if dcc_corr is None:
                raise PortfolioError("the dcc estimator needs dcc_corr")
            forecast = dcc_corr[int(index.searchsorted(date)) - 1]
            if dcc_var is not None:
                variances = dcc_var[int(index.searchsorted(date)) - 1]
        cov = covariance(history, estimator, lam=lam, dcc_corr=forecast, dcc_var=variances)
        targets[date] = weights(cov, portfolio)
    frame = pd.DataFrame(targets, index=daily.columns).T
    return hold(daily, frame, pd.Timestamp(window.end))


def cell_name(portfolio: str, freq: str, estimator: str) -> str:
    return f"{portfolio}|{freq}|{estimator}"


def run_all(
    panels: Mapping[str, pd.DataFrame],
    design: Economics,
    window: DateWindow,
    *,
    lam: float = 0.94,
    dcc_corr: FloatArray | None = None,
    skip_short: bool = False,
) -> dict[str, PortfolioRun]:
    """Every cell of the design plus the equal-weight portfolio (``equal|-|-``).

    ``panels`` maps a frequency (``D``, ``W``, ...) to its core return panel; ``D`` is traded.
    The ``dcc`` estimator exists at the daily frequency only. ``skip_short`` drops a frequency
    whose estimation window is not yet full at the first rebalance instead of raising (the
    quarterly window of 32 rows does not exist before 2022, so the validation rehearsal cannot
    have it; the final run must).
    """
    daily = panels["D"]
    runs: dict[str, PortfolioRun] = {}
    start = rebalance_dates(pd.DatetimeIndex(daily.index), window)[0]
    for freq, n_rows in design.estimation.items():
        if skip_short and (panels[freq].index < start).sum() < n_rows:
            continue
        for estimator in design.estimators:
            if estimator == "dcc" and freq != "D":
                continue
            for portfolio in PORTFOLIOS:
                runs[cell_name(portfolio, freq, estimator)] = run_cell(
                    daily,
                    panels[freq],
                    n_rows,
                    estimator,
                    portfolio,
                    window,
                    lam=lam,
                    dcc_corr=dcc_corr,
                )
    dates = rebalance_dates(pd.DatetimeIndex(daily.index), window)
    equal = pd.DataFrame(1.0 / daily.shape[1], index=dates, columns=daily.columns)
    runs[cell_name("equal", "-", "-")] = hold(daily, equal, pd.Timestamp(window.end))
    return runs


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------


def _bootstrap_vols(returns: pd.DataFrame, n_bootstrap: int, block: float, seed: int) -> FloatArray:
    """Annualised volatility of every column in each stationary-bootstrap resample of the days."""
    x = returns.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    out = np.empty((n_bootstrap, x.shape[1]))
    for b in range(n_bootstrap):
        out[b] = x[stationary_bootstrap_indices(len(x), block, rng)].std(axis=0, ddof=1)
    return np.asarray(out * np.sqrt(TRADING_DAYS))


def h5_cells(design: Economics, columns: Sequence[str]) -> list[tuple[str, str]]:
    """``(cell, its sample-covariance cell)`` pairs that enter the H5 statistic."""
    pairs = []
    for freq in design.estimation:
        for estimator in design.h5_estimators:
            cell = cell_name("minvar", freq, estimator)
            if cell in columns:
                pairs.append((cell, cell_name("minvar", freq, "sample")))
    return pairs


def volatility_table(
    runs: Mapping[str, PortfolioRun],
    design: Economics,
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Realised volatility of every cell against its sample-covariance cell, and the H5 reading.

    All resampling is joint over days (the same resampled days for every cell), with a mean
    block of ``design.block_days``. ``diff_vs_sample`` is the cell's annualised volatility minus
    that of the sample covariance at the same frequency and portfolio; negative is better.
    """
    returns = pd.DataFrame({name: run.returns for name, run in runs.items()}).dropna()
    vols = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    boot = pd.DataFrame(
        _bootstrap_vols(returns, n_bootstrap, design.block_days, seed), columns=returns.columns
    )
    rows = []
    for name in returns.columns:
        portfolio, freq, estimator = name.split("|")
        base = cell_name(portfolio, freq, "sample")
        row: dict[str, object] = {
            "cell": name,
            "portfolio": portfolio,
            "frequency": freq,
            "estimator": estimator,
            "n_days": len(returns),
            "ann_vol": float(vols[name]),
            "vol_ci_lower": float(boot[name].quantile(alpha / 2)),
            "vol_ci_upper": float(boot[name].quantile(1 - alpha / 2)),
        }
        if base in returns.columns and base != name:
            diff = boot[name] - boot[base]
            row.update(
                diff_vs_sample=float(vols[name] - vols[base]),
                diff_ci_lower=float(diff.quantile(alpha / 2)),
                diff_ci_upper=float(diff.quantile(1 - alpha / 2)),
            )
        rows.append(row)
    table = pd.DataFrame(rows)

    def statistic(pairs: Sequence[tuple[str, str]]) -> tuple[float, pd.Series]:
        cells, bases = [c for c, _ in pairs], [b for _, b in pairs]
        point = float((vols[cells].to_numpy() - vols[bases].to_numpy()).mean())
        draws = pd.Series((boot[cells].to_numpy() - boot[bases].to_numpy()).mean(axis=1))
        return point, draws

    pairs = h5_cells(design, list(returns.columns))
    if not pairs:
        raise PortfolioError("no cell of the H5 statistic was run")
    point, draws = statistic(pairs)
    p_value = float((1 + (draws >= 0).sum()) / (n_bootstrap + 1))
    verdict: dict[str, object] = {
        "statistic": point,
        "ci_lower": float(draws.quantile(alpha / 2)),
        "ci_upper": float(draws.quantile(1 - alpha / 2)),
        "p_one_sided": p_value,
        "alpha": alpha,
        "n_cells": len(pairs),
        "cells": [c for c, _ in pairs],
        "n_days": len(returns),
        "h5_supported": bool(point < 0 and p_value < alpha),
    }
    low = [p for p in pairs if p[0].split("|")[1] in ("M", "Q")]
    high = [p for p in pairs if p[0].split("|")[1] == "D"]
    variant = design.h5_variant
    if variant is not None:  # the declared candidate against the estimator it would replace
        swap = [
            (cell_name("minvar", f, variant.candidate), cell_name("minvar", f, variant.replaces))
            for f in design.estimation
            if cell_name("minvar", f, variant.candidate) in returns.columns
        ]
        if swap:
            v_point, v_draws = statistic(swap)
            lower, upper = v_draws.quantile(alpha / 2), v_draws.quantile(1 - alpha / 2)
            verdict["variant"] = {
                "candidate": variant.candidate,
                "replaces": variant.replaces,
                "difference": v_point,
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "rule_met": bool(v_point < 0 and upper < 0),
            }
    if low and high:  # "especially at monthly and quarterly horizons": descriptive
        (p_low, d_low), (p_high, d_high) = statistic(low), statistic(high)
        gap = d_low - d_high
        verdict["monthly_quarterly_minus_daily"] = {
            "monthly_quarterly": p_low,
            "daily": p_high,
            "difference": p_low - p_high,
            "ci_lower": float(gap.quantile(alpha / 2)),
            "ci_upper": float(gap.quantile(1 - alpha / 2)),
        }
    return table, verdict


def backtest_table(
    runs: Mapping[str, PortfolioRun], costs_bps: Sequence[float], *, pbo_splits: int = 8
) -> tuple[pd.DataFrame, float]:
    """Return, turnover and Sharpe of every cell at each cost level, and the PBO across cells.

    ``sharpe`` is annualised with a zero risk-free rate; ``sharpe_se`` is Lo's (2002) i.i.d.
    standard error; ``deflated_sharpe`` (``purgedcv``) is the probability that the true Sharpe is
    above what the best of as many unrelated trials as there are cells would show by luck. The
    PBO (``purgedcv``, CSCV) is the share of splits in which the cell with the best in-sample
    Sharpe is below the median out of sample. Costs are charged on the rebalance day.
    """
    returns = pd.DataFrame({name: run.returns for name, run in runs.items()}).dropna()
    daily_sharpe = returns.mean() / returns.std(ddof=1)
    var_sharpe = float(daily_sharpe.var(ddof=1))
    years = len(returns) / TRADING_DAYS
    rows = []
    for name, run in runs.items():
        for bps in costs_bps:
            cost = (run.turnover * bps / 1e4).reindex(returns.index).fillna(0.0)
            net = returns[name] - cost
            sr = float(net.mean() / net.std(ddof=1))
            rows.append(
                {
                    "cell": name,
                    "cost_bps": bps,
                    "ann_return": float(net.mean() * TRADING_DAYS),
                    "ann_vol": float(net.std(ddof=1) * np.sqrt(TRADING_DAYS)),
                    "sharpe": sr * np.sqrt(TRADING_DAYS),
                    "sharpe_se": float(
                        np.sqrt((1 + 0.5 * sr**2) / len(net)) * np.sqrt(TRADING_DAYS)
                    ),
                    "deflated_sharpe": float(
                        purgedcv.deflated_sharpe_ratio(net.to_numpy(), len(runs), var_sharpe)
                    ),
                    "turnover_per_year": float(run.turnover.sum() / years),
                }
            )
    pbo = purgedcv.probability_of_backtest_overfitting(returns.to_numpy().T, n_splits=pbo_splits)
    return pd.DataFrame(rows), float(pbo.pbo)
