"""Step 8.2: hedge ratios and hedge effectiveness by regime (H6).

**Question served.** H6: is hedge effectiveness (variance reduction) lower in the worst-decile
stress periods than in calm periods? The design and the verdict rule are in
``config/experiment.yaml`` (``economics``), declared before any hedge was computed.

**Hedge.** Each core ETF ``i`` is hedged with one instrument ``m`` (SPY): the hedged return on
day ``t`` is ``r_i,t - beta_i,t-1 r_m,t``, where ``beta_i,t-1`` is estimated with data dated up
to ``t - 1``. Three single-instrument ratios: ``rolling`` (OLS beta over the trailing window,
the primary one), ``ewma`` (RiskMetrics second moments) and ``static`` (one beta from the data
before the evaluated window). ``lasso`` is a sparse multi-instrument hedge: a cross-validated
lasso of the ETF on several liquid instruments, refitted on the first session of each month
from the trailing window dated before it.

**Effectiveness** in a set of days ``G``: ``HE_i(G) = 1 - Var(hedged_i | G) / Var(r_i | G)``,
variances about the mean inside ``G`` (Ederington 1979). **Regimes** are labels of the
evaluated window itself, by the VIX close of the same day: stress above its ``stress_quantile``,
calm at or below its ``calm_quantile``. They are ex-post descriptions, not signals: nobody
could have traded on them, and they are not used to set any hedge ratio.

**H6 statistic**: the mean over the ETFs of ``HE(stress) - HE(calm)`` for the rolling hedge,
with a stationary bootstrap over days (returns and labels resampled together).

Assumptions and limits: for a single instrument and a constant in-regime beta, ``HE`` is the
squared in-regime correlation minus a penalty for using the wrong beta, so it mixes two things:
how correlated the ETF is with the hedge in that regime, and how stale the beta is. Correlations
with the market rise in high-volatility periods (H2), which pushes ``HE`` **up** in stress, the
opposite of what H6 expects; a lower stress ``HE`` would have to come from beta instability.
About a tenth of the window is stress, in a few clusters, so the stress estimate rests on few
independent episodes. Close-to-close prices, no costs, no margin, no shorting constraints.

Reference (CHECKED on Crossref 2026-11-02): Ederington (1979), *The Hedging Performance
of the New Futures Markets*, Journal of Finance 34(1), 157-170.
"""

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit

from etfcorr.config import DateWindow
from etfcorr.correlation.horizon_analysis import stationary_bootstrap_indices
from etfcorr.correlation.regime_correlation import forbes_rigobon_adjust
from etfcorr.evaluation.portfolio import rebalance_dates
from etfcorr.features import BoolArray, FloatArray

PRIMARY = "rolling"


class HedgingError(ValueError):
    """Invalid hedging request."""


def rolling_beta(assets: pd.DataFrame, hedge: pd.Series, window: int) -> pd.DataFrame:
    """OLS beta of each column on ``hedge`` over the trailing ``window`` rows ending at ``t``."""
    cov = assets.rolling(window, min_periods=window).cov(hedge)
    return cov.div(hedge.rolling(window, min_periods=window).var(), axis=0)


def ewma_beta(assets: pd.DataFrame, hedge: pd.Series, lam: float) -> pd.DataFrame:
    """RiskMetrics beta: EWMA of ``r_i r_m`` over EWMA of ``r_m^2`` (zero means), up to ``t``."""
    cross = assets.mul(hedge, axis=0).ewm(alpha=1.0 - lam, adjust=False).mean()
    return cross.div((hedge**2).ewm(alpha=1.0 - lam, adjust=False).mean(), axis=0)


def static_beta(assets: pd.DataFrame, hedge: pd.Series, before: pd.Timestamp) -> pd.DataFrame:
    """One OLS beta per column from the rows dated strictly before ``before``, on every row."""
    past = assets.index < before
    if past.sum() < 30:
        raise HedgingError("fewer than 30 rows before the evaluated window for the static beta")
    beta = assets.loc[past].apply(lambda c: c.cov(hedge.loc[past])) / hedge.loc[past].var()
    return pd.DataFrame(np.tile(beta.to_numpy(), (len(assets), 1)), assets.index, assets.columns)


def hedged_returns(assets: pd.DataFrame, hedge: pd.Series, beta: pd.DataFrame) -> pd.DataFrame:
    """``r_i,t - beta_i,t-1 r_m,t``: the ratio known the day before is the one applied."""
    return assets - beta.shift(1).mul(hedge, axis=0)


def lasso_hedged_returns(
    assets: pd.DataFrame,
    instruments: pd.DataFrame,
    window_days: int,
    window: DateWindow,
    *,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sparse multi-instrument hedge, refitted monthly: ``(hedged returns, share selected)``.

    On the first session of each month in ``window`` a ``LassoCV`` (three time-ordered folds, no
    intercept penalty) of each ETF on ``instruments`` is fitted on the ``window_days`` rows dated
    strictly before it, and its coefficients are held to the next refit. The second frame gives,
    per refit date and instrument, the share of ETFs whose hedge uses that instrument.
    """
    index = pd.DatetimeIndex(assets.index)
    dates = rebalance_dates(index, window)
    hedged = pd.DataFrame(np.nan, index=index, columns=assets.columns)
    used = {}
    for k, date in enumerate(dates):
        past = index < date
        x = instruments.loc[past].iloc[-window_days:].to_numpy(dtype=float)
        if len(x) < window_days:
            raise HedgingError(f"fewer than {window_days} rows before {date.date()}")
        stop = (
            dates[k + 1] if k + 1 < len(dates) else pd.Timestamp(window.end) + pd.Timedelta(days=1)
        )
        held = (index >= date) & (index < stop)
        coefs = []
        for name in assets.columns:
            y = assets.loc[past, name].iloc[-window_days:].to_numpy(dtype=float)
            model = LassoCV(cv=TimeSeriesSplit(3), random_state=seed, max_iter=5000).fit(x, y)
            coefs.append(model.coef_)
            hedged.loc[held, name] = (
                assets.loc[held, name] - instruments.loc[held].to_numpy() @ model.coef_
            )
        used[date] = (np.abs(np.array(coefs)) > 1e-12).mean(axis=0)
    return hedged, pd.DataFrame(used, index=instruments.columns).T


def regime_masks(
    vix: pd.Series, stress_quantile: float, calm_quantile: float
) -> tuple[BoolArray, BoolArray]:
    """``(stress, calm)`` over the rows of ``vix`` (already cut to the evaluated window)."""
    level = vix.to_numpy(dtype=float)
    if not np.isfinite(level).all() or len(level) < 20:
        raise HedgingError("need at least 20 finite VIX levels")
    stress = level > np.quantile(level, stress_quantile)
    calm = level <= np.quantile(level, calm_quantile)
    return np.asarray(stress), np.asarray(calm)


def effectiveness(unhedged: FloatArray, hedged: FloatArray, mask: BoolArray) -> FloatArray:
    """``1 - Var(hedged) / Var(unhedged)`` per column over the rows selected by ``mask``."""
    if mask.sum() < 2:
        return np.full(unhedged.shape[1], np.nan)
    return np.asarray(1.0 - hedged[mask].var(axis=0, ddof=1) / unhedged[mask].var(axis=0, ddof=1))


def h6_table(
    unhedged: pd.DataFrame,
    hedged: Mapping[str, pd.DataFrame],
    stress: BoolArray,
    calm: BoolArray,
    *,
    block_days: int = 21,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Hedge effectiveness by regime for every method: ``(summary, per-asset, H6 reading)``.

    All frames share the index of ``unhedged`` (the evaluated window, complete rows). The same
    resampled days are used for every method. ``diff`` is the mean over ETFs of
    ``HE(stress) - HE(calm)``; its one-sided p-value is the add-one share of bootstrap draws at
    or above zero. The verdict is read from the ``rolling`` method.
    """
    if PRIMARY not in hedged:
        raise HedgingError(f"the primary method {PRIMARY!r} was not run")
    raw = unhedged.to_numpy(dtype=float)
    n = len(raw)
    everything = np.ones(n, dtype=bool)
    arrays = {name: frame.to_numpy(dtype=float) for name, frame in hedged.items()}
    for name, values in arrays.items():
        if values.shape != raw.shape or not np.isfinite(values).all():
            raise HedgingError(f"{name}: hedged returns must be complete and aligned")

    def diff(values: FloatArray, base: FloatArray, s: BoolArray, c: BoolArray) -> float:
        return float(np.nanmean(effectiveness(base, values, s) - effectiveness(base, values, c)))

    rng = np.random.default_rng(seed)
    draws = {name: np.empty(n_bootstrap) for name in arrays}
    for b in range(n_bootstrap):
        idx = stationary_bootstrap_indices(n, block_days, rng)
        for name, values in arrays.items():
            draws[name][b] = diff(values[idx], raw[idx], stress[idx], calm[idx])

    rows: list[dict[str, float | str]] = []
    assets: dict[str, FloatArray] = {}
    for name, values in arrays.items():
        he_stress, he_calm = effectiveness(raw, values, stress), effectiveness(raw, values, calm)
        d = draws[name][np.isfinite(draws[name])]
        rows.append(
            {
                "method": name,
                "n_days": n,
                "n_stress": int(stress.sum()),
                "n_calm": int(calm.sum()),
                "he_all": float(effectiveness(raw, values, everything).mean()),
                "he_stress": float(he_stress.mean()),
                "he_calm": float(he_calm.mean()),
                "diff": float((he_stress - he_calm).mean()),
                "diff_ci_lower": float(np.quantile(d, alpha / 2)),
                "diff_ci_upper": float(np.quantile(d, 1 - alpha / 2)),
                "p_one_sided": float((1 + (d >= 0).sum()) / (len(d) + 1)),
                "share_assets_lower_in_stress": float((he_stress < he_calm).mean()),
            }
        )
        assets[f"{name}_stress"], assets[f"{name}_calm"] = he_stress, he_calm
    summary = pd.DataFrame(rows)
    primary = {k: float(v) for k, v in rows[list(arrays).index(PRIMARY)].items() if k != "method"}
    verdict: dict[str, object] = {
        "method": PRIMARY,
        "statistic": primary["diff"],
        "ci_lower": primary["diff_ci_lower"],
        "ci_upper": primary["diff_ci_upper"],
        "p_one_sided": primary["p_one_sided"],
        "alpha": alpha,
        "he_stress": primary["he_stress"],
        "he_calm": primary["he_calm"],
        "n_stress": int(stress.sum()),
        "n_calm": int(calm.sum()),
        "h6_supported": bool(primary["diff"] < 0 and primary["p_one_sided"] < alpha),
    }
    return summary, pd.DataFrame(assets, index=unhedged.columns), verdict


# --------------------------------------------------------------------------------------
# Exploratory follow-up E1 (docs/FOLLOW_UPS.md item 1): not part of the H6 verdict
# --------------------------------------------------------------------------------------

TRADING_DAYS = 252


def regime_risk(
    unhedged: pd.DataFrame,
    hedged: pd.DataFrame,
    hedge: pd.Series,
    stress: BoolArray,
    calm: BoolArray,
    *,
    tail: float = 0.05,
) -> pd.DataFrame:
    """Per ETF and regime: what the hedge removes, corrected for volatility, and what is left.

    Columns, each with a ``_stress`` and a ``_calm`` version: ``he`` (effectiveness), ``r2`` (the
    squared in-regime correlation with the hedge), ``vol_unhedged`` and ``vol_hedged``
    (annualised), ``var_`` and ``es_`` of both (the ``tail`` worst daily loss and the mean loss
    beyond it, as positive numbers). ``r2_adj_stress`` is the stress correlation after the
    Forbes-Rigobon adjustment to the calm variance of the hedge, squared: the share of variance a
    perfectly sized hedge would remove if the hedge were as volatile as in calm days. All frames
    are aligned on the same complete rows.
    """
    raw, hed, m = unhedged.to_numpy(float), hedged.to_numpy(float), hedge.to_numpy(float)
    out: dict[str, FloatArray] = {}
    rho = {}
    for name, mask in (("stress", stress), ("calm", calm)):
        if mask.sum() < 3:
            raise HedgingError(f"fewer than 3 {name} days")
        x, y = raw[mask], hed[mask]
        centred = x - x.mean(axis=0)
        mc = m[mask] - m[mask].mean()
        rho[name] = (centred * mc[:, None]).sum(axis=0) / np.sqrt(
            (centred**2).sum(axis=0) * (mc**2).sum()
        )
        out[f"he_{name}"] = effectiveness(raw, hed, mask)
        out[f"r2_{name}"] = rho[name] ** 2
        for label, values in (("unhedged", x), ("hedged", y)):
            cut = np.quantile(values, tail, axis=0)
            out[f"vol_{label}_{name}"] = values.std(axis=0, ddof=1) * np.sqrt(TRADING_DAYS)
            out[f"var_{label}_{name}"] = -cut
            out[f"es_{label}_{name}"] = -np.array(
                [values[values[:, j] <= cut[j], j].mean() for j in range(values.shape[1])]
            )
    delta = m[stress].var(ddof=1) / m[calm].var(ddof=1) - 1.0
    out["r2_adj_stress"] = forbes_rigobon_adjust(rho["stress"], delta) ** 2
    return pd.DataFrame(out, index=unhedged.columns)


def regime_risk_table(
    unhedged: pd.DataFrame,
    hedged: pd.DataFrame,
    hedge: pd.Series,
    stress: BoolArray,
    calm: BoolArray,
    *,
    block_days: int = 21,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Means over ETFs of :func:`regime_risk` and of four stress-minus-calm gaps, with intervals.

    The gaps: ``he_diff`` (the H6 statistic for these labels), ``r2_diff``, ``r2_adj_diff`` (the
    volatility-adjusted gap) and ``vol_hedged_diff`` (leftover volatility). Days, labels and the
    hedge are resampled together (stationary bootstrap); a draw with fewer than three
    distinct days of a regime is dropped and counted in ``n_draws``.
    """

    def stats(idx: npt.NDArray[np.int64] | slice) -> pd.Series:
        s = regime_risk(
            unhedged.iloc[idx], hedged.iloc[idx], hedge.iloc[idx], stress[idx], calm[idx]
        )
        mean = s.mean()
        mean["he_diff"] = mean["he_stress"] - mean["he_calm"]
        mean["r2_diff"] = mean["r2_stress"] - mean["r2_calm"]
        mean["r2_adj_diff"] = mean["r2_adj_stress"] - mean["r2_calm"]
        mean["vol_hedged_diff"] = mean["vol_hedged_stress"] - mean["vol_hedged_calm"]
        return mean

    point = stats(slice(None))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap):
        idx = stationary_bootstrap_indices(len(unhedged), block_days, rng)
        # distinct days: one stress day drawn three times has no variance
        if min(len(np.unique(idx[stress[idx]])), len(np.unique(idx[calm[idx]]))) >= 3:
            draws.append(stats(idx))
    boot = pd.DataFrame(draws)
    return pd.DataFrame(
        {
            "value": point,
            "ci_lower": boot.quantile(alpha / 2),
            "ci_upper": boot.quantile(1 - alpha / 2),
            "n_draws": len(boot),
        }
    ).rename_axis("statistic")
