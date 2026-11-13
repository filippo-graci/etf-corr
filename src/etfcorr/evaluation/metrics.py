"""Step 5.6: the loss functions fixed for Phase 5 (``config/experiment.yaml`` ``forecasting``).

Every loss returns **one value per date**, so that a mean, a difference between two models and
a block-bootstrap interval are all computed from the same per-date series. Phase 6 adds the
formal tests (Diebold-Mariano with the HLN correction, Clark-West, the Model Confidence Set) in
``evaluation/dm_mcs.py``; nothing here is a significance test.

* ``mse_fisher_z`` (T1): :func:`squared_error` of the Fisher-z average correlation.
* ``frobenius`` (T1_matrix): :func:`pair_rmse`, the root mean squared error over pairs in
  correlation units (the Frobenius norm of the error matrix up to a constant factor).
* ``qlike`` (T2): :func:`qlike`, ``v / f - log(v / f) - 1`` for a realised variance ``v`` and a
  forecast ``f``, zero when ``f = v``. It is robust to noise in the volatility proxy (Patton
  (2011), *Volatility Forecast Comparison Using Imperfect Volatility Proxies*, Journal of
  Econometrics 160(1), 246-256; CHECKED 2026-10-11). The models forecast the *log* variance, and
  ``exp`` of a log forecast is a median-type forecast, biased low for the mean; the same
  transformation is applied to every model.
* ``rank_ic`` (T3): :func:`rank_ic`, the cross-sectional Spearman correlation per date. A
  constant forecast has an undefined correlation and is given 0 (no skill). Higher is better,
  unlike the three losses above.

:func:`r2_oos` is the out-of-sample R squared against a benchmark forecast (Campbell and
Thompson (2008); see ``models.baselines``). :func:`mean_ci` is a
stationary-bootstrap percentile interval for the mean of a per-date series (Politis and Romano
(1994), CHECKED in step 3.2), re-using the resampler of ``horizon_analysis``.
:func:`project_to_correlation` returns a positive semidefinite correlation matrix by eigenvalue
clipping and rescaling, and says whether a projection was needed.

Limits: with daily origins and an ``h``-day target the per-date losses are autocorrelated over
at least ``h - 1`` lags. The default mean block length, ``2 h`` (floored at ``n^(1/3)``), is a
rule of thumb and not an optimal choice; with about 500 validation dates and ``h = 21`` it
leaves about a dozen blocks, so intervals are wide and themselves imprecise. The percentile
interval is not bias-corrected. Clipping is not the nearest correlation matrix in the Frobenius
norm (Higham's algorithm is); it is enough to report how often a forecast was not PSD.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from etfcorr.correlation.horizon_analysis import (
    default_mean_block_length,
    stationary_bootstrap_indices,
)
from etfcorr.features import UNIT, FloatArray

PSD_TOL = 1e-10


class MetricError(ValueError):
    """Invalid input to a loss function."""


class MeanCI(NamedTuple):
    mean: float
    lower: float
    upper: float
    n: int


def _aligned(y: pd.Series, f: pd.Series) -> tuple[pd.Series, pd.Series]:
    if not y.index.equals(f.index):
        raise MetricError("target and forecast must be on the same index")
    keep = y.notna() & f.notna()
    return y[keep].astype(float), f[keep].astype(float)


def squared_error(y: pd.Series, f: pd.Series) -> pd.Series:
    """``(y - f)^2`` on the rows where both are known."""
    a, b = _aligned(y, f)
    return ((a - b) ** 2).rename("squared_error")


def absolute_error(y: pd.Series, f: pd.Series) -> pd.Series:
    """``|y - f|`` on the rows where both are known."""
    a, b = _aligned(y, f)
    return (a - b).abs().rename("absolute_error")


def qlike(variance: pd.Series, forecast: pd.Series) -> pd.Series:
    """Patton's QLIKE ``v / f - log(v / f) - 1`` per row; both inputs must be positive."""
    v, f = _aligned(variance, forecast)
    if (v <= 0).any() or (f <= 0).any():
        raise MetricError("QLIKE needs positive variances and positive forecasts")
    ratio = v / f
    return (ratio - np.log(ratio) - 1.0).rename("qlike")


def by_date(loss: pd.Series) -> pd.Series:
    """Mean over units of a per-(date, unit) loss; a per-date series is returned unchanged."""
    if isinstance(loss.index, pd.MultiIndex):
        return loss.groupby(level=0).mean()
    return loss


def pair_rmse(true: FloatArray, forecast: FloatArray) -> FloatArray:
    """Per-date root mean squared error over the strictly-upper-triangular pairs."""
    if true.shape != forecast.shape or true.ndim != 3:
        raise MetricError(f"expected two (T, N, N) stacks, got {true.shape} and {forecast.shape}")
    i, j = np.triu_indices(true.shape[1], k=1)
    return np.asarray(np.sqrt(((true[:, i, j] - forecast[:, i, j]) ** 2).mean(axis=1)))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def rank_ic(y: pd.Series, f: pd.Series) -> pd.Series:
    """Cross-sectional Spearman correlation of forecast and outcome per date (0 if constant)."""
    a, b = _aligned(y, f)
    if UNIT not in a.index.names:
        raise MetricError("rank_ic needs a (date, unit) index")
    frame = pd.DataFrame({"y": a, "f": b})
    values = {
        date: _spearman(g["y"].to_numpy(), g["f"].to_numpy()) for date, g in frame.groupby(level=0)
    }
    return pd.Series(values, name="rank_ic", dtype=float)


def hit_rate(y: pd.Series, f: pd.Series) -> pd.Series:
    """Per-date share of units whose forecast has the sign of the outcome (zeros dropped)."""
    a, b = _aligned(y, f)
    keep = (a != 0) & (b != 0)
    hits = (np.sign(a[keep]) == np.sign(b[keep])).astype(float)
    return by_date(hits).rename("hit_rate")


def ic_ir(ic: pd.Series) -> float:
    """Information ratio of a per-date rank IC series: mean divided by standard deviation.

    Not annualised and not adjusted for the overlap of the targets: with an ``h``-day target the
    daily ICs are autocorrelated, so this is a descriptive ratio, not a t-statistic.
    """
    x = ic.dropna().to_numpy(dtype=float)
    if len(x) < 2 or float(x.std(ddof=1)) == 0.0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1))


def r2_oos(loss_model: pd.Series, loss_benchmark: pd.Series) -> float:
    """``1 - sum(model losses) / sum(benchmark losses)`` on the dates both series have."""
    both = pd.concat([loss_model, loss_benchmark], axis=1, join="inner").dropna()
    denominator = float(both.iloc[:, 1].sum())
    if len(both) == 0 or denominator <= 0:
        return float("nan")
    return 1.0 - float(both.iloc[:, 0].sum()) / denominator


def mean_ci(
    values: pd.Series,
    *,
    mean_block_length: float | None = None,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> MeanCI:
    """Mean of a per-date series with a stationary-bootstrap percentile interval."""
    x = values.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 2:
        return MeanCI(float(x.mean()) if n else float("nan"), float("nan"), float("nan"), n)
    block = default_mean_block_length(n) if mean_block_length is None else mean_block_length
    rng = np.random.default_rng(seed)
    means = np.array(
        [x[stationary_bootstrap_indices(n, block, rng)].mean() for _ in range(n_bootstrap)]
    )
    lower, upper = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return MeanCI(float(x.mean()), float(lower), float(upper), n)


def block_length_for(horizon: int, n_obs: int) -> float:
    """Rule-of-thumb mean block length for losses of an ``h``-day overlapping target: ``2 h``."""
    return float(max(2 * horizon, default_mean_block_length(max(n_obs, 2))))


def project_to_correlation(corr: FloatArray) -> tuple[FloatArray, bool]:
    """A PSD unit-diagonal matrix from a symmetric one, and whether it had to be changed.

    Eigenvalues below zero are clipped to zero and the result is rescaled to a unit diagonal.
    """
    c = np.asarray(corr, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not np.isfinite(c).all():
        raise MetricError("expected one finite square matrix")
    values, vectors = np.linalg.eigh((c + c.T) / 2.0)
    if values.min() >= -PSD_TOL:
        return c, False
    fixed = (vectors * np.clip(values, 0.0, None)) @ vectors.T
    d = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(d, d)
    np.fill_diagonal(fixed, 1.0)
    return fixed, True
