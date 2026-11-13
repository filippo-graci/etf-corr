"""Step 6.2: forecast-comparison tests (Diebold-Mariano with HLN, Clark-West, Model Confidence Set).

All three work on the per-date losses or forecasts that ``evaluation.ladder`` produces.

* :func:`diebold_mariano`: equal predictive accuracy of two forecasts (Diebold and Mariano 1995).
  With ``d_t = L_t(a) - L_t(b)`` and ``n`` dates, the statistic is

  .. math:: DM = \\bar d / \\sqrt{\\hat\\omega / n}, \\qquad
            \\hat\\omega = \\hat\\gamma_0 + 2 \\sum_{k=1}^{h-1} \\hat\\gamma_k

  (rectangular kernel with ``h - 1`` autocovariances, because the errors of an optimal ``h``-step
  forecast are at most MA(``h - 1``)), multiplied by the small-sample factor of Harvey, Leybourne
  and Newbold (1997), ``sqrt((n + 1 - 2h + h(h - 1)/n) / n)``, and referred to a Student ``t``
  with ``n - 1`` degrees of freedom. For ``h = 1`` this is exactly the paired t-test (a unit
  test checks it against ``scipy.stats.ttest_1samp``).
* :func:`clark_west`: for **nested** models (Clark and West 2007). Under the null that the small
  model is true the larger model's squared error is inflated by the noise of its extra
  parameters; the adjusted difference ``(y - f_s)^2 - [(y - f_l)^2 - (f_s - f_l)^2]`` removes
  that term. One-sided, standard normal critical values.
* :func:`model_confidence_set`: the set of models that contains the best one with a given
  confidence (Hansen, Lunde and Nason 2011), through ``arch.bootstrap.MCS``, not re-implemented.

Assumptions and limits, to state next to any p-value:

* **Overlapping targets.** With daily origins and ``h = 21`` the loss differential is strongly
  autocorrelated and ``n`` dates are about ``n / h`` independent observations. The rectangular
  estimate of the long-run variance is noisy with 20 lags and can be **negative**; then the
  Bartlett (Newey-West) kernel with the same number of lags is used instead and the result says
  so (``kernel``). A calibration on simulated MA(``h - 1``) differentials
  (``scripts/dm_calibration_6_2.py``) measures the actual size at the project's ``n`` and ``h``;
  read p-values against that, not against the nominal level.
* DM compares *forecasts*, not models: with expanding-window re-estimation and nested models its
  limit is not standard (hence Clark-West). Penalised and tree models are not nested in the
  simple baselines in the regression sense, so DM is used for them.
* Clark-West's normal approximation is the authors' recommendation for rolling estimation; for
  an expanding window it is an approximation.
* The MCS depends on the block length (``2 h`` here, the rule of ``metrics.block_length_for``)
  and has low power in short samples: a large set means "cannot tell them apart", not "all are
  equally good".
* None of these is corrected for the number of comparisons made.

Why an own DM function when ``statsmodels.tsa.stattools.diebold_mariano_test`` exists: that one
takes the observations and two forecasts and a per-observation loss, whereas the losses here
are per-date aggregates over assets or pairs (QLIKE averaged over 25 assets, a cross-sectional
rank IC), and it offers only the Newey-West variance. A unit test checks that this module's
Bartlett path reproduces the statsmodels statistic and p-value exactly.

References (existence, authors and venue checked by search on 2026-10-11): Diebold and Mariano
(1995), *Comparing Predictive Accuracy*, JBES 13(3), 253-263; Harvey, Leybourne and Newbold
(1997), *Testing the equality of prediction mean squared errors*, International Journal of
Forecasting 13(2), 281-291; Clark and West (2007), *Approximately normal tests for equal
predictive accuracy in nested models*, Journal of Econometrics 138(1), 291-311; Hansen, Lunde
and Nason (2011), *The Model Confidence Set*, Econometrica 79(2), 453-497.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from arch.bootstrap import MCS
from scipy import stats

Alternative = Literal["two-sided", "less", "greater"]
Kernel = Literal["rectangular", "bartlett"]


class ComparisonError(ValueError):
    """Invalid input to a forecast-comparison test."""


@dataclass(frozen=True)
class DMResult:
    """Diebold-Mariano test with the HLN correction. ``mean_diff < 0`` means ``a`` is better."""

    statistic: float
    p_value: float
    mean_diff: float
    n: int
    horizon: int
    long_run_variance: float
    kernel: Kernel
    alternative: Alternative


@dataclass(frozen=True)
class CWResult:
    """Clark-West test. A positive ``statistic`` favours the larger model."""

    statistic: float
    p_value: float
    mean_adjusted_diff: float
    n: int
    horizon: int
    kernel: Kernel


def autocovariances(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocovariances ``gamma_0 .. gamma_max_lag`` (divisor ``n``, mean removed)."""
    n = len(x)
    centred = x - x.mean()
    lags = min(max_lag, n - 1)
    return np.array([float(centred[k:] @ centred[: n - k]) / n for k in range(lags + 1)])


def long_run_variance(
    x: np.ndarray, lags: int, kernel: Kernel = "rectangular"
) -> tuple[float, Kernel]:
    """Long-run variance of ``x`` with ``lags`` autocovariances, and the kernel actually used.

    A non-positive rectangular estimate falls back to the Bartlett kernel, which is positive
    by construction.
    """
    if lags < 0:
        raise ComparisonError(f"lags must be >= 0, got {lags}")
    gamma = autocovariances(np.asarray(x, dtype=float), lags)
    k = np.arange(1, len(gamma))
    if kernel == "rectangular":
        value = float(gamma[0] + 2.0 * gamma[1:].sum())
        if value > 0:
            return value, "rectangular"
    elif kernel != "bartlett":
        raise ComparisonError(f"unknown kernel {kernel!r}")
    weights = 1.0 - k / (len(gamma))
    return float(gamma[0] + 2.0 * (weights * gamma[1:]).sum()), "bartlett"


def _common(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    both = pd.concat([a, b], axis=1, join="inner").dropna()
    return both.iloc[:, 0].to_numpy(dtype=float), both.iloc[:, 1].to_numpy(dtype=float)


def _p_value(statistic: float, alternative: Alternative, df: int | None) -> float:
    dist = stats.norm if df is None else stats.t(df)
    if alternative == "less":
        return float(dist.cdf(statistic))
    if alternative == "greater":
        return float(dist.sf(statistic))
    if alternative == "two-sided":
        return float(2.0 * dist.sf(abs(statistic)))
    raise ComparisonError(f"unknown alternative {alternative!r}")


def diebold_mariano(
    loss_a: pd.Series,
    loss_b: pd.Series,
    horizon: int,
    *,
    alternative: Alternative = "two-sided",
    kernel: Kernel = "rectangular",
) -> DMResult:
    """DM test of equal expected loss on the common dates, HLN-corrected.

    ``alternative="less"`` tests whether ``a`` has the lower expected loss. For a score where
    higher is better (rank IC), pass the negated series.
    """
    if horizon < 1:
        raise ComparisonError(f"horizon must be >= 1, got {horizon}")
    a, b = _common(loss_a, loss_b)
    n = len(a)
    if n <= horizon or n < 3:
        raise ComparisonError(f"{n} common dates are too few for horizon {horizon}")
    d = a - b
    lrv, used = long_run_variance(d, horizon - 1, kernel)
    if lrv <= 0:  # identical losses
        return DMResult(
            float("nan"), float("nan"), float(d.mean()), n, horizon, lrv, used, alternative
        )
    hln = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic = float(d.mean() / np.sqrt(lrv / n) * hln)
    return DMResult(
        statistic,
        _p_value(statistic, alternative, n - 1),
        float(d.mean()),
        n,
        horizon,
        lrv,
        used,
        alternative,
    )


def clark_west(
    y: pd.Series, forecast_small: pd.Series, forecast_large: pd.Series, horizon: int
) -> CWResult:
    """Clark-West test that the larger of two nested models forecasts better (one-sided)."""
    if horizon < 1:
        raise ComparisonError(f"horizon must be >= 1, got {horizon}")
    frame = pd.concat([y, forecast_small, forecast_large], axis=1, join="inner").dropna()
    n = len(frame)
    if n <= horizon or n < 3:
        raise ComparisonError(f"{n} common dates are too few for horizon {horizon}")
    target, small, large = (frame.iloc[:, k].to_numpy(dtype=float) for k in range(3))
    adjusted = (target - small) ** 2 - ((target - large) ** 2 - (small - large) ** 2)
    lrv, used = long_run_variance(adjusted, horizon - 1)
    if lrv <= 0:
        return CWResult(float("nan"), float("nan"), float(adjusted.mean()), n, horizon, used)
    statistic = float(adjusted.mean() / np.sqrt(lrv / n))
    return CWResult(
        statistic, _p_value(statistic, "greater", None), float(adjusted.mean()), n, horizon, used
    )


def model_confidence_set(
    losses: pd.DataFrame,
    *,
    size: float = 0.10,
    block_size: int,
    reps: int = 2000,
    seed: int = 0,
    higher_is_better: bool = False,
) -> pd.DataFrame:
    """MCS p-value of each model and whether it is in the set at level ``size``.

    ``losses`` has one column per model and one row per date (rows with a missing value are
    dropped). Sorted by p-value, best last; ``included`` is ``p_value > size``.
    """
    frame = losses.dropna()
    if frame.shape[1] < 2 or len(frame) <= block_size:
        raise ComparisonError("need at least two models and more dates than the block size")
    values = -frame if higher_is_better else frame
    mcs = MCS(
        values, size=size, reps=reps, block_size=block_size, bootstrap="stationary", seed=seed
    )
    mcs.compute()
    out = pd.DataFrame({"p_value": mcs.pvalues["Pvalue"].astype(float)})
    out["included"] = out["p_value"] > size
    out["mean_loss"] = frame.mean().reindex(out.index)
    out.index.name = "model"
    return out
