"""Step 4.5b: diagnostics for the lead-lag scan of step 4.5 (exploratory, descriptive).

Question served: the real-data scan of step 4.5 showed only 10 of 3000 raw cross-autocorrelation
cells below p = 0.05 (a null gives 5% in expectation) and 46 of 1500 for the asymmetry statistic.
Two explanations were left untested in ``docs/checks/4.5.md``:

* **H_dep (dependence).** The raw cells are almost one statistic: ``Cov(r_j,t, r_i,t-k) = b_i b_j
  gamma_f(k)`` plus idiosyncratic terms, so when the common factor's own autocorrelation is near
  zero the whole family is small together and the *share* of cells below 0.05 is far from binomial
  (its null distribution is wide, with mean 5%). A count of 10 can then be ordinary.
  **[Inference]**
* **H_tail (outlier days).** A handful of extreme days dominate the standardised products and
  inflate the bootstrap standard error, so the normal-reference p-values are conservative.
  **[Inference]**

The functions here measure what separates the two, on the discovery window only:

1. :func:`factor_autocorrelation`: the lagged autocorrelations of the equal-weight panel average
   (the common factor proxy) with a stationary-bootstrap z, which decides whether the raw family
   *should* be small;
2. :func:`top_day_share`: the share of each cell's ``sum z_t^2`` carried by its ``k`` largest days
   (concentration of the product mass), compared with a row-shuffled copy of the same returns and
   with i.i.d. Gaussian data of the same size;
3. :func:`shuffle_null`: the scan repeated on row-shuffled copies (serial dependence destroyed,
   marginals and the contemporaneous cross-section kept), giving the null *distribution* of the
   count of cells with ``p < alpha`` and of the median bootstrap SE, and :func:`empirical_position`
   to place the real count in it;
4. :func:`winsorise_standardised`: the standardised returns clipped at ``+-k`` sample standard
   deviations, a declared sensitivity variant (not a replacement of the step 4.5 specification, and
   not a removal of any period).

Reading rule, fixed before any real-data run (see ``scripts/diagnose_4_5b.py``): the real count is
**consistent with H_dep** if it lies inside the central 90% of the shuffled-null counts; it points
to **H_tail** only if it lies below the 5th percentile *and* the median SE of the real window
exceeds that of the shuffled copies by more than 20% *and* winsorising moves the count towards the
null median. Otherwise (below the 5th percentile without those conditions) the anomaly is reported
as **unexplained**, and a count above the 95th percentile as above the null range. Neither outcome
changes the step 4.5 specification or any hypothesis; a different primary statistic would be a new
exploratory variant.

No literature is cited for the diagnostics (they are descriptive); the stationary bootstrap and the
scan are those of ``lead_lag.py`` and ``horizon_analysis.py`` (citations and check status there).
No third-party code is adapted. Every function is a whole-window statistic of the frame it is
given: the caller restricts the frame to the discovery window with
:func:`etfcorr.correlation.lead_lag.slice_window`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from etfcorr.correlation import lead_lag as ll
from etfcorr.correlation.horizon_analysis import default_mean_block_length

__all__ = [
    "ShuffleNull",
    "classify_conservatism",
    "empirical_position",
    "excess_kurtosis",
    "factor_autocorrelation",
    "shuffle_null",
    "shuffle_rows",
    "top_day_share",
    "top_days",
    "winsorise_standardised",
]


def excess_kurtosis(returns: pd.DataFrame) -> pd.Series:
    """Excess kurtosis per column (0 for a Gaussian), population moments, NaN rejected."""
    x = _complete(returns)
    return pd.Series(stats.kurtosis(x, axis=0, fisher=True), index=returns.columns, name="kurtosis")


def top_days(returns: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """The ``n`` dates with the largest mean squared standardised return across assets.

    Columns: ``date``, ``mean_sq`` (mean over assets of ``u^2``) and ``ew_return`` (the equal-weight
    average standardised return that day, to show the sign).
    """
    if n < 1:
        raise ll.LeadLagError(f"n must be >= 1, got {n}")
    u = ll.standardise(returns)
    score = (u**2).mean(axis=1)
    order = np.argsort(-score, kind="stable")[:n]
    return pd.DataFrame(
        {
            "date": returns.index[order],
            "mean_sq": score[order],
            "ew_return": u.mean(axis=1)[order],
        }
    )


def winsorise_standardised(returns: pd.DataFrame, k: float = 4.0) -> pd.DataFrame:
    """Columns standardised (mean 0, sd 1, ddof 1) and clipped at ``+-k``; same index and columns.

    ``lead_lag_scan`` standardises again, so the scan sees a mildly rescaled clipped series.
    """
    if k <= 0.0:
        raise ll.LeadLagError(f"k must be > 0, got {k}")
    u = ll.standardise(returns)
    return pd.DataFrame(np.clip(u, -k, k), index=returns.index, columns=returns.columns)


def shuffle_rows(returns: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Rows permuted jointly (the cross-section of each day kept), original index and columns."""
    rng = np.random.default_rng(seed)
    values = returns.to_numpy(dtype=float)[rng.permutation(len(returns))]
    return pd.DataFrame(values, index=returns.index, columns=returns.columns)


def top_day_share(z: np.ndarray, k: int = 5) -> np.ndarray:
    """Per column, the share of ``sum_t z_t^2`` carried by the ``k`` largest ``z_t^2``.

    ``z`` is a product-series matrix from :func:`etfcorr.correlation.lead_lag.lagged_products`.
    Columns with zero total are given share 0. The reference level for comparison is the same
    statistic on i.i.d. Gaussian data of the same size, computed by the caller.
    """
    if z.ndim != 2 or not 1 <= k < z.shape[0]:
        raise ll.LeadLagError("z must be 2-D with 1 <= k < number of rows")
    sq = z**2
    total = sq.sum(axis=0)
    top = np.partition(sq, sq.shape[0] - k, axis=0)[-k:].sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(np.where(total > 0.0, top / total, 0.0), dtype=float)


def factor_autocorrelation(
    returns: pd.DataFrame,
    lags: tuple[int, ...] = ll.DEFAULT_LAGS,
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Autocorrelation of the equal-weight average of the standardised returns, by lag.

    Columns: ``lag``, ``autocorr`` (mean of ``f_t f_{t-k}`` for the standardised average ``f``),
    ``se`` and ``z`` from the stationary bootstrap with the data-driven block length (the same
    machinery as the scan), and ``p_value`` (two-sided normal). A proxy for the common factor whose
    lagged autocovariance is what the raw cells measure; it is not tested as a hypothesis.
    """
    ks = ll._check_lags(lags)
    u = ll.standardise(returns)
    f = u.mean(axis=1)
    f = (f - f.mean()) / f.std(ddof=1)
    kmax = max(ks)
    cols = np.stack([f[kmax - k : len(f) - k] * f[kmax:] for k in ks], axis=1)
    block = default_mean_block_length(cols.shape[0])
    means = ll.bootstrap_means(cols, block, n_bootstrap, seed)
    est = cols.mean(axis=0)
    se = means.std(axis=0, ddof=1)
    z = est / se
    return pd.DataFrame(
        {
            "lag": list(ks),
            "autocorr": est,
            "se": se,
            "z": z,
            "p_value": 2.0 * stats.norm.sf(np.abs(z)),
        }
    )


@dataclass(frozen=True)
class ShuffleNull:
    """Counts of cells with ``p < alpha`` and median bootstrap SEs over row-shuffled copies."""

    counts: np.ndarray
    median_se: np.ndarray
    n_tests: int
    alpha: float

    def quantiles(self, qs: tuple[float, ...] = (0.05, 0.5, 0.95)) -> dict[float, float]:
        return {q: float(np.quantile(self.counts, q)) for q in qs}


def shuffle_null(
    returns: pd.DataFrame,
    *,
    statistic: ll.Statistic = "asymmetry",
    transform: ll.Transform = "none",
    lags: tuple[int, ...] = ll.DEFAULT_LAGS,
    n_shuffles: int = 100,
    n_bootstrap: int = 500,
    alpha: float = ll.DEFAULT_ALPHA,
    seed: int = 0,
) -> ShuffleNull:
    """Repeat the scan on ``n_shuffles`` row-shuffled copies and record the null count and SE.

    Shuffle ``s`` uses seed ``seed + s`` for the permutation and ``seed`` for the bootstrap, so the
    result is deterministic.
    """
    if n_shuffles < 2:
        raise ll.LeadLagError(f"n_shuffles must be >= 2, got {n_shuffles}")
    counts = np.empty(n_shuffles)
    med = np.empty(n_shuffles)
    n_tests = 0
    for s in range(n_shuffles):
        scan = ll.lead_lag_scan(
            shuffle_rows(returns, seed + s),
            lags=lags,
            statistic=statistic,
            transform=transform,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            seed=seed,
        )
        counts[s] = float((scan.table["p_value"] < alpha).sum())
        med[s] = float(np.median(scan.table["se"].to_numpy(dtype=float)))
        n_tests = scan.n_tests
    return ShuffleNull(counts=counts, median_se=med, n_tests=n_tests, alpha=alpha)


def empirical_position(null_counts: np.ndarray, observed: float) -> tuple[float, float]:
    """``(P(null <= observed), P(null >= observed))`` with the ``+1`` correction in both.

    ``(1 + #{null <= obs}) / (1 + n)`` and ``(1 + #{null >= obs}) / (1 + n)``; both are at least
    ``1 / (n + 1)`` and they sum to more than 1 when ``observed`` equals a null value.
    """
    n = len(null_counts)
    if n < 1:
        raise ll.LeadLagError("null_counts is empty")
    le = (1 + int(np.sum(null_counts <= observed))) / (1 + n)
    ge = (1 + int(np.sum(null_counts >= observed))) / (1 + n)
    return float(le), float(ge)


def classify_conservatism(
    real_count: float,
    null_counts: np.ndarray,
    *,
    se_ratio: float,
    winsorised_count: float,
) -> str:
    """Apply the reading rule of the module docstring to one statistic.

    ``se_ratio`` is the real window's median bootstrap SE divided by the median over the shuffled
    copies' medians; ``winsorised_count`` is the real count after :func:`winsorise_standardised`.
    Returns a one-line verdict starting with ``H_dep``, ``above null``, ``H_tail`` or
    ``unexplained``.
    """
    if len(null_counts) < 2:
        raise ll.LeadLagError("need at least 2 null counts")
    lo, med, hi = (float(np.quantile(null_counts, q)) for q in (0.05, 0.5, 0.95))
    if real_count > hi:
        return f"above null range [{lo:.0f}, {hi:.0f}]: anti-conservative or a real signal"
    if real_count >= lo:
        return f"H_dep: inside the central 90% of the null [{lo:.0f}, {hi:.0f}]"
    moves_to_null = abs(winsorised_count - med) < abs(real_count - med)
    if se_ratio > 1.2 and moves_to_null:
        return (
            f"H_tail supported: below the null 5th percentile {lo:.0f}, SE ratio {se_ratio:.2f} "
            f"> 1.2 and winsorising moves the count to {winsorised_count:.0f} "
            f"(null median {med:.0f})"
        )
    return (
        f"unexplained: below the null 5th percentile {lo:.0f}, but SE ratio {se_ratio:.2f} and "
        f"winsorised count {winsorised_count:.0f} do not meet the H_tail criteria"
    )


def _complete(returns: pd.DataFrame) -> np.ndarray:
    x = returns.to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ll.LeadLagError("returns contain NaN or infinite values")
    return x
