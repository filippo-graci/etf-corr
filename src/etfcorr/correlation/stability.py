"""Estimator-stability table (step 3.5): sampling spread of each correlation estimator, by horizon.

Question served: the Phase 3 milestone "estimator-stability table" (``project_plan.md`` section 10,
week 3). The plan does not define its columns; the definition below is a design decision recorded in
``docs/checks/3.5.md`` (an **[Inference]**, not a cited protocol). This module is descriptive. It is
**not** one of the pre-registered hypotheses H1-H6 and ranks nothing for forecasting or hedging:
whether a shrunk, denoised or detoned matrix is *more accurate* out of sample is H5 (Phase 8).

What "stability" means here (and what it does not)
--------------------------------------------------
For each horizon and each estimator, the complete-case return sample (T rows, N assets) is
resampled ``n_bootstrap`` times with the Politis-Romano stationary bootstrap (reusing
:func:`etfcorr.correlation.horizon_analysis.stationary_bootstrap_indices`, so rows keep their local
serial dependence), the estimator is re-run on each replicate, and the table reports:

* the point estimate's own diagnostics: condition number (``inf`` when the matrix is numerically
  singular: always for ``detoned``, whose rank is at most N - ``n_market``, and for a sample matrix
  with T <= N), smallest eigenvalue, PSD flag, the average off-diagonal correlation (the project's
  coupling indicator);
* the bootstrap distribution of the average off-diagonal correlation: mean, bias against the point
  estimate, standard deviation and a percentile interval;
* ``rms_pair_sd``: the root-mean-square, over the N(N-1)/2 pairs, of each pair's bootstrap
  standard deviation. It is the spread *around the bootstrap mean* of each entry, so it measures
  variance only, not bias, and does not depend on where the estimator is centred;
* ``rel_spread``: ``rms_pair_sd`` divided by the root-mean-square of the point-estimate's
  off-diagonal entries (spread relative to the size of the structure being estimated);
* ``eigvec_overlap``: mean squared cosine between the leading eigenvector of a replicate matrix and
  that of the point-estimate matrix (1 = the dominant mode never moves);
* ``cond_boot_median`` and ``psd_share_boot``: the median condition number and the share of
  replicate matrices that are PSD (the latter is an empirical check on Kendall, which is PSD on
  complete rows by the argument in ``estimators.py``).

Common random numbers: within a horizon, every estimator is evaluated on the **same** set of
bootstrap row-index arrays, so differences between estimators are not confounded by resampling
noise. The index sets depend only on ``seed``, the horizon and T (not on the estimator list), so a
table computed for one estimator equals the corresponding rows of a table computed for all of them.

Estimators: ``pearson``, ``spearman``, ``kendall``, ``ledoit_wolf``, ``oas``, ``ewma`` (all through
:func:`etfcorr.correlation.estimators.estimate`), plus two post-processed Pearson matrices:
``denoised`` (:func:`etfcorr.correlation.rmt.denoise_corr`, Marchenko-Pastur edge with
``n_obs = T``) and ``detoned`` (:func:`etfcorr.correlation.rmt.detone_corr`, ``n_market``
eigenmodes removed).

Assumptions and known failure modes
-----------------------------------
* **Lower variance is not higher accuracy.** Shrinkage and denoising buy variance reduction with
  bias toward the identity (or toward a flat noise floor); the table reports the bias in the
  average correlation (``boot_bias``) only for that one summary, not for the full matrix. A
  stable-but-wrong estimator looks good here. Do not use the table alone to pick an estimator.
* **Detoned values are relative.** The average off-diagonal of a detoned matrix is not on the same
  scale as the others (cross-sector entries are negative by construction), so compare its
  ``avg_corr`` columns with nothing; ``rel_spread`` and ``eigvec_overlap`` are the comparable
  columns, and ``eigvec_overlap`` there refers to the leading *residual* mode, not the market mode.
* **EWMA is order-dependent.** The point estimate weights the actual last rows most; a resampled
  row order does not reproduce that. The spread therefore mixes sampling error with time-variation
  of the correlation, and ``boot_bias`` and ``eigvec_overlap`` for EWMA carry that mixture.
* **Quarterly results are exploratory.** With T close to N (about 36 quarterly rows for N = 25
  after the 2022 cut) the sample matrix is ill-conditioned and the stationary bootstrap draws only
  a handful of distinct blocks; ``cond`` and ``psd_share_boot`` show it. The default block length
  ``round(T ** (1/3))`` is an **[Inference]** (Politis-White 2004 is not implemented).
* The bootstrap is i.i.d.-block resampling of rows: it does not model volatility regimes, so the
  spread is the spread under a stationary world. Fat tails are kept (rows are resampled, not
  simulated).
* No replicate is dropped: an estimator error on a replicate propagates (dropping unstable
  replicates would bias the spread toward the stable ones), as in ``horizon_analysis``.
* The window is whole-sample descriptive. ``end`` exists so the caller can keep the sealed test
  window unread (the script defaults it to the validation end, 2022-12-31): a table used to choose
  an estimator that is later evaluated on 2023 onward must not have read 2023 onward.

Citations (status)
------------------
* Politis, D. N., Romano, J. P. (1994). The stationary bootstrap. JASA 89(428), 1303-1313, DOI
  10.1080/01621459.1994.10476870. CHECKED (step 3.2, ``REFERENCES_SEED.md``).
* Ledoit, O., Wolf, M. (2004), Honey, I shrunk the sample covariance matrix, J. Portfolio
  Management 30(4): CHECKED (``REFERENCES_SEED.md``). Chen et al. (2010) OAS: CHECKED on Crossref
  2026-11-02.
* Laloux et al. (1999), Plerou et al. (2002), Bun-Bouchaud-Potters (2017): CHECKED (step 3.1).
* The "estimator-stability" columns (RMS pairwise bootstrap SD, relative spread, eigenvector
  overlap) are this project's own descriptive choices, not attributed to a paper.

Third-party code: none adapted; nothing new for ``THIRD_PARTY.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, get_args

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from etfcorr.config import Horizon
from etfcorr.correlation.estimators import (
    DEFAULT_EWMA_LAMBDA,
    eigen_diagnostics,
    estimate,
)
from etfcorr.correlation.horizon_analysis import (
    default_mean_block_length,
    leading_eigenvector_overlap,
    stationary_bootstrap_indices,
)
from etfcorr.correlation.rmt import denoise_corr, detone_corr

__all__ = [
    "ESTIMATORS",
    "TABLE_COLUMNS",
    "EstimatorName",
    "StabilityError",
    "bootstrap_index_sets",
    "estimator_matrix",
    "estimator_stability_table",
    "restrict_window",
    "stability_summary_lines",
]

EstimatorName = Literal[
    "pearson", "spearman", "kendall", "ledoit_wolf", "oas", "ewma", "denoised", "detoned"
]
ESTIMATORS: tuple[EstimatorName, ...] = get_args(EstimatorName)

# Position of each horizon in the seed sequence: fixed, so a horizon's replicates do not depend on
# which other horizons are requested.
_HORIZON_ORDER: tuple[str, ...] = ("D", "W", "M", "Q")

MIN_BOOTSTRAP = 20
MIN_OBS = 8

TABLE_COLUMNS: tuple[str, ...] = (
    "horizon",
    "estimator",
    "n_obs",
    "n_assets",
    "q",
    "cond",
    "min_eigenvalue",
    "is_psd",
    "avg_corr",
    "boot_mean",
    "boot_bias",
    "boot_sd",
    "ci_lower",
    "ci_upper",
    "rms_pair_sd",
    "rel_spread",
    "eigvec_overlap",
    "cond_boot_median",
    "psd_share_boot",
    "n_bootstrap",
    "mean_block_length",
    "window_start",
    "window_end",
)


class StabilityError(ValueError):
    """Invalid input to, or an invalid configuration of, an estimator-stability call."""


# --------------------------------------------------------------------------- window and matrices


def restrict_window(
    returns: pd.DataFrame, start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    """Rows dated in the closed interval ``[start, end]`` (either bound optional).

    The input is not modified. The index must be a strictly increasing ``DatetimeIndex``. This is
    the one place a caller keeps later data (for example the sealed test window) out of a table.
    """
    if not isinstance(returns, pd.DataFrame):
        raise StabilityError(f"returns must be a pandas DataFrame, got {type(returns)}")
    index = returns.index
    if not isinstance(index, pd.DatetimeIndex):
        raise StabilityError("returns must have a DatetimeIndex")
    if not (index.is_monotonic_increasing and index.is_unique):
        raise StabilityError("returns index must be strictly increasing")
    mask = np.ones(len(index), dtype=bool)
    if start is not None:
        mask &= np.asarray(index >= pd.Timestamp(start))
    if end is not None:
        mask &= np.asarray(index <= pd.Timestamp(end))
    out = returns.loc[mask]
    if out.empty:
        raise StabilityError(f"no rows in the window [{start}, {end}]")
    return out.copy()


def estimator_matrix(
    sample: pd.DataFrame,
    name: EstimatorName,
    *,
    ewma_lambda: float = DEFAULT_EWMA_LAMBDA,
    n_market: int = 1,
) -> pd.DataFrame:
    """One labelled correlation matrix from ``sample`` (complete-case rows) by estimator name.

    ``denoised`` and ``detoned`` start from the Pearson matrix of the same ``sample``;
    ``denoised`` uses the sample's own row count as ``n_obs`` for the Marchenko-Pastur edge.
    """
    if name == "denoised":
        pearson = estimate(sample, "pearson")
        return denoise_corr(pearson, n_obs=len(sample))
    if name == "detoned":
        return detone_corr(estimate(sample, "pearson"), n_market=n_market)
    if name == "ewma":
        return estimate(sample, "ewma", lam=ewma_lambda)
    if name in ("pearson", "spearman", "kendall", "ledoit_wolf", "oas"):
        return estimate(sample, name)
    raise StabilityError(f"unknown estimator {name!r}; expected one of {ESTIMATORS}")


def bootstrap_index_sets(
    n_obs: int, n_bootstrap: int, mean_block_length: float, seed: int, horizon: str
) -> NDArray[np.int64]:
    """``(n_bootstrap, n_obs)`` stationary-bootstrap row indices for one horizon.

    Seeded from ``(seed, horizon)`` only, so the replicates of a horizon do not depend on the other
    horizons or on the estimator list.
    """
    if horizon not in _HORIZON_ORDER:
        raise StabilityError(f"unknown horizon {horizon!r}; expected one of {_HORIZON_ORDER}")
    if n_bootstrap < 1:
        raise StabilityError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), _HORIZON_ORDER.index(horizon)]))
    sets = np.empty((n_bootstrap, n_obs), dtype=np.int64)
    for b in range(n_bootstrap):
        sets[b] = stationary_bootstrap_indices(n_obs, mean_block_length, rng)
    return sets


# --------------------------------------------------------------------------- one table cell

RANK_TOL = 1e-10


def _condition(min_eigenvalue: float, max_eigenvalue: float, svd_condition: float) -> float:
    """Condition number, or ``inf`` when the matrix is numerically singular.

    A detoned matrix has rank at most ``N - n_market`` by construction, and a sample matrix with
    ``T <= N`` is singular too; the SVD condition number of such a matrix is a floating-point
    artefact of order ``1 / eps`` that differs between otherwise identical runs, so it is replaced
    by ``inf`` when the smallest eigenvalue is below ``RANK_TOL`` of the largest.
    """
    if max_eigenvalue <= 0.0 or min_eigenvalue <= RANK_TOL * max_eigenvalue:
        return float("inf")
    return svd_condition


def _cell(
    values: NDArray[np.float64],
    columns: pd.Index,
    name: EstimatorName,
    index_sets: NDArray[np.int64],
    *,
    ewma_lambda: float,
    n_market: int,
    alpha: float,
) -> dict[str, float | bool]:
    n_obs, n_assets = values.shape
    point = estimator_matrix(
        pd.DataFrame(values, columns=columns), name, ewma_lambda=ewma_lambda, n_market=n_market
    )
    iu = np.triu_indices(n_assets, k=1)
    point_pairs = point.to_numpy(dtype=float)[iu]
    diag = eigen_diagnostics(point)
    point_cond = _condition(diag.min_eigenvalue, diag.max_eigenvalue, diag.condition_number)

    n_boot = index_sets.shape[0]
    pairs = np.empty((n_boot, point_pairs.size), dtype=float)
    overlap = np.empty(n_boot, dtype=float)
    cond = np.empty(n_boot, dtype=float)
    psd = np.empty(n_boot, dtype=bool)
    for b in range(n_boot):
        boot = pd.DataFrame(values[index_sets[b]], columns=columns)
        matrix = estimator_matrix(boot, name, ewma_lambda=ewma_lambda, n_market=n_market)
        pairs[b] = matrix.to_numpy(dtype=float)[iu]
        overlap[b] = leading_eigenvector_overlap(point, matrix, 1)
        d = eigen_diagnostics(matrix)
        cond[b] = _condition(d.min_eigenvalue, d.max_eigenvalue, d.condition_number)
        psd[b] = d.is_psd

    avg_point = float(point_pairs.mean())
    avg_boot = pairs.mean(axis=1)
    pair_sd = pairs.std(axis=0, ddof=1)
    rms_sd = float(np.sqrt(np.mean(pair_sd**2)))
    rms_level = float(np.sqrt(np.mean(point_pairs**2)))
    return {
        "n_obs": float(n_obs),
        "n_assets": float(n_assets),
        "q": n_assets / n_obs,
        "cond": point_cond,
        "min_eigenvalue": diag.min_eigenvalue,
        "is_psd": diag.is_psd,
        "avg_corr": avg_point,
        "boot_mean": float(avg_boot.mean()),
        "boot_bias": float(avg_boot.mean()) - avg_point,
        "boot_sd": float(avg_boot.std(ddof=1)),
        "ci_lower": float(np.quantile(avg_boot, alpha / 2.0)),
        "ci_upper": float(np.quantile(avg_boot, 1.0 - alpha / 2.0)),
        "rms_pair_sd": rms_sd,
        "rel_spread": rms_sd / rms_level if rms_level > 0 else float("nan"),
        "eigvec_overlap": float(overlap.mean()),
        "cond_boot_median": float(np.median(cond)),
        "psd_share_boot": float(psd.mean()),
    }


# --------------------------------------------------------------------------- the table


def estimator_stability_table(
    returns_by_horizon: Mapping[Horizon, pd.DataFrame],
    estimators: Sequence[EstimatorName] = ESTIMATORS,
    *,
    start: str | None = None,
    end: str | None = None,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    mean_block_length: float | None = None,
    ewma_lambda: float = DEFAULT_EWMA_LAMBDA,
    n_market: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    """The estimator-stability table: one row per (horizon, estimator), columns ``TABLE_COLUMNS``.

    For each horizon the frame is cut to ``[start, end]`` (rows dated after ``end`` are never
    read), reduced to its complete-case rows (so T is identical for every estimator), and every
    estimator is run on the point sample and on ``n_bootstrap`` shared stationary-bootstrap
    replicates. ``mean_block_length`` defaults per horizon to ``round(T ** (1/3))`` (floored at 2).
    See the module docstring for the columns and for what the table cannot say.
    """
    if not returns_by_horizon:
        raise StabilityError("returns_by_horizon is empty; nothing to compute")
    if not estimators:
        raise StabilityError("estimators is empty; nothing to compute")
    unknown = [e for e in estimators if e not in ESTIMATORS]
    if unknown:
        raise StabilityError(f"unknown estimators {unknown}; expected a subset of {ESTIMATORS}")
    if n_bootstrap < MIN_BOOTSTRAP:
        raise StabilityError(
            f"n_bootstrap must be >= {MIN_BOOTSTRAP} for a meaningful spread, got {n_bootstrap}"
        )
    if not 0.0 < alpha < 0.5:
        raise StabilityError(f"alpha must be in (0, 0.5), got {alpha}")

    rows: list[dict[str, object]] = []
    for horizon, frame in returns_by_horizon.items():
        if horizon not in _HORIZON_ORDER:
            raise StabilityError(f"unknown horizon {horizon!r}; expected one of {_HORIZON_ORDER}")
        windowed = restrict_window(frame, start, end)
        sample = windowed.dropna(how="any")
        if len(sample) < MIN_OBS:
            raise StabilityError(
                f"horizon {horizon}: only {len(sample)} complete-case rows in the window "
                f"(need >= {MIN_OBS}); the sample has {len(windowed)} rows over "
                f"{windowed.shape[1]} columns"
            )
        if sample.shape[1] < 3:
            raise StabilityError(
                f"horizon {horizon}: need at least 3 assets, got {sample.shape[1]}"
            )
        block = (
            default_mean_block_length(len(sample))
            if mean_block_length is None
            else mean_block_length
        )
        values = sample.to_numpy(dtype=float)
        index_sets = bootstrap_index_sets(len(sample), n_bootstrap, block, seed, horizon)
        window_start = str(sample.index[0].date())
        window_end = str(sample.index[-1].date())
        for name in estimators:
            cell = _cell(
                values,
                sample.columns,
                name,
                index_sets,
                ewma_lambda=ewma_lambda,
                n_market=n_market,
                alpha=alpha,
            )
            rows.append(
                {
                    "horizon": horizon,
                    "estimator": name,
                    **cell,
                    "n_bootstrap": n_bootstrap,
                    "mean_block_length": float(block),
                    "window_start": window_start,
                    "window_end": window_end,
                }
            )
    table = pd.DataFrame(rows, columns=list(TABLE_COLUMNS))
    table["n_obs"] = table["n_obs"].astype(int)
    table["n_assets"] = table["n_assets"].astype(int)
    table["is_psd"] = table["is_psd"].astype(bool)
    return table


def stability_summary_lines(table: pd.DataFrame) -> list[str]:
    """One printable line per table row, with the two caveats a reader needs next to the numbers."""
    missing = [c for c in TABLE_COLUMNS if c not in table.columns]
    if missing:
        raise StabilityError(f"table is missing columns {missing}")
    lines: list[str] = []
    for row in table.itertuples(index=False):
        psd = "PSD" if bool(row.is_psd) else "NOT PSD"
        note = "  [relative scale: compare rel_spread only]" if row.estimator == "detoned" else ""
        lines.append(
            f"{row.horizon} {row.estimator:<11} T={row.n_obs:<5} q={row.q:.3f} "
            f"cond={row.cond:>9.3g} {psd:<7} avg={row.avg_corr:+.3f} "
            f"boot_sd={row.boot_sd:.4f} rms_pair_sd={row.rms_pair_sd:.4f} "
            f"rel={row.rel_spread:.3f} eigvec={row.eigvec_overlap:.3f}{note}"
        )
    lines.append(
        "lower spread is lower variance, not higher accuracy (bias toward the identity or a "
        "noise floor is not in these columns); the accuracy comparison is H5 (Phase 8). "
        "Quarterly rows are exploratory (T close to N)."
    )
    return lines
