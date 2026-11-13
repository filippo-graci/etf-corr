"""Synthetic methodology study (exploratory, not Gate 1/Gate 2 -- bears on any future H1-H6 work
using `bootstrap_horizon_pair_similarity`): does jointly resampling from a shared daily path, and
deriving the coarser horizon from that *same* resampled path, give a narrower confidence interval
than the library's current independent-per-horizon resampling -- as the module docstring's "Design
simplification, stated explicitly" predicts but has not, until now, tested?

Run with: uv run python scripts/bootstrap_joint_vs_independent_3_2.py

Background: `bootstrap_horizon_pair_similarity` block-resamples each horizon's returns
*independently* at every bootstrap replicate, a stated conservative simplification -- real D/W/M
returns share the same underlying daily price path, so a genuinely joint scheme (resample the
daily path once per replicate, re-derive every horizon from that one resampled path) would let
some of the resampling noise cancel through shared randomness, expected to *narrow*, not widen,
the reported interval. Implementing that properly inside the library would mean reaching into
`etfcorr.returns`'s resampling logic from inside a bootstrap loop, which the module's own
docstring says breaks its "pure function of already-loaded returns" contract -- so it has not been
built. This script tests the *prediction* on synthetic data with a known daily-to-weekly
relationship, without touching the library's contract: the joint scheme is implemented locally,
here, reusing only public library primitives (`stationary_bootstrap_indices`, `estimate`,
`horizon_pair_similarity`), never modifying `horizon_analysis.py` itself.

Method: `etfcorr.synthetic.make_factor_returns` generates a single daily panel (3 sectors of 5,
matching the sanity-tested setup already used in `scripts/eigenvector_loadings_3_2.py`,
market_loading=0.6, sector_loading=0.5). A synthetic "weekly" series is derived by summing every
5 consecutive daily log-returns (non-overlapping blocks) -- a simplified stand-in for the real
pipeline's Friday-close weekly resampling, disclosed as such, not a claim that it matches
`etfcorr.returns` exactly. Two resampling schemes are then compared at `n_market=1`,
`n_bootstrap=300`, 8 seeds:

- **Independent** (current library behaviour, unmodified): `bootstrap_horizon_pair_similarity`
  called as-is on the daily and weekly frames.
- **Joint** (local to this script only): one stationary-bootstrap index array is drawn over the
  *daily* calendar per replicate; the resampled daily returns and a re-derived resampled weekly
  series (same 5-day summation applied to the resampled daily path, so both come from one shared
  resampled history) are then passed through `estimate` + `horizon_pair_similarity` (the library's
  own point-estimate machinery, reused verbatim) to get one replicate's
  `frobenius_diff`/`eigvec_diff`; percentile CIs are built the same way `_percentile_ci` does
  internally (reimplemented as a two-line function here, since that helper is private).

Compares CI width (upper - lower) between the two schemes for both metrics. This is a
synthetic-data-only study: no real (vendor) data or repository-specific environment is involved,
so results are reproducible from a clean checkout given the fixed seeds, and were run directly in
this delivery (see the Delivery block) rather than requiring a separate local run.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from etfcorr.correlation.estimators import estimate
from etfcorr.correlation.horizon_analysis import (
    bootstrap_horizon_pair_similarity,
    default_mean_block_length,
    horizon_pair_similarity,
    stationary_bootstrap_indices,
)
from etfcorr.synthetic import make_factor_returns

N_DAYS = 1000  # divisible by 5, so the weekly aggregation below has no remainder
N_BOOTSTRAP = 300
ALPHA = 0.10
N_SEEDS = 8


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Non-overlapping 5-day log-return sums -- a simplified stand-in for real weekly resampling,
    disclosed as such (see the module docstring); not `etfcorr.returns`'s own Friday-close logic.
    """
    n_weeks = len(daily) // 5
    trimmed = daily.iloc[: n_weeks * 5]
    blocks = trimmed.to_numpy().reshape(n_weeks, 5, trimmed.shape[1]).sum(axis=1)
    return pd.DataFrame(blocks, columns=daily.columns)


def percentile_ci(replicates: np.ndarray, point: float, alpha: float) -> tuple[float, float]:
    lower = float(np.quantile(replicates, alpha / 2.0))
    upper = float(np.quantile(replicates, 1.0 - alpha / 2.0))
    return lower, upper


def joint_bootstrap(
    daily: pd.DataFrame, seed: int, n_bootstrap: int, alpha: float
) -> tuple[float, float, float, float]:
    """Returns (fro_width, eig_width, fro_point, eig_point) for the locally-implemented joint
    scheme: one daily-calendar resample per replicate, weekly re-derived from that same path."""
    weekly = to_weekly(daily)
    point = horizon_pair_similarity(
        estimate(daily, "pearson"), estimate(weekly, "pearson"), "D", "W", n_market=1
    )
    mbl = default_mean_block_length(len(daily))
    rng = np.random.default_rng(seed)
    fro_diffs = np.empty(n_bootstrap)
    eig_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = stationary_bootstrap_indices(len(daily), mbl, rng)
        boot_daily = pd.DataFrame(daily.to_numpy()[idx], columns=daily.columns)
        boot_weekly = to_weekly(boot_daily)
        pair = horizon_pair_similarity(
            estimate(boot_daily, "pearson"),
            estimate(boot_weekly, "pearson"),
            "D",
            "W",
            n_market=1,
        )
        fro_diffs[i] = pair.frobenius_diff
        eig_diffs[i] = pair.eigvec_diff
    fro_lo, fro_hi = percentile_ci(fro_diffs, point.frobenius_diff, alpha)
    eig_lo, eig_hi = percentile_ci(eig_diffs, point.eigvec_diff, alpha)
    return fro_hi - fro_lo, eig_hi - eig_lo, point.frobenius_diff, point.eigvec_diff


start = time.perf_counter()
print(
    f"3-sector synthetic panel (5 assets/sector, market_loading=0.6, sector_loading=0.5), "
    f"N_DAYS={N_DAYS}, n_bootstrap={N_BOOTSTRAP}, alpha={ALPHA}, {N_SEEDS} seeds"
)
print(
    f"{'seed':>5}  {'fro width indep':>16}  {'fro width joint':>16}  {'ratio':>7}  "
    f"{'eig width indep':>16}  {'eig width joint':>16}  {'ratio':>7}"
)

fro_ratios = []
eig_ratios = []
for seed in range(N_SEEDS):
    panel = make_factor_returns(
        N_DAYS, (5, 5, 5), market_loading=0.6, sector_loading=0.5, seed=seed
    )
    daily = panel.returns
    weekly = to_weekly(daily)

    _, fro_ci_indep, eig_ci_indep = bootstrap_horizon_pair_similarity(
        daily, weekly, "D", "W", n_market=1, n_bootstrap=N_BOOTSTRAP, alpha=ALPHA, seed=seed
    )
    fro_width_indep = fro_ci_indep.upper - fro_ci_indep.lower
    eig_width_indep = eig_ci_indep.upper - eig_ci_indep.lower

    fro_width_joint, eig_width_joint, _, _ = joint_bootstrap(daily, seed, N_BOOTSTRAP, ALPHA)

    fro_ratio = fro_width_joint / fro_width_indep
    eig_ratio = eig_width_joint / eig_width_indep
    fro_ratios.append(fro_ratio)
    eig_ratios.append(eig_ratio)
    print(
        f"{seed:>5}  {fro_width_indep:>16.4f}  {fro_width_joint:>16.4f}  {fro_ratio:>7.3f}  "
        f"{eig_width_indep:>16.4f}  {eig_width_joint:>16.4f}  {eig_ratio:>7.3f}"
    )

elapsed = time.perf_counter() - start
print()
print(
    f"mean ratio (joint/independent width): Frobenius {np.mean(fro_ratios):.3f}, "
    f"eigvec {np.mean(eig_ratios):.3f}  (<1 means joint is narrower, as the docstring predicts)"
)
print(f"wall-clock: {elapsed:.2f}s")
