"""Synthetic methodology study (exploratory, not a Gate 1/Gate 2 requirement, and not H1-specific
-- this bears on any future step that calls `bootstrap_horizon_pair_similarity` at `n_market>1`,
including further H1 robustness work and any later re-use of the same engine): does the disclosed
Frobenius null bias (see `horizon_analysis.py`'s module docstring, "Known limitation, empirically
measured") get worse as `n_market` grows, and does the eigenvector-overlap metric -- which the
existing test suite confirms is well-calibrated at `n_market=1` -- stay well-calibrated at higher
`n_market` too? The latter question matters directly for `docs/H1_FINDINGS.md` section 4A's
`n_market=n_signal` result, which relied on eigenvector overlap being the trustworthy metric.

Run with: uv run python scripts/frobenius_bias_nmarket_3_2.py

Method: exactly the existing regression tests' null-noise setup
(`tests/unit/test_horizon_analysis.py::test_bootstrap_horizon_pair_similarity_null_noise_*`,
`etfcorr.synthetic.make_noise_returns(500, 8, seed=seed)`, "W" as a 5-day subsample of "D", so the
true population correlation is the identity matrix and any positive `frobenius_diff`/`eigvec_diff`
or CI exclusion of 0 is by construction a false positive, not a real effect), extended across
`n_market in (1, 2, 3, 4, 5)` (n_assets=8, so `n_market` must stay below 8) and more seeds (15,
vs. the tests' 5) at the tests' own bootstrap replicate count (150) for a steadier estimate of
exclusion frequency than the tests need for a pass/fail check. `alpha=0.10` throughout, matching
the existing eigvec-focused test's own choice, so a well-calibrated CI should exclude 0 in about
10% of seeds.

This is a synthetic-data-only study: no real (vendor) data or repository-specific environment is
involved, so results are reproducible from a clean checkout given the fixed seeds, and were run
directly in this delivery (see the Delivery block) rather than requiring a separate local run.
"""

from __future__ import annotations

import time

import numpy as np

from etfcorr.correlation.horizon_analysis import bootstrap_horizon_pair_similarity
from etfcorr.synthetic import make_noise_returns

N_SEEDS = 15
N_BOOTSTRAP = 150
ALPHA = 0.10
N_MARKET_VALUES = (1, 2, 3, 4, 5)

start = time.perf_counter()
print(
    f"Null-noise panel (n_assets=8, T=500, 'W'=5-day subsample of 'D'), "
    f"{N_SEEDS} seeds, n_bootstrap={N_BOOTSTRAP}, alpha={ALPHA}"
)
print(
    f"{'n_market':>8}  {'mean fro_diff':>14}  {'fro excl%':>10}  "
    f"{'mean eig_diff':>14}  {'eig excl%':>10}"
)

for n_market in N_MARKET_VALUES:
    fro_diffs = []
    eig_diffs = []
    fro_excl = 0
    eig_excl = 0
    for seed in range(N_SEEDS):
        panel = make_noise_returns(500, 8, seed=seed)
        returns_d = panel.returns
        returns_w = panel.returns.iloc[::5]
        point, fro_ci, eig_ci = bootstrap_horizon_pair_similarity(
            returns_d,
            returns_w,
            "D",
            "W",
            n_market=n_market,
            n_bootstrap=N_BOOTSTRAP,
            alpha=ALPHA,
            seed=seed,
        )
        fro_diffs.append(point.frobenius_diff)
        eig_diffs.append(point.eigvec_diff)
        fro_excl += int(fro_ci.excludes_zero)
        eig_excl += int(eig_ci.excludes_zero)
    print(
        f"{n_market:>8}  {np.mean(fro_diffs):>+14.4f}  {100 * fro_excl / N_SEEDS:>9.1f}%  "
        f"{np.mean(eig_diffs):>+14.4f}  {100 * eig_excl / N_SEEDS:>9.1f}%"
    )

elapsed = time.perf_counter() - start
print()
print(
    f"Under a well-calibrated {100 * (1 - ALPHA):.0f}% CI, excl% should be about "
    f"{100 * ALPHA:.0f}% by chance alone; a Frobenius entry already known biased at "
    f"n_market=1 (per the existing regression test) is expected near 100%."
)
print(f"wall-clock: {elapsed:.2f}s")
