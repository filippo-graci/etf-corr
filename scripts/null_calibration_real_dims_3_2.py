"""Synthetic methodology study (exploratory, not Gate 1/Gate 2 -- but directly bears on how much
to trust `docs/H1_FINDINGS.md` sections 3-4 and 4A): what is the eigenvector-overlap and Frobenius
null-bias noise floor at the *actual* project dimensions (N=23 core tickers, and the real D/W/M
row counts), rather than the existing test suite's legacy N=8 fixture?

Run with: uv run python scripts/null_calibration_real_dims_3_2.py

Motivation: `scripts/frobenius_bias_nmarket_3_2.py` (this session, N=8, T=500/100) found that the
eigenvector-overlap metric's point estimate carries a small negative bias under a true null (mean
`eigvec_diff` ranging -0.02 to -0.21 across n_market=1-5), even though its 90% CI essentially never
excludes 0 there -- a nuance the existing test suite never checked, since it only asks whether the
CI excludes 0, not how the point estimate's magnitude compares to a real effect. This matters
directly: `docs/H1_FINDINGS.md`'s real point estimates are similarly small (-0.03 to -0.06 at
`n_market=1`; -0.03 to -0.06 at `n_market=n_signal`) -- so the open question is whether the real
finding's magnitude clears this null-bias noise floor, and the N=8/T=500 fixture is the wrong
scale to answer that: the real study uses N=23 and very different T per horizon (D=3201,
W=664, M=151), and null-bias magnitude is expected to depend on both N and T (Marchenko-Pastur
noise scales with N/T).

Method: independent noise panels (`etfcorr.synthetic.make_noise_returns`, population correlation =
identity -- true `frobenius_diff`/`eigvec_diff` = 0 by construction under any `n_market`, since
there is no real structure to reveal by detoning) at N=23 and the *actual* row counts used in the
real H1 run for each horizon pair, with `n_market` set to the same values `docs/H1_FINDINGS.md`
actually used: `n_market=1` for every pair (section 3-4), and the section 4A `n_market=n_signal`
values (D-W=4, D-M=4, W-M=3). 12 seeds, `n_bootstrap=250`, `alpha=0.10` (matching the existing
project convention for this kind of null check). Unlike the D-vs-5-day-subsampled-W setup in the
existing test suite (where "W" is derived from the same realized "D" draw), this uses two fully
independent noise panels per pair, since the real D/W/M return panels are independently estimated
from resampled prices, not literally subsampled from each other -- matching the real data
generating process more closely than the legacy fixture does.

This is a synthetic-data-only study: no real (vendor) data or repository-specific environment is
involved, so results are reproducible from a clean checkout given the fixed seeds, and were run
directly in this delivery (see the Delivery block) rather than requiring a separate local run.
"""

from __future__ import annotations

import time

import numpy as np

from etfcorr.correlation.horizon_analysis import bootstrap_horizon_pair_similarity
from etfcorr.synthetic import make_noise_returns

N_ASSETS = 23  # matches the real core panel
N_SEEDS = 12
N_BOOTSTRAP = 250
ALPHA = 0.10

# (label, T_a, T_b, n_market at the confirmatory n_market=1 spec is always 1;
#  n_market_signal is the section 4A max(n_signal_a, n_signal_b) value actually used)
PAIRS = (
    ("D-W", 3201, 664, 4),
    ("D-M", 3201, 151, 4),
    ("W-M", 664, 151, 3),
)

# real point estimates already reported in docs/H1_FINDINGS.md, for direct comparison below
REAL_EIGVEC_DIFF_NMARKET1 = {"D-W": -0.0542, "D-M": -0.0415, "W-M": -0.0340}
REAL_EIGVEC_DIFF_NSIGNAL = {"D-W": -0.0308, "D-M": -0.0609, "W-M": -0.0309}


def run(label: str, t_a: int, t_b: int, n_market: int) -> tuple[float, float, int, int]:
    fro_diffs = []
    eig_diffs = []
    fro_excl = 0
    eig_excl = 0
    for seed in range(N_SEEDS):
        panel_a = make_noise_returns(t_a, N_ASSETS, seed=1000 * seed)
        panel_b = make_noise_returns(t_b, N_ASSETS, seed=1000 * seed + 1)
        point, fro_ci, eig_ci = bootstrap_horizon_pair_similarity(
            panel_a.returns,
            panel_b.returns,
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
    return float(np.mean(fro_diffs)), float(np.mean(eig_diffs)), fro_excl, eig_excl


start = time.perf_counter()
print(
    f"Independent noise panels, N={N_ASSETS}, {N_SEEDS} seeds, n_bootstrap={N_BOOTSTRAP}, "
    f"alpha={ALPHA}"
)
print()

print("=== n_market=1 (matches docs/H1_FINDINGS.md section 3-4) ===")
print(
    f"{'pair':>6}  {'T_a':>6}  {'T_b':>6}  {'mean fro_diff':>14}  {'fro excl%':>10}  "
    f"{'mean eig_diff':>14}  {'eig excl%':>10}  {'real eig_diff':>14}"
)
for label, t_a, t_b, _ in PAIRS:
    mfro, meig, fro_excl, eig_excl = run(label, t_a, t_b, 1)
    real = REAL_EIGVEC_DIFF_NMARKET1[label]
    print(
        f"{label:>6}  {t_a:>6}  {t_b:>6}  {mfro:>+14.4f}  {100 * fro_excl / N_SEEDS:>9.1f}%  "
        f"{meig:>+14.4f}  {100 * eig_excl / N_SEEDS:>9.1f}%  {real:>+14.4f}"
    )

print()
print("=== n_market=n_signal (matches docs/H1_FINDINGS.md section 4A) ===")
print(
    f"{'pair':>6}  {'T_a':>6}  {'T_b':>6}  {'n_mkt':>5}  {'mean fro_diff':>14}  "
    f"{'fro excl%':>10}  {'mean eig_diff':>14}  {'eig excl%':>10}  {'real eig_diff':>14}"
)
for label, t_a, t_b, n_market in PAIRS:
    mfro, meig, fro_excl, eig_excl = run(label, t_a, t_b, n_market)
    real = REAL_EIGVEC_DIFF_NSIGNAL[label]
    print(
        f"{label:>6}  {t_a:>6}  {t_b:>6}  {n_market:>5}  {mfro:>+14.4f}  "
        f"{100 * fro_excl / N_SEEDS:>9.1f}%  {meig:>+14.4f}  {100 * eig_excl / N_SEEDS:>9.1f}%  "
        f"{real:>+14.4f}"
    )

elapsed = time.perf_counter() - start
print()
print(
    "'real eig_diff' is the point estimate already reported on real data (docs/H1_FINDINGS.md); "
    "compare its magnitude and sign against the null-noise mean and excl% columns above."
)
print(f"wall-clock: {elapsed:.2f}s")
