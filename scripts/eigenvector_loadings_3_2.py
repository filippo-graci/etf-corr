"""Exploratory check for Phase 3, step 3.2's H1 follow-up (not a Gate 1/Gate 2 requirement):
what do the real, non-market "signal" eigenvalues (beyond the market mode) actually load on?
Run with: uv run python scripts/eigenvector_loadings_3_2.py

Step 3.1's own real-data eigen_spectrum found more signal eigenvalues than the single market mode
H1's n_market=1 removes: 4 at the daily horizon, 2 at quarterly (core-23 panel). Per Plerou,
Gopikrishnan, Rosenow, Amaral, Guhr & Stanley (2002), "Random Matrix Approach to
Cross-Correlations in Financial Data," Phys. Rev. E 65, 066126 [CHECKED, see docs/H1_FINDINGS.md
section 2.4] -- eigenvectors beyond the market mode typically load heavily on assets from the
same or related industries, and one identifies a given eigenvector's economic meaning by
inspecting which assets have the largest-magnitude components ("loadings") on it. This script
does exactly that on the real core-23 daily and quarterly correlation matrices, printing the
top-loading tickers (and their configured sector) for eigenvalues ranked 2nd-4th (i.e. the ones
immediately after the market mode), so the pattern can be compared against config/universe.yaml's
own sector labels.

Purely descriptive/exploratory: this does not test H1 (docs/H1_FINDINGS.md section 4 is the
formal, multiplicity-corrected H1 result) and does not change any confirmatory result. Eigenvector
sign is arbitrary (numpy.linalg.eigh does not fix a sign convention), so only the *pattern* of
which tickers load together and with matching/opposing sign is meaningful, not the sign itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from etfcorr.config import find_repo_root
from etfcorr.correlation.estimators import estimate
from etfcorr.correlation.rmt import eigen_spectrum, spectrum_summary_line

N_TOP_LOADINGS = 8  # how many tickers to print per eigenvector, ranked by |loading|
N_EXTRA_EIGENVECTORS = 3  # inspect ranks 2..(1+N_EXTRA_EIGENVECTORS), i.e. 2nd-4th by default

root = find_repo_root()
with open(root / "config" / "universe.yaml") as f:
    universe = yaml.safe_load(f)
core_entries = [t for t in universe["tickers"] if t["panel"] == "core"]
core_tickers = [t["ticker"] for t in core_entries]
sector_of = {t["ticker"]: t["sector"] for t in core_entries}
print(f"Core universe: N={len(core_tickers)} tickers")
print()

for label, fname in (("DAILY", "returns_D.parquet"), ("QUARTERLY", "returns_Q.parquet")):
    print(f"=== {label} panel ===")
    df = pd.read_parquet(root / "data" / "processed" / fname)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(df.columns.levels[-1][0], axis=1, level=-1)
    cols = [t for t in core_tickers if t in df.columns]
    sub = df[cols].dropna()
    n_obs, n_assets = sub.shape
    print(f"  T={n_obs}, N={n_assets}")

    corr = estimate(sub, "pearson")
    spectrum = eigen_spectrum(corr, n_obs=n_obs)
    print(f"  {spectrum_summary_line(spectrum)}")

    values, vectors = np.linalg.eigh(corr.to_numpy(dtype=float))
    order = np.argsort(values)[::-1]
    eigenvalues = values[order]
    eigenvectors = vectors[:, order]
    n_available = min(1 + N_EXTRA_EIGENVECTORS, len(eigenvalues))

    for rank in range(1, n_available):  # rank 0 is the market mode, skip it
        eigval = eigenvalues[rank]
        vec = eigenvectors[:, rank]
        share = eigval / n_assets
        is_signal = eigval > spectrum.mp_edge
        top_idx = np.argsort(-np.abs(vec))[:N_TOP_LOADINGS]
        print(
            f"  eigenvalue rank {rank + 1}: value={eigval:.4f} "
            f"({100 * share:.1f}% of trace), "
            f"{'ABOVE' if is_signal else 'below'} MP edge {spectrum.mp_edge:.4f}"
        )
        for i in top_idx:
            ticker = cols[i]
            print(f"    {ticker:8s} sector={sector_of.get(ticker, '?'):15s} loading={vec[i]:+.4f}")
    print()
