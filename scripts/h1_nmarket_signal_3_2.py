"""Robustness check for Phase 3, step 3.2's H1 result: what happens if detoning removes each
horizon's own real signal-eigenvalue count (``n_market = n_signal``), instead of just the single
market mode (``n_market = 1``) that ``docs/H1_FINDINGS.md`` sections 3-4 use? Not a Gate 1/Gate 2
requirement -- an exploratory sensitivity check listed as an open follow-up in
``docs/H1_FINDINGS.md`` section 6.
Run with: uv run python scripts/h1_nmarket_signal_3_2.py

Motivation: step 3.1's real-data ``eigen_spectrum`` found more "signal" eigenvalues (above the
Marchenko-Pastur noise edge) than the single market mode H1's baseline (``n_market=1``) removes:
4 at the daily horizon on the core-23 panel, 2 at quarterly. Section 2.4's eigenvector-loading
inspection (``scripts/eigenvector_loadings_3_2.py``) identified what daily's ranks 2-4 actually
load on: metals_mining (rank 2, above the MP edge), energy_trad (rank 3, above the MP edge), and a
mixed healthcare-vs-(clean_energy+ai_semis) style axis (rank 4, barely above the MP edge). If those
extra eigenvalues are real sector structure rather than noise, removing only the market mode may
leave sector-level structure in the "detoned" matrix that differs across horizons for reasons
unrelated to H1's market-mode-removal mechanism -- this script checks whether the H1 comparison
changes materially once each horizon's own detected signal count is removed instead.

API constraint, stated explicitly: ``bootstrap_horizon_pair_similarity`` (and
``h1_cross_horizon_test``, which calls it) takes exactly one ``n_market`` shared by both horizons
in a pair -- ``detone_corr`` is applied separately to each horizon's matrix but with the same
mode count, since ``HorizonPairSimilarity``/``horizon_pair_similarity`` were designed around one
detoning depth per comparison, not one per horizon. There is no way to ask for an asymmetric
"remove 4 from daily, 2 from quarterly" detoning within a single pair through the existing public
API. **Design choice, disclosed**: for each pair this script uses
``n_market_pair = max(n_signal[horizon_a], n_signal[horizon_b])`` -- the more conservative
(larger) of the two horizons' own detected signal counts, so neither horizon is under-detoned
relative to what its own spectrum suggests. This changes ``n_top`` too, since
``horizon_pair_similarity`` defaults ``n_top = n_market`` (the eigenvector-overlap metric compares
exactly the modes being removed).

This script re-derives ``n_signal`` for D, W and M fresh (only D and Q were computed in step 3.1;
W and M were not previously reported), via the same ``eigen_spectrum`` call
``eigenvector_loadings_3_2.py`` uses, on the identical core-23 panel and complete-case row count
``estimate``/``bootstrap_horizon_pair_similarity`` would use internally (via ``corr_report``, so
``n_obs`` matches exactly, not an approximation).

Calls ``bootstrap_horizon_pair_similarity`` directly per pair (not ``h1_cross_horizon_test``,
which applies one global ``n_market`` to every pair), mirroring that function's own internal
``seed=seed+offset`` pattern (``itertools.combinations`` order) for reproducibility, and reuses
the Holm-Bonferroni multiplicity correction from ``scripts/h1_table_3_2.py`` (same two families:
within-metric, and all 6 tests together) so the result is directly comparable to
``docs/H1_FINDINGS.md`` sections 3-4's ``n_market=1`` baseline.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
import yaml

from etfcorr.config import find_repo_root
from etfcorr.correlation.estimators import corr_report
from etfcorr.correlation.horizon_analysis import (
    DEFAULT_ALPHA,
    DEFAULT_N_BOOTSTRAP,
    H1PairResult,
    bootstrap_horizon_pair_similarity,
    h1_pair_summary_line,
)
from etfcorr.correlation.rmt import eigen_spectrum, spectrum_summary_line


def bootstrap_p_value(replicates: tuple[float, ...]) -> float:
    """Two-sided percentile bootstrap p-value for H0: true before/after difference = 0.

    [Inference / standard practice, not citation-specific] -- identical to
    ``scripts/h1_table_3_2.py``'s own function, duplicated here to keep each exploratory script
    self-contained (this repo's established convention: no shared imports between scripts/*.py).
    """
    arr = np.asarray(replicates, dtype=float)
    p_le = float(np.mean(arr <= 0.0))
    p_ge = float(np.mean(arr >= 0.0))
    return min(1.0, 2.0 * min(p_le, p_ge))


def holm_bonferroni(
    p_values: dict[str, float], alpha: float = DEFAULT_ALPHA
) -> dict[str, tuple[float, bool]]:
    """Holm-Bonferroni step-down correction. Returns {label: (p_raw, holm_significant)}."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, tuple[float, bool]] = {}
    still_significant = True
    for rank, (label, p) in enumerate(items, start=1):
        threshold = alpha / (m - rank + 1)
        significant = still_significant and p <= threshold
        out[label] = (p, significant)
        if not significant:
            still_significant = False
    return out


root = find_repo_root()
with open(root / "config" / "universe.yaml") as f:
    universe = yaml.safe_load(f)
core_tickers = [t["ticker"] for t in universe["tickers"] if t["panel"] == "core"]
print(f"Core universe: N={len(core_tickers)} tickers")

returns: dict[str, pd.DataFrame] = {}
for h in ("D", "W", "M"):
    df = pd.read_parquet(root / "data" / "processed" / f"returns_{h}.parquet")
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(df.columns.levels[-1][0], axis=1, level=-1)
    missing = [t for t in core_tickers if t not in df.columns]
    if missing:
        print(f"  WARNING: missing from {h} panel: {missing}")
    cols = [t for t in core_tickers if t in df.columns]
    returns[h] = df[cols]
    print(f"  {h}: {df.shape[0]} raw rows, {len(cols)} core tickers")

print()
print("=== Step 1: real n_signal per horizon (core-23 panel) ===")
n_signal: dict[str, int] = {}
for h in ("D", "W", "M"):
    report = corr_report(returns[h], "pearson")
    spectrum = eigen_spectrum(report.corr, n_obs=report.n_obs)
    n_signal[h] = spectrum.n_signal
    print(f"  {h}: {spectrum_summary_line(spectrum)}")

print()
print("=== Step 2: n_market_pair = max(n_signal[a], n_signal[b]) per pair ===")
n_market_pair: dict[tuple[str, str], int] = {}
for a, b in itertools.combinations(("D", "W", "M"), 2):
    n_market_pair[(a, b)] = max(n_signal[a], n_signal[b])
    print(
        f"  {a}-{b}: n_signal[{a}]={n_signal[a]}, n_signal[{b}]={n_signal[b]} "
        f"-> n_market={n_market_pair[(a, b)]}"
    )

print()
print("=== Step 3: H1 pairwise test with per-pair n_market = n_signal (bootstrap, n=2000) ===")
start = time.perf_counter()
result: dict[tuple[str, str], H1PairResult] = {}
for offset, (a, b) in enumerate(itertools.combinations(("D", "W", "M"), 2)):
    nm = n_market_pair[(a, b)]
    point, fro_ci, eig_ci = bootstrap_horizon_pair_similarity(
        returns[a],
        returns[b],
        a,
        b,
        "pearson",
        n_market=nm,
        n_bootstrap=DEFAULT_N_BOOTSTRAP,
        seed=offset,
    )
    result[(a, b)] = H1PairResult(similarity=point, frobenius_ci=fro_ci, eigvec_ci=eig_ci)
elapsed = time.perf_counter() - start

print()
for _pair, r in result.items():
    print(h1_pair_summary_line(r))

fro_p: dict[str, float] = {}
eig_p: dict[str, float] = {}
for pair, r in result.items():
    label = f"{pair[0]}-{pair[1]}"
    fro_p[label] = bootstrap_p_value(r.frobenius_ci.replicates)
    eig_p[label] = bootstrap_p_value(r.eigvec_ci.replicates)

fro_holm = holm_bonferroni(fro_p)
eig_holm = holm_bonferroni(eig_p)

print()
print(f"=== Holm-Bonferroni within each metric (family size m=3, alpha={DEFAULT_ALPHA}) ===")
print("Frobenius distance:")
for label, (p, sig) in sorted(fro_holm.items()):
    print(f"  {label}: p={p:.4f}  Holm-significant={sig}")
print("Eigenvector overlap:")
for label, (p, sig) in sorted(eig_holm.items()):
    print(f"  {label}: p={p:.4f}  Holm-significant={sig}")

all_p = {f"fro-{k}": v for k, v in fro_p.items()} | {f"eig-{k}": v for k, v in eig_p.items()}
all_holm = holm_bonferroni(all_p)
print()
print("=== Conservative alternative: Holm across all 6 tests (3 pairs x 2 metrics) ===")
for label, (p, sig) in sorted(all_holm.items()):
    print(f"  {label}: p={p:.4f}  Holm-significant={sig}")

print()
print(f"wall-clock: {elapsed:.2f}s")
