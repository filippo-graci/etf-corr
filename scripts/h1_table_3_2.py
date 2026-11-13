"""Multiplicity-corrected H1 summary table for Phase 3, step 3.2 (exploratory follow-up to
scripts/exploratory_3_2.py's uncorrected real-data run -- not a Gate 1/Gate 2 requirement).
Run with: uv run python scripts/h1_table_3_2.py

scripts/exploratory_3_2.py reported per-pair 95% CIs but explicitly flagged that no
multiple-comparison correction had been applied across the 3 simultaneous horizon-pair tests
(D-W, D-M, W-M) -- h1_cross_horizon_test's own docstring is explicit that it returns *uncorrected*
per-pair CIs and defers any correction to a later scoreboard step. This script fills that gap for
step 3.2's own result (not the eventual H1-H6 cross-hypothesis correction named in
config/experiment.yaml's inference.multiplicity=holm, which is a separate, larger correction left
to Phase 8's scoreboard, not built yet).

Method: for each (pair, metric), compute a two-sided percentile bootstrap p-value directly from
the stored replicate distribution (BootstrapCI.replicates) -- p = 2 * min(P(replicate <= 0),
P(replicate >= 0)), capped at 1.0. This is the standard, generic nonparametric bootstrap
hypothesis-testing approach (inverting a percentile CI into a p-value); it is not itself drawn
from a specific citation, and is disclosed as [Inference]/standard practice, not attributed to a
paper. Holm-Bonferroni step-down correction (Holm 1979, "A Simple Sequentially Rejective
Multiple Test Procedure", Scand. J. Statist. 6(2):65-70 -- CHECKED 2026-11-02 via its JSTOR record)
is then applied twice: (a) the primary table, within each metric separately (family size m=3: the 3
horizon pairs), since Frobenius distance and eigenvector overlap are different, non-comparable
statistics addressing the same qualitative question, not one undifferentiated family; (b) a more
conservative alternative treating all 6 tests (3 pairs x 2 metrics) as one family, reported for
comparison. Both use config/experiment.yaml's own alpha=0.05.

Uses the same core-23 panel, n_market=1, n_bootstrap=2000, seed=0 as
scripts/exploratory_3_2.py, so the p-values/corrections below are for that exact run, not a new
one.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import yaml

from etfcorr.config import find_repo_root
from etfcorr.correlation.horizon_analysis import (
    DEFAULT_ALPHA,
    h1_cross_horizon_test,
    h1_pair_summary_line,
)


def bootstrap_p_value(replicates: tuple[float, ...]) -> float:
    """Two-sided percentile bootstrap p-value for H0: true before/after difference = 0.

    [Inference / standard practice, not citation-specific]: p = 2 * min(P(replicate <= 0),
    P(replicate >= 0)), capped at 1.0.
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

returns = {}
for h in ("D", "W", "M"):
    df = pd.read_parquet(root / "data" / "processed" / f"returns_{h}.parquet")
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(df.columns.levels[-1][0], axis=1, level=-1)
    cols = [t for t in core_tickers if t in df.columns]
    returns[h] = df[cols]

start = time.perf_counter()
result = h1_cross_horizon_test(returns, "pearson", n_bootstrap=2000, seed=0)
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
