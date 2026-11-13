"""Exploratory check for Phase 3, step 3.2 (not a Gate 1/Gate 2 requirement -- the real-data H1
cross-horizon similarity/bootstrap-CI result, on the actual core panel, not a synthetic
illustration). Run with: uv run python scripts/exploratory_3_2.py

Uses config/experiment.yaml's n_bootstrap=2000 default. This is the actual confirmatory H1 test
on real data -- report whatever comes out, including a result that does not support H1, per the
project's evidence rules. Not gating: step 3.2's LOCAL-PASSED status does not depend on this
script's output.

Restricted to the core panel (config/universe.yaml, panel == "core"), following
scripts/exploratory_3_1.py's own established pattern. data/processed/returns_{D,W,M}.parquet
holds the full 56-ticker universe (core + extended + control); estimate() restricts to
complete-case rows across all *requested* columns, and the full universe's complete-case overlap
is constrained by its latest-starting member (IBIT, 2024-01-11), which at the monthly horizon
leaves too few rows for N=56 -- passing all 56 columns directly (an earlier version of this
script's own mistake, not a bug in horizon_analysis.py/rmt.py) produces a near-rank-1 correlation
matrix and a RMTError from detone_corr for every ticker. The core-23 panel has no such
late-starting member (common start 2014-01-02, confirmed at step 1.1b) and is what step 3.1's own
exploratory checks already validated as well-conditioned at all three horizons.
"""

from __future__ import annotations

import time

import pandas as pd
import yaml

from etfcorr.config import find_repo_root
from etfcorr.correlation.horizon_analysis import h1_cross_horizon_test, h1_pair_summary_line

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
    missing = [t for t in core_tickers if t not in df.columns]
    if missing:
        print(f"  WARNING: missing from {h} panel: {missing}")
    cols = [t for t in core_tickers if t in df.columns]
    returns[h] = df[cols]
    print(f"  {h}: {df.shape[0]} raw rows, {len(cols)} core tickers")

start = time.perf_counter()
result = h1_cross_horizon_test(returns, "pearson", n_bootstrap=2000, seed=0)
elapsed = time.perf_counter() - start

print()
for _pair, r in result.items():
    print(h1_pair_summary_line(r))
print()
print(f"wall-clock: {elapsed:.2f}s")
