"""Exploratory check for Phase 3, step 3.1 (not a Gate 1/Gate 2 requirement -- informative before
step 3.2 designs the real H1 bootstrap engine). Run with: uv run python scripts/exploratory_3_1.py

Two things:
1. eigen_spectrum / denoise_corr / detone_corr on the real core-23 daily and quarterly panels.
2. Whether the installed skfolio actually exposes DenoiseCovariance/DetoneCovariance
   (project_plan.md section 6.4's named cross-check, not confirmed this session).
"""

from __future__ import annotations

import pandas as pd
import yaml

from etfcorr.config import find_repo_root
from etfcorr.correlation.estimators import estimate
from etfcorr.correlation.rmt import (
    denoise_corr,
    detone_corr,
    eigen_spectrum,
    spectrum_summary_line,
)

root = find_repo_root()

with open(root / "config" / "universe.yaml") as f:
    universe = yaml.safe_load(f)

core_entries = [t for t in universe["tickers"] if t["panel"] == "core"]
core_tickers = [t["ticker"] for t in core_entries]
sector_of = {t["ticker"]: t["sector"] for t in core_entries}

print(f"Core universe: N={len(core_tickers)} tickers")
by_sector: dict[str, list[str]] = {}
for tk in core_tickers:
    by_sector.setdefault(sector_of[tk], []).append(tk)
for sec, tks in by_sector.items():
    print(f"  {sec}: {tks}")
print()

for label, fname in (("DAILY", "returns_D.parquet"), ("QUARTERLY", "returns_Q.parquet")):
    print(f"=== {label} panel ===")
    returns = pd.read_parquet(root / "data" / "processed" / fname)
    if isinstance(returns.columns, pd.MultiIndex):
        returns = returns.xs(returns.columns.levels[-1][0], axis=1, level=-1)
    missing = [t for t in core_tickers if t not in returns.columns]
    if missing:
        print(f"  WARNING: missing from panel: {missing}")
    cols = [t for t in core_tickers if t in returns.columns]
    sub = returns[cols].dropna()
    n_obs, n_assets = sub.shape
    print(f"  T={n_obs}, N={n_assets}, q=N/T={n_assets / n_obs:.4f}")

    corr = estimate(sub, "pearson")
    spectrum = eigen_spectrum(corr, n_obs=n_obs)
    print(f"  {spectrum_summary_line(spectrum)}")

    denoised = denoise_corr(corr, n_obs=n_obs)
    print(f"  denoise_corr: trace={denoised.to_numpy().trace():.4f} (should equal N={n_assets})")

    try:
        detoned = detone_corr(corr, n_market=1)
        cross_pairs = [
            detoned.loc[a, b]
            for i, a in enumerate(cols)
            for b in cols[i + 1 :]
            if sector_of[a] != sector_of[b]
        ]
        same_pairs = [
            detoned.loc[a, b]
            for i, a in enumerate(cols)
            for b in cols[i + 1 :]
            if sector_of[a] == sector_of[b]
        ]
        neg_cross = sum(1 for v in cross_pairs if v < 0)
        neg_same = sum(1 for v in same_pairs if v < 0)
        print(
            f"  detone_corr: {neg_cross}/{len(cross_pairs)} cross-sector pairs negative "
            f"({100 * neg_cross / len(cross_pairs):.1f}%); "
            f"{neg_same}/{len(same_pairs)} within-sector pairs negative "
            f"({100 * neg_same / len(same_pairs) if same_pairs else 0:.1f}%)"
        )
    except Exception as e:
        print(f"  detone_corr: RAISED {type(e).__name__}: {e}")
    print()

print("=== skfolio API check ===")
try:
    import skfolio.moments as m

    names = [
        n for n in dir(m) if "denoise" in n.lower() or "deton" in n.lower() or "cov" in n.lower()
    ]
    print(f"  skfolio.moments version-relevant names: {names}")
    for cls_name in ("DenoiseCovariance", "DetoneCovariance"):
        exists = hasattr(m, cls_name)
        print(f"  skfolio.moments.{cls_name}: {'FOUND' if exists else 'NOT FOUND'}")
except ImportError as e:
    print(f"  skfolio import failed: {e}")
