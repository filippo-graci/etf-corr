"""Real-data run of the factor decomposition (Phase 4, step 4.2b, EXPLORATORY). Run with:
``uv run python scripts/exploratory_4_2b.py`` (about 1 minute; ``--quick`` is a smoke run with 100
draws, not a result).

It asks the second clause of H2 a delta-free question on the same sample as step 4.2: how much of
the rise in average cross-sector correlation in the high-VIX tercile would remain if only the
volatility of the common factors had changed (loadings and residual covariance held at their
low-VIX values)? Inputs, regime labels, pair scope, alignment and bootstrap settings are those of
``scripts/exploratory_4_2.py``; it prints only and writes nothing.

**Status: exploratory.** It is not part of the pre-registered H2 and does not change it. The
specification below was fixed before this script was ever run on real data (docs/checks/4.2b.md,
docs/DECISION_LOG.md, 2026-10-02):

* primary: the SPY return as the single factor, predetermined (t-1) VIX-tercile labels, low
  tercile as the reference, cross-sector pairs, 2000 stationary-bootstrap draws, seed 0, default
  mean block length, 95% percentile intervals;
* every other line is a robustness variant or a contrast and is printed whatever it shows: the
  equal-weight average of the core panel as the factor; SPY plus the first 2 and the first 4
  latent principal components of the residual returns (a full-sample, in-sample construction);
  day-t labels; within-sector pairs; mean block lengths 60 and 120. They qualify the primary line;
  none is chosen afterwards. No verdict is computed here (that is step 4.6, from the 4.2 table).

Read the lines with the limitation in mind: variance of a common factor that the chosen factors do
not span is booked as a change in residual covariance, and the sign of the resulting bias in the
volatility-only share is not known in advance (docs/checks/4.2b.md). The spread of the shares across
the factor sets is the information, not any one of them.
"""

import sys
import time

import pandas as pd
import yaml

from etfcorr.config import find_repo_root
from etfcorr.correlation.regime_correlation import Alignment, PairScope, regime_correlations
from etfcorr.correlation.regime_factor import (
    FactorBootstrapResult,
    bootstrap_regime_factor,
    factor_summary_lines,
    latent_factors,
)
from etfcorr.regimes import vix_terciles

QUICK = "--quick" in sys.argv
N_BOOT = 100 if QUICK else 2000

root = find_repo_root()
with open(root / "config" / "universe.yaml") as f:
    universe = yaml.safe_load(f)
core = [t for t in universe["tickers"] if t["panel"] == "core"]
core_tickers = [t["ticker"] for t in core]
sectors = {t["ticker"]: t["sector"] for t in core}

returns_all = pd.read_parquet(root / "data" / "processed" / "returns_D.parquet")
if isinstance(returns_all.columns, pd.MultiIndex):
    returns_all = returns_all.xs(returns_all.columns.levels[-1][0], axis=1, level=-1)
missing = [t for t in core_tickers if t not in returns_all.columns]
if missing:
    raise SystemExit(f"core tickers missing from returns_D.parquet: {missing}")
if "SPY" not in returns_all.columns:
    raise SystemExit("SPY is missing from returns_D.parquet (the primary factor)")
returns = returns_all[core_tickers]

prices = pd.read_parquet(root / "data" / "interim" / "prices_clean.parquet")
if not isinstance(prices.columns, pd.MultiIndex):
    raise SystemExit("expected (ticker, field) MultiIndex columns in prices_clean.parquet")
vix = prices["^VIX"]["close"].rename("^VIX")
labels = vix_terciles(vix, mode="full_sample")

spy = returns_all["SPY"].rename("SPY")
equal_weight = returns.mean(axis=1).rename("EW")
pcs = latent_factors(returns, spy, 4)

smoke = "QUICK SMOKE RUN, NOT A RESULT. " if QUICK else ""
print(f"EXPLORATORY, IN-SAMPLE, DESCRIPTIVE (no model fitted). {smoke}")
print(
    f"core panel N={len(core_tickers)}, {len(returns)} daily returns "
    f"{returns.index[0].date()} to {returns.index[-1].date()}; "
    f"VIX terciles from {int(labels.notna().sum())} levels; SPY in the core panel: "
    f"{'SPY' in core_tickers}"
)


def run(
    title: str,
    factors: pd.DataFrame | pd.Series,
    *,
    alignment: Alignment = "predetermined",
    pairs: PairScope = "cross",
    mean_block_length: float | None = None,
) -> FactorBootstrapResult:
    start = time.perf_counter()
    result = bootstrap_regime_factor(
        returns,
        labels,
        sectors,
        factors,
        alignment=alignment,
        pairs=pairs,
        mean_block_length=mean_block_length,
        n_bootstrap=N_BOOT,
        seed=0,
    )
    print(f"\n--- {title} ({time.perf_counter() - start:.1f}s)")
    for line in factor_summary_lines(result):
        print(line)
    return result


primary = run("PRIMARY (pre-specified): SPY return as the single factor", spy)
run("variant: equal-weight average of the core panel as the factor", equal_weight)
run("variant: SPY + 2 latent principal components", pd.concat([spy, pcs[["PC1", "PC2"]]], axis=1))
run("variant: SPY + 4 latent principal components", pd.concat([spy, pcs], axis=1))
run("variant: contemporaneous (day-t) labels", spy, alignment="contemporaneous")
run("contrast: within-sector pairs", spy, pairs="within")
for mbl in (60.0, 120.0):
    run(
        f"sensitivity: mean block length {mbl:.0f} (the primary line's default is 15)",
        spy,
        mean_block_length=mbl,
    )

# Cross-check against the LOCAL-PASSED 4.2 engine: the regime-wise raw averages must coincide, and
# the Shapley terms must add up to the raw change, for every factor set.
engine = regime_correlations(returns, labels, sectors).table
gap = (primary.point.table["avg_raw"] - engine["avg_raw"]).abs().max()
print(f"\ncross-check vs the 4.2 engine: max |avg_raw difference| = {gap:.2e} (expect ~1e-15)")
for g in primary.point.table.index[1:]:
    shap_sum = sum(
        primary.stat(name, int(g)).point
        for name in ("shap_factor_variance", "shap_loadings", "shap_residual_covariance")
    )
    print(
        f"identity, regime {int(g)}: |Shapley sum - raw change| = "
        f"{abs(shap_sum - primary.stat('diff_raw', int(g)).point):.2e} (expect ~1e-16)"
    )
print(
    f"primary line used {primary.point.n_rows} days; "
    f"mean block length {primary.mean_block_length:.1f}"
)
