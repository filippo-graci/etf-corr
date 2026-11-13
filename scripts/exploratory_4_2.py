"""Real-data run of the H2 engine (Phase 4, step 4.2). Run with:
``uv run python scripts/exploratory_4_2.py`` (about 2 minutes; ``--quick`` is a smoke run with 100
draws, not a result).

This is the descriptive, in-sample H2 table on the core panel: VIX full-sample terciles (step 4.1),
the log returns in ``data/processed/returns_D.parquet`` (step 2.1), ``data/interim/
prices_clean.parquet`` for the VIX level. It prints only; nothing is written.

**The specification below was fixed before this script was ever run on real data**
(docs/checks/4.2.md, docs/DECISION_LOG.md, 2026-10-02):

* primary: predetermined (t-1) regime labels, VIX full-sample terciles with the low tercile as the
  reference regime, cross-sector pairs, Pearson on daily log returns, Forbes-Rigobon adjustment
  with a per-asset delta averaged over the two directions of each pair, 2000 stationary-bootstrap
  draws, default mean block length, seed 0, 95% percentile intervals;
* every other line is a robustness variant or a contrast and is printed whatever it shows. They
  are not alternatives to pick from afterwards: the primary line is the confirmatory one, the rest
  qualify it. No verdict is computed here (that is step 4.6).

Read the lines with the calibration findings in mind (docs/checks/4.2.md): on simulated data the
per-asset-delta adjustment removes only part of a pure variance-driven increase (it is the
conservative variant for the claim that volatility explains the increase), a delta taken from a
market series removes most of it (and over-attributes when dependence also rises), and the
volatility-standardised correlation is close to the raw one by construction.
"""

import sys
import time

import pandas as pd
import yaml

from etfcorr.config import find_repo_root
from etfcorr.correlation.regime_correlation import (
    Alignment,
    DeltaMode,
    H2Result,
    PairScope,
    bootstrap_regime_correlations,
    h2_summary_lines,
    vol_standardize,
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
returns = returns_all[core_tickers]

prices = pd.read_parquet(root / "data" / "interim" / "prices_clean.parquet")
if not isinstance(prices.columns, pd.MultiIndex):
    raise SystemExit("expected (ticker, field) MultiIndex columns in prices_clean.parquet")
vix = prices["^VIX"]["close"].rename("^VIX")
labels = vix_terciles(vix, mode="full_sample")

smoke = "QUICK SMOKE RUN, NOT A RESULT. " if QUICK else ""
print(f"IN-SAMPLE, DESCRIPTIVE (no model fitted). {smoke}")
print(
    f"core panel N={len(core_tickers)}, {len(returns)} daily returns "
    f"{returns.index[0].date()} to {returns.index[-1].date()}; "
    f"VIX terciles from {int(labels.notna().sum())} levels"
)


def run(
    title: str,
    data: pd.DataFrame = returns,
    *,
    alignment: Alignment = "predetermined",
    pairs: PairScope = "cross",
    delta_mode: DeltaMode = "pair",
    delta_series: pd.Series | None = None,
    mean_block_length: float | None = None,
) -> H2Result:
    start = time.perf_counter()
    result = bootstrap_regime_correlations(
        data,
        labels,
        sectors,
        alignment=alignment,
        pairs=pairs,
        delta_mode=delta_mode,
        delta_series=delta_series,
        mean_block_length=mean_block_length,
        n_bootstrap=N_BOOT,
        seed=0,
    )
    print(f"\n--- {title} ({time.perf_counter() - start:.1f}s)")
    for line in h2_summary_lines(result):
        print(line)
    return result


primary = run("PRIMARY (pre-specified)")
n_rows = primary.point.n_rows

run(
    "variant: delta from the SPY return (market series)",
    delta_mode="series",
    delta_series=returns_all["SPY"],
)
run("variant: delta from the VIX log change", delta_mode="series", delta_series=returns_all["^VIX"])
run("variant: contemporaneous (day-t) labels", alignment="contemporaneous")
run(
    "variant: lagged-EWMA volatility-standardised returns (not a deflator, see docstring)",
    data=vol_standardize(returns),
)
run("contrast: within-sector pairs", pairs="within")
for mbl in (60.0, 120.0):
    run(
        f"sensitivity: mean block length {mbl:.0f} (default is the primary line's)",
        mean_block_length=mbl,
    )

print(f"\nprimary line used {n_rows} days; mean block length {primary.mean_block_length:.1f}")
