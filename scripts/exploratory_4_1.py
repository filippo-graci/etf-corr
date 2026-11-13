"""Exploratory check for Phase 4, step 4.1 (not gating): what the regime labels look like on the
real panel. Run with: uv run python scripts/exploratory_4_1.py

Reads data/interim/prices_clean.parquet (step 1.3), prints only -- writes nothing. It answers
three practical questions before the H2 engine (step 4.2) is built on these labels:

1. Is ``^VIX`` usable as a *level* (positive, few gaps), and does its ``close`` equal
   ``adj_close``? (``returns.py`` reads ``adj_close``; an index has no dividends, so they should
   match, but that is checked here rather than assumed.)
2. How do the full-sample (descriptive, the confirmatory H2 sample) and expanding (causal)
   tercile labels compare: cutpoints, occupancy, persistence (number of regime episodes)?
3. How are SPY drawdown states distributed with the default (-5%, -20%) thresholds?

Nothing here is a result about H2: no correlation is computed.
"""

import numpy as np
import pandas as pd

from etfcorr.config import find_repo_root
from etfcorr.regimes import (
    DRAWDOWN_NAMES,
    TERCILE_NAMES,
    TERCILE_PROBS,
    drawdown_states,
    regime_summary,
    vix_terciles,
)

root = find_repo_root()
panel = pd.read_parquet(root / "data" / "interim" / "prices_clean.parquet")
if not isinstance(panel.columns, pd.MultiIndex):
    raise SystemExit("expected (ticker, field) MultiIndex columns in prices_clean.parquet")


def field(ticker: str, name: str) -> pd.Series:
    series = panel[ticker][name]
    return series.rename(ticker)


vix = field("^VIX", "close")
vix_adj = field("^VIX", "adj_close")
print(
    f"^VIX close: {len(vix)} rows, {int(vix.isna().sum())} NaN, "
    f"min {vix.min():.2f}, median {vix.median():.2f}, max {vix.max():.2f}"
)
both = pd.concat([vix, vix_adj], axis=1).dropna()
print(
    f"^VIX close vs adj_close: {len(both)} rows with both, "
    f"max abs diff {float((both.iloc[:, 0] - both.iloc[:, 1]).abs().max()):.3g}"
)
print(f"index {vix.index.min().date()} to {vix.index.max().date()}")

cuts = np.quantile(vix.dropna().to_numpy(), TERCILE_PROBS)
print(f"\nfull-sample VIX tercile cutpoints: {cuts[0]:.2f}, {cuts[1]:.2f}")

full = vix_terciles(vix, mode="full_sample")
expanding = vix_terciles(vix, mode="expanding", min_obs=252)
for label, series in (
    ("full_sample (descriptive, non-causal)", full),
    ("expanding (causal)", expanding),
):
    print(f"\n{label}: {int(series.notna().sum())} labelled of {len(series)}")
    summary = regime_summary(series, n_regimes=3)
    summary.index = pd.Index(TERCILE_NAMES, name="regime")
    print(summary.round(3).to_string())

agree = (full == expanding)[expanding.notna()].mean()
print(f"\nshare of dates where expanding and full-sample labels agree: {agree:.3f}")

spy = field("SPY", "adj_close")
states = drawdown_states(spy)
summary = regime_summary(states, n_regimes=3)
summary.index = pd.Index(DRAWDOWN_NAMES, name="state")
print(f"\nSPY adj_close drawdown states ({len(spy)} rows):")
print(summary.round(3).to_string())
