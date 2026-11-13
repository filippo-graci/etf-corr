"""Real-data look at the forward-filtered HMM regimes (Phase 4, step 4.3). Run with:
``uv run python scripts/exploratory_4_3.py`` (about one to two minutes; ``--quick`` runs one null
shuffle instead of five and is a smoke run, not a result).

Reads ``data/processed/returns_D.parquet`` (step 2.1), ``config/universe.yaml`` (core panel and
sectors) and ``data/interim/prices_clean.parquet`` (the VIX level, used only to *describe* the
regimes, never to fit them). Prints only; nothing is written. **Descriptive, not a test of any
hypothesis, and not part of the pre-registered H2.**

Specification fixed before this script was ever run on real data (docs/checks/4.3.md):

* observations (``hmm_observations``): trailing 63-day log mean volatility and mean Fisher-z
  cross-sector correlation of the core panel's daily log returns;
* primary model: **2 states**, first fit on 504 valid rows, refit every 63 rows on an expanding
  window, 5 cold starts for the first fit and warm starts after, seed 0, states ordered by the
  log-volatility mean (state 0 calm, state 1 stress);
* printed variants (not alternatives to choose from afterwards): 3 states; and the null.

**The null** is the same pipeline run on the real returns with the *rows shuffled* (seeded): it
keeps every marginal distribution and the contemporaneous cross-sectional structure and destroys
serial dependence, hence volatility clustering and any regime. Because the observations are
trailing-window statistics, the HMM reports persistent, separated states even there
(``scripts/hmm_calibration_4_3.py``), so a real-data fit means something only to the extent that it
stands out from this null. The comparison is descriptive: five shuffles are a rough yardstick, not
a p-value.

The filter-versus-smoother line uses the *last* refit's parameters for both posteriors, so it
isolates what smoothing changes; it is an illustration of the look-ahead, not a model result.
"""

import sys
import time

import numpy as np
import pandas as pd
import yaml

from etfcorr.config import find_repo_root
from etfcorr.regime_hmm import (
    HmmRegimes,
    forward_filtered_hmm,
    forward_probabilities,
    hmm_observations,
    hmm_summary_lines,
    smoothed_probabilities,
)
from etfcorr.regimes import TERCILE_NAMES, regime_summary, vix_terciles

QUICK = "--quick" in sys.argv
N_NULL = 1 if QUICK else 5

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

smoke = "QUICK SMOKE RUN, NOT A RESULT. " if QUICK else ""
print(f"DESCRIPTIVE, NOT A HYPOTHESIS TEST. {smoke}")
print(
    f"core panel N={len(core_tickers)}, {len(returns)} daily returns "
    f"{returns.index[0].date()} to {returns.index[-1].date()}"
)


def stats_line(out: HmmRegimes) -> str:
    fits = out.fits
    return (
        f"separation min/median/max {fits['min_separation'].min():.2f}/"
        f"{fits['min_separation'].median():.2f}/{fits['min_separation'].max():.2f}; "
        f"p_stay medians "
        + "/".join(f"{fits[f'p_stay_{j}'].median():.3f}" for j in range(out.probs.shape[1]))
    )


start = time.perf_counter()
obs = hmm_observations(returns, sectors)
n_obs = int(obs.notna().all(axis=1).sum())
print(
    f"\nobservations: {len(obs)} rows, {n_obs} complete (first {obs.dropna().index[0].date()}); "
    f"log_vol mean {obs['log_vol'].mean():.3f}, z_cross_corr mean {obs['z_cross_corr'].mean():.3f}"
)

primary = forward_filtered_hmm(obs, 2)
print(f"\n--- PRIMARY: 2 states ({time.perf_counter() - start:.1f}s)")
for line in hmm_summary_lines(primary):
    print(line)
labels = primary.labels
print(
    f"labelled rows: {int(labels.notna().sum())} "
    f"(first {labels.dropna().index[0].date()}, last {labels.dropna().index[-1].date()})"
)
summary = regime_summary(labels, n_regimes=2)
summary.index = pd.Index(("calm", "stress"), name="state")
print(summary.round(3).to_string())

print("\nrelation to the VIX level (descriptive; the VIX was not used to fit the HMM)")
frame = pd.concat([labels.rename("state"), vix.rename("vix")], axis=1, sort=True).dropna()
by_state = frame.groupby("state")["vix"].agg(["count", "median", "mean"])
print(by_state.round(2).to_string())
full_terciles = vix_terciles(vix, mode="full_sample").reindex(frame.index)
share = pd.crosstab(full_terciles, frame["state"], normalize="index")
share.index = pd.Index([TERCILE_NAMES[int(i)] for i in share.index], name="VIX tercile")
print("share of days in each HMM state, by full-sample VIX tercile (rows sum to 1):")
print(share.round(3).to_string())

if primary.last_params is not None:
    x = obs.dropna().to_numpy(dtype=float)
    first_fit = obs.dropna().index.get_loc(primary.fits.index[0])
    filtered, _ = forward_probabilities(x, primary.last_params)
    smoothed = smoothed_probabilities(x, primary.last_params)
    differ = filtered[first_fit:].argmax(axis=1) != smoothed[first_fit:].argmax(axis=1)
    print(
        f"\nsmoothed vs filtered, last refit's parameters, {len(differ)} labelled rows: "
        f"hard labels differ on {float(differ.mean()):.3f} of them "
        f"(mean |probability difference| "
        f"{float(np.abs(filtered[first_fit:] - smoothed[first_fit:]).mean()):.3f})"
    )

start = time.perf_counter()
three = forward_filtered_hmm(obs, 3)
print(f"\n--- VARIANT: 3 states ({time.perf_counter() - start:.1f}s)")
for line in hmm_summary_lines(three):
    print(line)
summary3 = regime_summary(three.labels, n_regimes=3)
summary3.index = pd.Index(("calm", "middle", "stress"), name="state")
print(summary3.round(3).to_string())

print(f"\n--- NULL: {N_NULL} row-shuffled copies of the real returns, same pipeline, 2 states")
print(f"real : {stats_line(primary)}")
for seed in range(N_NULL):
    rng = np.random.default_rng(seed)
    shuffled = pd.DataFrame(
        returns.to_numpy()[rng.permutation(len(returns))],
        index=returns.index,
        columns=returns.columns,
    )
    null_out = forward_filtered_hmm(hmm_observations(shuffled, sectors), 2)
    print(f"null {seed}: {stats_line(null_out)}")
