"""Which asset and refit sit at the GARCH persistence bound in step 4.4? Run with:
``uv run python scripts/garch_bound_4_4.py`` (a few minutes; univariate fits only, no DCC stage).

The real-data run of ``exploratory_4_4.py`` reported a largest single GARCH persistence of 1.0000,
i.e. ``alpha + beta`` at the cap ``1 - 1e-6`` (``dcc.MAX_PERSISTENCE``) for at least one asset in at
least one refit, without saying which. This script repeats exactly the univariate fits of the
primary ``rolling_dcc`` run (Student-t, ``min_obs=504``, ``refit_every=63``, expanding window, each
fit warm-started from the same asset's previous fit with ``n_starts=2``, the first fit cold), which
depend on no other stage, and prints:

1. a consistency line (per-refit median persistence and the largest single value), to be compared
   with ``exploratory_4_4.py`` (median 0.9543/0.9827/0.9921, largest single 1.0000);
2. per asset: the number of refits at or above ``1 - 1e-4`` ("at the bound"), the first and last
   such refit date, and the asset's persistence range;
3. for the worst asset, its trajectory (alpha, beta, persistence, omega, nu) at every refit;
4. a sensitivity check: every at-the-bound fit is redone cold (default starts) and the persistence
   and log-likelihood compared with the warm-started one (a warm-start artefact would show as a cold
   fit with a clearly higher likelihood or a persistence well below the bound).

Prints only; nothing is written. **A diagnostic of the estimator, not a result about the ETFs.**
Interpretation rule fixed before the run: a persistence at the cap means the fitted variance process
is (numerically) integrated, so its unconditional variance is undefined; that is a legitimate GARCH
solution (documented in ``dcc.py``), not a failure, and the recursion and the standardised residuals
stay finite. It is acted on only if a cold refit finds a clearly better likelihood (then the warm
start is the problem), or if the bound is hit in most refits of one asset (then a constrained or
different marginal for that asset is a design question for Filippo).
"""

import numpy as np
import pandas as pd
import yaml

from etfcorr.config import find_repo_root
from etfcorr.correlation import dcc

NEAR_BOUND = 1.0 - 1e-4

root = find_repo_root()
with open(root / "config" / "universe.yaml") as f:
    universe = yaml.safe_load(f)
core_tickers = [t["ticker"] for t in universe["tickers"] if t["panel"] == "core"]

returns_all = pd.read_parquet(root / "data" / "processed" / "returns_D.parquet")
if isinstance(returns_all.columns, pd.MultiIndex):
    returns_all = returns_all.xs(returns_all.columns.levels[-1][0], axis=1, level=-1)
missing = [t for t in core_tickers if t not in returns_all.columns]
if missing:
    raise SystemExit(f"core tickers missing from returns_D.parquet: {missing}")
returns = returns_all[core_tickers]
if returns.isna().any().any():
    raise SystemExit("core returns contain NaN; this script mirrors the complete-panel run only")

x = returns.to_numpy(dtype=float)
index = returns.index
n_rows, n_assets = x.shape
n_trains = list(range(dcc.DEFAULT_MIN_OBS, n_rows + 1, dcc.DEFAULT_REFIT_EVERY))
print(
    f"core panel N={n_assets}, {n_rows} daily returns {index[0].date()} to {index[-1].date()}; "
    f"{len(n_trains)} refits, training rows {n_trains[0]} to {n_trains[-1]}; "
    f"cap {dcc.MAX_PERSISTENCE}"
)

fits: list[list[dcc.GarchFit]] = [[] for _ in range(n_assets)]
for j, n in enumerate(n_trains):
    for i in range(n_assets):
        prev = fits[i][-1] if j > 0 else None
        fits[i].append(
            dcc.fit_garch11(x[:n, i], dist="t", start=prev, n_starts=None if prev is None else 2)
        )

pers = np.array([[g.persistence for g in fits[i]] for i in range(n_assets)])  # (assets, refits)
per_refit_median = np.median(pers, axis=0)
print(
    f"\nconsistency: per-refit median persistence min/median/max "
    f"{per_refit_median.min():.4f}/{np.median(per_refit_median):.4f}/{per_refit_median.max():.4f}; "
    f"largest single {pers.max():.6f}; not converged in "
    f"{sum(not g.converged for row in fits for g in row)} fits"
)

rows = []
for i, ticker in enumerate(core_tickers):
    hit = np.flatnonzero(pers[i] >= NEAR_BOUND)
    rows.append(
        {
            "ticker": ticker,
            "n_at_bound": len(hit),
            "first": index[n_trains[hit[0]] - 1].date() if len(hit) else "",
            "last": index[n_trains[hit[-1]] - 1].date() if len(hit) else "",
            "pers_min": pers[i].min(),
            "pers_median": float(np.median(pers[i])),
            "pers_max": pers[i].max(),
        }
    )
table = pd.DataFrame(rows).set_index("ticker")
print(f"\nrefits with persistence >= {NEAR_BOUND} (of {len(n_trains)}), by asset:")
print(table.sort_values(["n_at_bound", "pers_max"], ascending=False).round(5).to_string())
total_hits = int((pers >= NEAR_BOUND).sum())
print(f"\ntotal (asset, refit) fits at the bound: {total_hits} of {pers.size}")

if total_hits == 0:
    print(
        "no fit is at or above the near-bound threshold; the 1.0000 in the earlier run is rounding."
    )
    raise SystemExit(0)

worst = int(np.argmax((pers >= NEAR_BOUND).sum(axis=1) * 10.0 + pers.max(axis=1)))
print(f"\ntrajectory of {core_tickers[worst]}:")
traj = pd.DataFrame(
    {
        "alpha": [g.alpha for g in fits[worst]],
        "beta": [g.beta for g in fits[worst]],
        "persistence": pers[worst],
        "omega": [g.omega for g in fits[worst]],
        "nu": [g.nu for g in fits[worst]],
    },
    index=pd.Index([index[n - 1].date() for n in n_trains], name="fit_end"),
)
print(traj.round(6).to_string())

print("\nsensitivity: every at-the-bound fit redone cold (default starts)")
print("ticker  fit_end     warm_pers   cold_pers   loglik_cold_minus_warm")
worst_gap = -np.inf
for i, j in zip(*np.nonzero(pers >= NEAR_BOUND), strict=True):
    cold = dcc.fit_garch11(x[: n_trains[j], i], dist="t")
    gap = cold.loglik - fits[i][j].loglik
    worst_gap = max(worst_gap, gap)
    print(
        f"{core_tickers[i]:<7} {index[n_trains[j] - 1].date()}  {pers[i, j]:.6f}   "
        f"{cold.persistence:.6f}   {gap:+.5f}"
    )
print(f"\nlargest log-likelihood gain from a cold refit: {worst_gap:+.5f}")
