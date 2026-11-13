"""Real-data look at the causal DCC (Phase 4, step 4.4). Run with:
``uv run python scripts/exploratory_4_4.py`` (see the runtime printed by the script;
``--quick`` uses ``refit_every=252`` and one null shuffle and is a smoke run, not a result).

Reads ``data/processed/returns_D.parquet`` (step 2.1), ``config/universe.yaml`` (core panel and
sectors) and ``data/interim/prices_clean.parquet`` (the VIX level, used only to *describe* the
output, never to fit it). Prints only; nothing is written. **Descriptive, exploratory, not a test
of any hypothesis and not part of the pre-registered H2** (H2 uses the VIX terciles and is run in
its own step).

Specification fixed before this script was ever run on real data (docs/checks/4.4.md):

* primary model: ``rolling_dcc`` with Student-t GARCH(1,1) marginals, symmetric DCC, first fit on
  504 valid rows, refit every 63 rows on an expanding window, defaults otherwise; the printed
  series is the one-step forecast ``R_{t+1|t}`` of the average cross-sector correlation (and the
  21-day-ahead forecast, an approximation, see the module docstring of ``dcc.py``);
* printed variants (not alternatives to choose from afterwards): ADCC (``g``), Gaussian marginals;
* the null: the same pipeline on the real returns with the rows *shuffled* (seeded), which keeps
  every marginal and the contemporaneous cross-section and destroys serial dependence, hence
  volatility clustering and any correlation dynamics; the fitted ``a`` should be near zero there.
  Five shuffles are a rough yardstick, not a p-value.

The comparison with two simple causal estimators of the same quantity (EWMA 0.94 and a 63-day
rolling correlation, both of the average *cross-sector* correlation) is a description of how the
three move together and of their level in each VIX tercile, **not a forecast evaluation** (that is
Phase 5, with a realised target and Diebold-Mariano/Clark-West). DCC correlations are computed on
GARCH-standardised residuals and so are, by construction, free of the part of co-movement that
comes from volatility alone (the Forbes-Rigobon heteroskedasticity bias applies to the raw
rolling and EWMA correlations); the by-tercile table shows how large that difference is here.
"""

import sys
import time

import numpy as np
import pandas as pd
import yaml
from scipy.signal import lfilter

from etfcorr.config import find_repo_root
from etfcorr.correlation import dcc
from etfcorr.regimes import TERCILE_NAMES, vix_terciles

QUICK = "--quick" in sys.argv
N_NULL = 1 if QUICK else 5
REFIT = 252 if QUICK else dcc.DEFAULT_REFIT_EVERY

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

CROSS = dcc.cross_sector_mask(core_tickers, sectors)
HORIZONS = (1, 21)

smoke = "QUICK SMOKE RUN, NOT A RESULT. " if QUICK else ""
print(f"DESCRIPTIVE, EXPLORATORY, NOT A HYPOTHESIS TEST. {smoke}")
print(
    f"core panel N={len(core_tickers)}, {len(returns)} daily returns "
    f"{returns.index[0].date()} to {returns.index[-1].date()}; "
    f"{int(CROSS.sum() // 2)} cross-sector pairs; refit_every={REFIT}"
)


def _normalise(c: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diagonal(c, axis1=-2, axis2=-1))
    return c / (d[..., :, None] * d[..., None, :])


def ewma_cross(x: np.ndarray, lam: float = 0.94, init: int = 63) -> np.ndarray:
    """Average cross-sector EWMA correlation at row t, using rows <= t; NaN for t < init."""
    n = x.shape[1]
    outer = (x[:, :, None] * x[:, None, :]).reshape(len(x), -1)
    s0 = outer[:init].mean(axis=0)
    s, _ = lfilter([1.0 - lam], [1.0, -lam], outer[init:], axis=0, zi=(lam * s0)[None, :])
    corr = _normalise(np.asarray(s).reshape(-1, n, n))
    out = np.full(len(x), np.nan)
    out[init:] = corr[:, CROSS].mean(axis=1)
    return out


def rolling_cross(x: np.ndarray, window: int = 63) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for t in range(window - 1, len(x)):
        out[t] = np.corrcoef(x[t - window + 1 : t + 1].T)[CROSS].mean()
    return out


def fit_line(path: dcc.DccPath) -> str:
    f = path.fits

    def r(col: str) -> str:
        v = f[col].to_numpy(dtype=float)
        return f"{v.min():.3f}/{np.median(v):.3f}/{v.max():.3f}"

    return f"a {r('a')}; b {r('b')}; a+b {r('a_plus_b')} (min/median/max over {len(f)} refits)"


start = time.perf_counter()
primary = dcc.rolling_dcc(returns, dist="t", refit_every=REFIT, pair_mask=CROSS, horizons=HORIZONS)
print(f"\n--- PRIMARY: t marginals, symmetric DCC ({time.perf_counter() - start:.0f}s)")
for line in dcc.dcc_summary_lines(primary):
    print(line)
h1 = primary.avg_corr["h1"].dropna()
h21 = primary.avg_corr["h21"].dropna()
print(
    f"forecast rows {len(h1)} (first {h1.index[0].date()}, last {h1.index[-1].date()}); "
    f"avg cross-sector R_(t+1|t): mean {h1.mean():.3f}, sd {h1.std():.3f}, "
    f"min {h1.min():.3f}, max {h1.max():.3f}"
)
print(
    f"h=21 forecast: mean {h21.mean():.3f}, sd {h21.std():.3f}; "
    f"sd of (h21 - h1) {float((h21 - h1).std()):.3f}"
)
print("parameters by refit (last 6):")
print(primary.fits[["a", "b", "a_plus_b", "nu_median"]].tail(6).round(4).to_string())

print("\n--- SIMPLE CAUSAL ESTIMATORS OF THE SAME QUANTITY (description, not an evaluation)")
x = returns.to_numpy(dtype=float)
if np.isnan(x).any():
    raise SystemExit("core returns contain NaN; the simple baselines need a complete panel")
series = pd.DataFrame(
    {
        "DCC h1": primary.avg_corr["h1"],
        "EWMA 0.94": pd.Series(ewma_cross(x), index=returns.index),
        "rolling 63": pd.Series(rolling_cross(x), index=returns.index),
    }
).dropna()
print(
    f"common rows {len(series)}; levels (mean/sd): "
    + "; ".join(f"{c} {series[c].mean():.3f}/{series[c].std():.3f}" for c in series)
)
print("correlation of the series (levels):")
print(series.corr().round(3).to_string())
print("correlation of the 21-day changes:")
print(series.diff(21).dropna().corr().round(3).to_string())

print("\nlevel by full-sample VIX tercile (VIX only describes; mean of each series on that day)")
terciles = vix_terciles(vix, mode="full_sample").reindex(series.index)
by_t = series.groupby(terciles).mean()
by_t.index = pd.Index([TERCILE_NAMES[int(i)] for i in by_t.index], name="VIX tercile")
by_t["count"] = series.groupby(terciles).size().to_numpy()
print(by_t.round(3).to_string())
spread = by_t.iloc[-1][list(series.columns)] - by_t.iloc[0][list(series.columns)]
print("high minus low tercile: " + "; ".join(f"{c} {v:+.3f}" for c, v in spread.items()))

start = time.perf_counter()
asym = dcc.rolling_dcc(returns, dist="t", asymmetric=True, refit_every=REFIT, pair_mask=CROSS)
print(f"\n--- VARIANT: ADCC ({time.perf_counter() - start:.0f}s)")
for line in dcc.dcc_summary_lines(asym):
    print(line)
both = pd.concat(
    [primary.avg_corr["h1"], asym.avg_corr["h1"]], axis=1, keys=["dcc", "adcc"]
).dropna()
print(
    f"corr of the two series {both.corr().iloc[0, 1]:.4f}; mean abs difference "
    f"{float((both['dcc'] - both['adcc']).abs().mean()):.4f}"
)

start = time.perf_counter()
gauss = dcc.rolling_dcc(returns, dist="normal", refit_every=REFIT, pair_mask=CROSS)
print(f"\n--- VARIANT: Gaussian marginals ({time.perf_counter() - start:.0f}s)")
for line in dcc.dcc_summary_lines(gauss):
    print(line)
both = pd.concat(
    [primary.avg_corr["h1"], gauss.avg_corr["h1"]], axis=1, keys=["t", "normal"]
).dropna()
print(
    f"corr of the two series {both.corr().iloc[0, 1]:.4f}; mean abs difference "
    f"{float((both['t'] - both['normal']).abs().mean()):.4f}"
)

print(f"\n--- NULL: {N_NULL} row-shuffled copies of the real returns, same pipeline (t, symmetric)")
print(f"real : {fit_line(primary)}")
for seed in range(N_NULL):
    rng = np.random.default_rng(seed)
    shuffled = pd.DataFrame(
        returns.to_numpy()[rng.permutation(len(returns))],
        index=returns.index,
        columns=returns.columns,
    )
    null_path = dcc.rolling_dcc(
        shuffled, dist="t", refit_every=REFIT, pair_mask=CROSS, keep_matrices=False
    )
    nv = null_path.avg_corr["h1"].dropna()
    print(f"null {seed}: {fit_line(null_path)}; avg cross-sector sd {nv.std():.3f}")
