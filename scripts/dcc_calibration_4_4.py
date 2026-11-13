"""Synthetic calibration of the step 4.4 DCC (offline, no real data). Run with:
``uv run python scripts/dcc_calibration_4_4.py`` (about ten minutes; ``--quick`` is a smoke run
with fewer refits, one seed and fewer Monte Carlo paths, not a result).

All panels have N=25 assets in five sectors of five and 3201 days, the size of the core panel,
Student-t(6) innovations and GARCH(1,1) volatility clustering, with the true correlation
``rbar`` = 0.6 within a sector and 0.3 across (so the true average cross-sector correlation is 0.3).
The estimator is ``rolling_dcc`` with its defaults except ``refit_every=252`` (about 11 refits
instead of 42, to keep the script short).

1. **NULL: constant correlation** (``a = b = 0``). Volatility clusters and tails are fat but the
   correlation never moves. Printed: the fitted ``a`` and ``b`` over refits and the range of the
   forecast average cross-sector correlation. This is the "synthetic noise gives about zero skill"
   check in its correlation-dynamics form: any movement in the path is estimation noise.
2. **PLANTED time-varying correlation** (``a = 0.03, b = 0.95``). Printed: the recovered ``a, b``;
   and the root-mean-square error of the one-step forecast of the average cross-sector
   *conditional* correlation (known in a simulation) for the DCC against three simple causal
   baselines (EWMA with lambda 0.94, 63-day rolling, expanding sample correlation). **This is a
   best case for the DCC (the data come from a DCC) and a sanity check that it can beat the
   baselines when the model is true, not evidence about ETFs.**
3. **Forecast approximation.** The h-step forecast ``Qbar + (a+b)^(h-1) (Q_{T+1} - Qbar)`` replaces
   ``E[z z']`` by ``Q``; the Monte Carlo mean of the average correlation of ``R_{T+h}`` from the
   same starting state shows how large the resulting error is.
4. **Timing** of the rolling fit at N=25.

Properties of the estimator on one data-generating process; nothing is written.
"""

import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from etfcorr.correlation import dcc

QUICK = "--quick" in sys.argv
N_SEEDS = 1 if QUICK else 2
REFIT = 504 if QUICK else 252
N_PATHS = 400 if QUICK else 3000
SIZES = (5, 5, 5, 5, 5)
N_OBS = 3201
N = sum(SIZES)
SECTOR_OF = np.repeat(range(len(SIZES)), SIZES)
RBAR = np.where(SECTOR_OF[:, None] == SECTOR_OF[None, :], 0.6, 0.3)
np.fill_diagonal(RBAR, 1.0)
CROSS = SECTOR_OF[:, None] != SECTOR_OF[None, :]
NU = 6.0


def _run(sim: dcc.DccSimulation) -> tuple[dcc.DccPath, float]:
    start = time.perf_counter()
    path = dcc.rolling_dcc(sim.returns, dist="t", refit_every=REFIT, pair_mask=CROSS)
    return path, time.perf_counter() - start


def _range(values: pd.Series | np.ndarray) -> str:
    v = np.asarray(values, dtype=float)
    return f"min/median/max {v.min():.4f}/{np.median(v):.4f}/{v.max():.4f}"


def _normalise(c: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diagonal(c, axis1=-2, axis2=-1))
    return c / (d[..., :, None] * d[..., None, :])


def rolling_baseline(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for t in range(window - 1, len(x)):
        out[t] = np.corrcoef(x[t - window + 1 : t + 1].T)[CROSS].mean()
    return out


def ewma_baseline(x: np.ndarray, lam: float = 0.94, init: int = 63) -> np.ndarray:
    outer = (x[:, :, None] * x[:, None, :]).reshape(len(x), -1)
    s0 = outer[:init].mean(axis=0)
    s, _ = lfilter([1.0 - lam], [1.0, -lam], outer[init:], axis=0, zi=(lam * s0)[None, :])
    corr = _normalise(np.asarray(s).reshape(-1, N, N))
    out = np.full(len(x), np.nan)
    out[init:] = corr[:, CROSS].mean(axis=1)
    return out


def expanding_baseline(x: np.ndarray, init: int = 63) -> np.ndarray:
    outer = (x[:, :, None] * x[:, None, :]).reshape(len(x), -1)
    s = np.cumsum(outer, axis=0) / np.arange(1, len(x) + 1)[:, None]
    corr = _normalise(s.reshape(-1, N, N))
    out = corr[:, CROSS].mean(axis=1)
    out[: init - 1] = np.nan
    return out


print("1. NULL: constant correlation (a = b = 0), t(6) tails, GARCH volatility clustering")
for seed in range(1, N_SEEDS + 1):
    sim = dcc.simulate_dcc(N_OBS, RBAR, a=0.0, b=0.0, nu=NU, seed=seed)
    path, secs = _run(sim)
    est = path.avg_corr["h1"].dropna()
    print(
        f"  seed {seed}: {len(path.fits)} refits in {secs:.0f}s; truth {0.3:.3f}; "
        f"a {_range(path.fits['a'])}; b {_range(path.fits['b'])}"
    )
    print(
        f"           avg cross-sector forecast: mean {est.mean():.3f}, sd {est.std():.3f}, "
        f"min {est.min():.3f}, max {est.max():.3f}"
    )

print("\n2. PLANTED time-varying correlation (a = 0.03, b = 0.95), t(6), best case for the DCC")
sim = dcc.simulate_dcc(N_OBS, RBAR, a=0.03, b=0.95, nu=NU, seed=11)
path, secs = _run(sim)
print(
    f"  {len(path.fits)} refits in {secs:.0f}s; a {_range(path.fits['a'])}; "
    f"b {_range(path.fits['b'])}; nu median per refit {_range(path.fits['nu_median'])}"
)
x = sim.returns.to_numpy()
truth_next = np.full(N_OBS, np.nan)
truth_next[:-1] = dcc.average_correlation(sim.corr[1:], CROSS)  # R_{t+1}, the target of row t
forecasts = {
    "DCC": path.avg_corr["h1"].to_numpy(),
    "EWMA 0.94": ewma_baseline(x),
    "rolling 63": rolling_baseline(x, 63),
    "expanding": expanding_baseline(x),
}
valid = np.isfinite(forecasts["DCC"]) & np.isfinite(truth_next)
for name, f in forecasts.items():
    ok = valid & np.isfinite(f)
    rmse = float(np.sqrt(np.mean((f[ok] - truth_next[ok]) ** 2)))
    corr = float(np.corrcoef(f[ok], truth_next[ok])[0, 1])
    print(f"  {name:<11} RMSE {rmse:.4f}  corr with truth {corr:.3f}  ({int(ok.sum())} rows)")
print(
    f"  truth over those rows: mean {np.nanmean(truth_next[valid]):.3f}, "
    f"sd {np.nanstd(truth_next[valid]):.3f}, range "
    f"{np.nanmin(truth_next[valid]):.3f} to {np.nanmax(truth_next[valid]):.3f}"
)

print("\n3. Forecast approximation vs Monte Carlo (N=3, start far above the target)")
rb3 = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
q1 = rb3 + 0.3 * (1.0 - np.eye(3))
for a, b in ((0.03, 0.95), (0.10, 0.85)):
    horizons = (1, 5, 21, 63)
    acc = {h: [] for h in horizons}
    for k in range(N_PATHS):
        s = dcc.simulate_dcc(max(horizons), rb3, a=a, b=b, q0=q1, burn=0, seed=1000 + k)
        for h in horizons:
            acc[h].append(float(dcc.average_correlation(s.corr[h - 1])))
    print(f"  a={a}, b={b}: horizon, formula, Monte Carlo mean (s.e.), formula - MC")
    for h in horizons:
        formula = float(dcc.average_correlation(dcc.forecast_correlation(q1, rb3, a, b, h)))
        mc = np.array(acc[h])
        se = float(mc.std(ddof=1) / np.sqrt(len(mc)))
        print(
            f"    h={h:>2}: {formula:.4f}  {mc.mean():.4f} ({se:.4f})  {formula - mc.mean():+.4f}"
        )

print("\n4. Timing")
print(f"  rolling_dcc, N={N}, T={N_OBS}, t marginals, refit_every={REFIT}: {secs:.0f}s")
