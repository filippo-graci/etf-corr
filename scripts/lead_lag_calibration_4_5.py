"""Synthetic calibration of the lead-lag scan (Phase 4, step 4.5). Run with:
``uv run python scripts/lead_lag_calibration_4_5.py`` (``--quick`` = fewer replications, smoke run).

Uses only simulated data (``simulate_lead_lag``: one-factor model, GARCH(1,1) volatility,
Student-t(5) shocks), so it needs no vendor data and writes nothing; it prints. Every number it
prints comes from the code it runs. Four questions, in the panel geometry of the real scan
(N=25 assets, 1258 daily rows ~ the 2016-2020 training window, lags 1..5, hence 1500 tests):

A. Size. Under the global null (no lead-lag), what share of two-sided p-values falls below 5%, 1%
   and 0.1%, and how often does BH report at least one discovery (should be about 5% or less)?
B. Power. How often is one planted cell (A00 leads A01 at lag 1 with coefficient theta) found
   by BH, as a function of theta; and what is the minimum detectable asymmetry?
C. Raw versus asymmetry statistic when the common factor is autocorrelated (AR(1) 0.1): the raw
   cross-autocorrelation produces many "discoveries" with no lead-lag at all; the asymmetry should
   not (Lo and MacKinlay 1990, mechanism; see lead_lag.py for the citation status).
D. Sensitivity of the size to the mean block length (5, data-driven default, 60).

Replications are independent seeds; with R replications a rate has binomial noise of about
``sqrt(p(1-p)/R)``, so read the printed rates with that in mind. No tuning, no use of the
results to alter the scan's defaults, which were fixed before this script was run.
"""

import sys
import time

import numpy as np

from etfcorr.correlation import lead_lag as ll

QUICK = "--quick" in sys.argv
R = 12 if QUICK else 60  # replications per cell
BOOT = 300 if QUICK else 500  # bootstrap draws per scan (production default: 2000)
N_DAYS, N_ASSETS, LAGS = 1258, 25, (1, 2, 3, 4, 5)


def sim(seed: int, **kw: object) -> "ll.pd.DataFrame":
    return ll.simulate_lead_lag(N_DAYS, N_ASSETS, seed=seed, **kw)  # type: ignore[arg-type]


def scan(frame: "ll.pd.DataFrame", seed: int = 0, **kw: object) -> ll.LeadLagScan:
    return ll.lead_lag_scan(frame, lags=LAGS, n_bootstrap=BOOT, seed=seed, **kw)  # type: ignore[arg-type]


smoke = "QUICK SMOKE RUN, NOT A RESULT. " if QUICK else ""
print(f"SYNTHETIC CALIBRATION. {smoke}R={R} replications per cell, B={BOOT} bootstrap draws")
print(f"panel: N={N_ASSETS}, n={N_DAYS}, lags={LAGS}, tests={N_ASSETS * (N_ASSETS - 1) // 2 * 5}")

t0 = time.perf_counter()
print("\n--- A. SIZE under the global null (GARCH-t, market loading 0.6)")
p_all, any_bh, any_by = [], 0, 0
for r in range(R):
    s = scan(sim(1000 + r), seed=r)
    p_all.append(s.table["p_value"].to_numpy())
    any_bh += int(s.table["reject_bh"].any())
    any_by += int(s.table["reject_by"].any())
p_cat = np.concatenate(p_all)
for level in (0.05, 0.01, 0.001):
    print(
        f"P(p < {level}) = {(p_cat < level).mean():.4f}  (nominal {level}; {p_cat.size} p-values)"
    )
print(f"replications with >=1 BH discovery: {any_bh}/{R}; with >=1 BY discovery: {any_by}/{R}")
print("expected under the null: BH/BY false-discovery frequency <= alpha = 0.05")

print("\n--- B. POWER for one planted cell (A00 leads A01, lag 1, coefficient theta)")
print("theta  implied corr  detected by BH   mean z     minimum detectable (this panel)")
for theta in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
    hits, zs, mdes = 0, [], []
    for r in range(R):
        s = scan(sim(2000 + r, planted=[(0, 1, 1, theta)]), seed=r)
        row = s.table[(s.table["asset_a"] == "A00") & (s.table["asset_b"] == "A01")]
        row = row[row["lag"] == 1].iloc[0]
        hits += int(bool(row["reject_bh"]) and row["estimate"] > 0)
        zs.append(float(row["z"]))
        mdes.append(s.minimum_detectable)
    corr = theta / np.sqrt(1.0 + theta**2)
    print(
        f"{theta:5.2f}  {corr:11.3f}  {hits:3d}/{R:<3d} ({hits / R:5.2f})  {np.mean(zs):7.2f}"
        f"     {np.median(mdes):.3f}"
    )
print("(implied corr = theta/sqrt(1+theta^2) is the lag-1 cross-correlation of the planted pair)")

print("\n--- C. RAW versus ASYMMETRY with an autocorrelated common factor (AR 0.1), no lead-lag")
rej_raw, rej_asym, p05_raw, p05_asym = [], [], [], []
for r in range(R):
    f = sim(3000 + r, factor_ar=0.1)
    a = scan(f, seed=r, statistic="asymmetry")
    w = scan(f, seed=r, statistic="raw")
    rej_asym.append(int(a.table["reject_bh"].sum()))
    rej_raw.append(int(w.table["reject_bh"].sum()))
    p05_asym.append(float((a.table["p_value"] < 0.05).mean()))
    p05_raw.append(float((w.table["p_value"] < 0.05).mean()))
print(
    f"raw      : mean BH discoveries {np.mean(rej_raw):7.1f}; share p<0.05 {np.mean(p05_raw):.3f}\n"
    f"asymmetry: mean BH discoveries {np.mean(rej_asym):7.1f}; share p<0.05 {np.mean(p05_asym):.3f}"
)

print("\n--- D. SIZE versus the mean block length (global null, GARCH-t)")
for block in (5.0, None, 60.0):
    ps = []
    for r in range(max(R // 2, 6)):
        s = scan(sim(4000 + r), seed=r, mean_block_length=block)
        ps.append(s.table["p_value"].to_numpy())
    p_cat = np.concatenate(ps)
    label = "default" if block is None else f"{block:.0f}"
    print(
        f"block {label:>7}: P(p<0.05) {(p_cat < 0.05).mean():.4f}, P(p<0.01) "
        f"{(p_cat < 0.01).mean():.4f}, P(p<0.001) {(p_cat < 0.001).mean():.4f}"
    )
print(f"\nruntime {time.perf_counter() - t0:.0f}s")
