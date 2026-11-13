"""Real-data lead-lag scan with FDR control (Phase 4, step 4.5). Run with:
``uv run python scripts/exploratory_4_5.py`` (``--quick`` = 300 bootstrap draws and two row
shuffles: a smoke run, not a result).

Reads ``data/processed/returns_D.parquet`` and (if present) ``returns_W.parquet`` (step 2.1) and
``config/universe.yaml`` (core panel, sectors); splits come from ``config/experiment.yaml``.
Prints only; nothing is written. **Exploratory: not part of the pre-registered H1-H6.**

Specification fixed before this script was ever run on real data (docs/checks/4.5.md):

* the scan is *discovery* on the training window (2016-01-01..2020-12-31) and *replication* on the
  validation window (2021-01-01..2022-12-31). Rows dated in the final test window are sliced away
  right after loading and never analysed; the script asserts it. This deliberately differs from
  steps 4.2-4.4, which described the full sample: a lead-lag table is a candidate feature set for
  Phases 5-6, and a feature chosen on a window that contains the evaluation data would leak;
* primary: asymmetry statistic ``c_ij(k) - c_ji(k)``, window-standardised log returns, lags 1..5
  days (1..4 weeks), stationary bootstrap with ``n_bootstrap`` from the config and the
  data-driven mean block length, BH over the whole family of (pair, lag) cells (BY printed), alpha
  from the config;
* variants printed, not alternatives to choose from afterwards: raw cross-autocorrelation,
  rank transform, mean block length 5 and 60, weekly horizon;
* null yardstick: the same scan on the discovery window with its rows shuffled (seeded), which
  destroys any lead-lag and keeps every marginal and the contemporaneous cross-section. Five
  shuffles are a yardstick, not a p-value.

Theory-derived expectations, to be compared with the output (not results): the family has
``25*24/2*5 = 1500`` daily cells, so about 75 have ``p < 0.05`` by chance alone; with no lead-lag
BH reports 0 discoveries with probability about 0.95; the minimum detectable asymmetry at this n is
about 0.15 (synthetic calibration, scripts/lead_lag_calibration_4_5.py), a lagged cross-correlation
of that size is larger than anything liquid-ETF daily data usually shows, so *finding nothing is a
plausible outcome and is reported as such*.
"""

import sys
import time

import numpy as np
import pandas as pd

from etfcorr.config import find_repo_root, load_experiment, load_universe
from etfcorr.correlation import lead_lag as ll

QUICK = "--quick" in sys.argv

root = find_repo_root()
exp = load_experiment(root / "config" / "experiment.yaml")
uni = load_universe(root / "config" / "universe.yaml")
core = [t for t in uni.tickers if t.panel == "core"]
core_tickers = [t.ticker for t in core]
sectors = {t.ticker: str(t.sector) for t in core}

TRAIN = (exp.splits.train.start.isoformat(), exp.splits.train.end.isoformat())
VAL = (exp.splits.val.start.isoformat(), exp.splits.val.end.isoformat())
TEST_START = exp.splits.test.start
ALPHA = exp.inference.alpha
BOOT = 300 if QUICK else exp.inference.n_bootstrap
N_NULL = 2 if QUICK else 5
SEED_DISC, SEED_VAL = 11, 12


def load_windows(horizon: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    path = root / "data" / "processed" / f"returns_{horizon}.parquet"
    if not path.is_file():
        return None
    frame = pd.read_parquet(path)
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.xs(frame.columns.levels[-1][0], axis=1, level=-1)
    missing = [t for t in core_tickers if t not in frame.columns]
    if missing:
        raise SystemExit(f"core tickers missing from returns_{horizon}.parquet: {missing}")
    frame = frame[core_tickers]
    disc = ll.slice_window(frame, *TRAIN)
    val = ll.slice_window(frame, *VAL)
    del frame  # the rows after the validation window are not used by anything below
    if disc.index.max() > pd.Timestamp(TRAIN[1]) or val.index.max() >= pd.Timestamp(TEST_START):
        raise SystemExit("a window reaches into the final test period; refusing to continue")
    for name, w in (("discovery", disc), ("validation", val)):
        if w.isna().any().any():
            bad = w.columns[w.isna().any()].tolist()
            raise SystemExit(f"{horizon} {name} window has NaN in {bad}; lags need a full panel")
    return disc, val


def counts_line(scan: ll.LeadLagScan) -> str:
    t = scan.table
    cross = t["cross_sector"]
    return (
        f"p<{scan.alpha}: {int((t['p_value'] < scan.alpha).sum())}/{scan.n_tests} "
        f"(chance ~{scan.alpha * scan.n_tests:.0f}); BH {int(t['reject_bh'].sum())}; "
        f"BY {int(t['reject_by'].sum())}; BH within/cross sector "
        f"{int((t['reject_bh'] & ~cross).sum())}/{int((t['reject_bh'] & cross).sum())}"
    )


smoke = "QUICK SMOKE RUN, NOT A RESULT. " if QUICK else ""
print(f"EXPLORATORY, NOT A HYPOTHESIS TEST. {smoke}")
print(f"core panel N={len(core_tickers)}; discovery {TRAIN[0]}..{TRAIN[1]}; validation "
      f"{VAL[0]}..{VAL[1]}; final test window not read; B={BOOT}; alpha={ALPHA}")  # fmt: skip

daily = load_windows("D")
if daily is None:
    raise SystemExit("data/processed/returns_D.parquet not found (run step 2.1 first)")
disc, val = daily
print(f"daily rows: discovery {len(disc)}, validation {len(val)}")

t0 = time.perf_counter()
d = ll.lead_lag_scan(disc, n_bootstrap=BOOT, seed=SEED_DISC, sectors=sectors, keep_null=True)
v = ll.lead_lag_scan(val, n_bootstrap=BOOT, seed=SEED_VAL, sectors=sectors, keep_null=True)
print(f"\n--- PRIMARY, daily, asymmetry ({time.perf_counter() - t0:.0f}s for both windows)")
print("discovery window:")
for line in ll.scan_summary_lines(d):
    print("  " + line)
print("  " + counts_line(d))
cols = ["leader", "follower", "lag", "estimate", "z", "q_bh", "cross_sector"]
print("top 10 cells by |z| (discovery):")
print(d.top(10)[cols].round(4).to_string(index=False))
print("validation window (independent scan, same family):")
print("  " + counts_line(v))
rep = ll.replication_table(d, v)
print(f"discovered cells {len(rep)}; replicated in validation: {int(rep['replicated'].sum())}")
if len(rep):
    show = [
        "asset_a",
        "asset_b",
        "lag",
        "estimate_disc",
        "z_disc",
        "estimate_val",
        "z_val",
        "q_rep",
    ]
    print(
        rep.sort_values("z_disc", key=np.abs, ascending=False)
        .head(10)[show]
        .round(4)
        .to_string(index=False)
    )
agree = ll.agreement_test(d, v)
print(
    f"agreement of the z-statistics across all {agree.n_tests} cells (Spearman, discovery vs "
    f"validation): {agree.statistic:.4f}, one-sided p {agree.p_value:.4f} "
    f"({agree.n_null} null draws)"
)
print("agreement is the only test here with power against many weak persistent effects; a")
print("significant value without any BH discovery means diffuse lead-lag, not a usable cell.")

print("\n--- VARIANTS (discovery window; descriptive)")
for label, kw in (
    ("raw cross-autocorrelation", {"statistic": "raw"}),
    ("rank transform", {"transform": "rank"}),
    ("mean block length 5", {"mean_block_length": 5.0}),
    ("mean block length 60", {"mean_block_length": 60.0}),
):
    s = ll.lead_lag_scan(disc, n_bootstrap=BOOT, seed=SEED_DISC, sectors=sectors, **kw)  # type: ignore[arg-type]
    print(f"{label:28s}: tests {s.n_tests}; " + counts_line(s))
print("(the raw statistic is contaminated by common-factor autocorrelation; compare, do not pick)")

print(f"\n--- NULL YARDSTICK: {N_NULL} row-shuffled copies of the discovery window")
for seed in range(N_NULL):
    rng = np.random.default_rng(seed)
    shuffled = pd.DataFrame(
        disc.to_numpy()[rng.permutation(len(disc))], index=disc.index, columns=disc.columns
    )
    s = ll.lead_lag_scan(shuffled, n_bootstrap=BOOT, seed=SEED_DISC, sectors=sectors)
    print(f"shuffle {seed}: " + counts_line(s))

weekly = load_windows("W")
print("\n--- WEEKLY (Friday close), lags 1..4 weeks")
if weekly is None:
    print("data/processed/returns_W.parquet not found: skipped")
else:
    wd, wv = weekly
    w_lags = (1, 2, 3, 4)
    sd = ll.lead_lag_scan(wd, lags=w_lags, n_bootstrap=BOOT, seed=SEED_DISC, sectors=sectors)
    sv = ll.lead_lag_scan(wv, lags=w_lags, n_bootstrap=BOOT, seed=SEED_VAL, sectors=sectors)
    print(f"weekly rows: discovery {len(wd)}, validation {len(wv)}; minimum detectable "
          f"asymmetry {sd.minimum_detectable:.3f} (discovery)")  # fmt: skip
    print("  discovery: " + counts_line(sd))
    print("  validation: " + counts_line(sv))
    wrep = ll.replication_table(sd, sv)
    print(f"  discovered cells {len(wrep)}; replicated: {int(wrep['replicated'].sum())}")
print(f"\ntotal runtime {time.perf_counter() - t0:.0f}s")
