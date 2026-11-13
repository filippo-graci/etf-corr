"""Diagnostic for the conservative raw lead-lag statistic (Phase 4, step 4.5b). Run with:
``uv run python scripts/diagnose_4_5b.py`` (``--quick`` = 20 shuffles and 300 bootstrap draws, a
smoke run, not a result; ``--synthetic clean|spiked`` replaces the real returns by a simulated
panel and needs no data: ``spiked`` plants four consecutive alternating crash days).

Reads ``data/processed/returns_D.parquet`` and ``config/universe.yaml`` (core panel) and
``config/experiment.yaml`` (splits). **Only the training window 2016-01-01..2020-12-31 is
analysed**; rows after it are sliced away right after loading and the script asserts it. Prints
only; nothing is written. Exploratory, descriptive, no hypothesis is tested or changed.

Question (docs/checks/4.5.md, Gate 2 results, point 6): the raw cross-autocorrelation scan had 10 of
3000 cells below p = 0.05 and the asymmetry scan 46 of 1500, against 5% expected per cell. Two
untested explanations: H_dep (the raw cells are one common-factor statistic, so their share below
0.05 has a very wide null distribution and 10 may be ordinary) and H_tail (a few clustered extreme
days inflate the bootstrap SE, so the normal p-values are conservative).

Specification fixed before this script was run on real data (src/etfcorr/correlation/
lead_lag_diagnostics.py, module docstring): the same scan settings as step 4.5 (lags 1..5, window
standardisation, stationary bootstrap with the data-driven block); ``N_SHUFFLES`` row-shuffled
copies for the null distribution of the count of cells with p < 0.05 (same bootstrap draws for the
real window and the copies, so counts are comparable); winsorising at +-4 sample standard
deviations as the single sensitivity variant; the verdict is the printed output of
``classify_conservatism`` (H_dep if the real count is inside the central 90% of the null; H_tail
only if it is below the 5th percentile, the real median SE exceeds the shuffled one by more than
20% and winsorising moves the count towards the null median; otherwise unexplained).

Theory-derived expectations (not results): in the shuffled copies the mean count is about 5% of the
cells (75 asymmetry, 150 raw) but the spread is large for raw (its cells are strongly dependent);
the shuffled median SE is close to the i.i.d. value ``1/sqrt(n)`` for raw and ``sqrt(2)/sqrt(n)``
for the asymmetry, times a factor above 1 for fat tails.
"""

import sys
import time

import numpy as np
import pandas as pd

from etfcorr.config import find_repo_root, load_experiment, load_universe
from etfcorr.correlation import lead_lag as ll
from etfcorr.correlation import lead_lag_diagnostics as dg

QUICK = "--quick" in sys.argv
SYNTHETIC = sys.argv[sys.argv.index("--synthetic") + 1] if "--synthetic" in sys.argv else None
N_SHUFFLES = 20 if QUICK else 100
BOOT = 300 if QUICK else 500
SEED = 5
WINS_K = 4.0
STATS: tuple[ll.Statistic, ...] = ("raw", "asymmetry")


def load_discovery() -> pd.DataFrame:
    if SYNTHETIC is not None:
        if SYNTHETIC not in ("clean", "spiked"):
            raise SystemExit("--synthetic takes 'clean' or 'spiked'")
        frame = ll.simulate_lead_lag(1259, 25, seed=21)
        if SYNTHETIC == "spiked":
            sd = frame.std().to_numpy()
            for i, day in enumerate((400, 401, 402, 403)):
                frame.iloc[day] = frame.iloc[day] + 15.0 * sd * (1.0 if i % 2 == 0 else -1.0)
        return frame
    root = find_repo_root()
    exp = load_experiment(root / "config" / "experiment.yaml")
    uni = load_universe(root / "config" / "universe.yaml")
    core = [t.ticker for t in uni.tickers if t.panel == "core"]
    path = root / "data" / "processed" / "returns_D.parquet"
    if not path.is_file():
        raise SystemExit("data/processed/returns_D.parquet not found (run step 2.1 first)")
    frame = pd.read_parquet(path)
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.xs(frame.columns.levels[-1][0], axis=1, level=-1)
    missing = [t for t in core if t not in frame.columns]
    if missing:
        raise SystemExit(f"core tickers missing from returns_D.parquet: {missing}")
    train = (exp.splits.train.start.isoformat(), exp.splits.train.end.isoformat())
    disc = ll.slice_window(frame[core], *train)
    del frame
    if disc.index.max() > pd.Timestamp(train[1]):
        raise SystemExit("the window reaches past the training end; refusing to continue")
    if disc.isna().any().any():
        raise SystemExit("the discovery window has NaN; lags need a full panel")
    return disc


def count_and_se(frame: pd.DataFrame, statistic: ll.Statistic) -> tuple[int, float]:
    scan = ll.lead_lag_scan(frame, statistic=statistic, n_bootstrap=BOOT, seed=SEED)
    return (
        int((scan.table["p_value"] < scan.alpha).sum()),
        float(np.median(scan.table["se"].to_numpy(dtype=float))),
    )


smoke = "QUICK SMOKE RUN, NOT A RESULT. " if QUICK else ""
src = f"SYNTHETIC '{SYNTHETIC}' PANEL (not real data). " if SYNTHETIC else ""
print(f"EXPLORATORY DIAGNOSTIC, NOT A HYPOTHESIS TEST. {src}{smoke}")
disc = load_discovery()
n, p = disc.shape
print(f"discovery window: {n} rows x {p} assets; shuffles {N_SHUFFLES}; B={BOOT}; seed {SEED}")
t0 = time.perf_counter()

print("\n--- A. THE PANEL")
kurt = dg.excess_kurtosis(disc)
print(
    f"excess kurtosis per asset: median {kurt.median():.1f}, max {kurt.max():.1f} ({kurt.idxmax()})"
)
print("five most extreme days (mean squared standardised return across assets):")
days = dg.top_days(disc, 5)
days[["mean_sq", "ew_return"]] = days[["mean_sq", "ew_return"]].round(3)
print(days.to_string(index=False))
print("autocorrelation of the equal-weight average (common-factor proxy), stationary bootstrap:")
print(dg.factor_autocorrelation(disc, n_bootstrap=BOOT, seed=SEED).round(4).to_string(index=False))
print("(raw cells ~ loadings x this autocovariance + idiosyncratic terms: if all |z| are small,")
print(" the whole raw family is expected to be small together)")

print("\n--- B. CONCENTRATION: share of sum z^2 carried by the 5 largest days, over cells")
rng = np.random.default_rng(0)
refs = {
    "real": disc,
    "shuffled (seed 0)": dg.shuffle_rows(disc, 0),
    "iid Gaussian": pd.DataFrame(rng.standard_normal((n, p)), columns=disc.columns),
}
print("statistic   data               median   90th pct   max")
for stat in STATS:
    for name, frame in refs.items():
        z, _ = ll.lagged_products(ll.standardise(frame), ll.DEFAULT_LAGS, stat)
        sh = dg.top_day_share(z, 5)
        print(
            f"{stat:10s}  {name:18s} {np.median(sh):6.3f}   {np.quantile(sh, 0.9):7.3f}   "
            f"{sh.max():6.3f}"
        )

print(f"\n--- C. COUNT OF CELLS WITH p < 0.05: real versus {N_SHUFFLES} row-shuffled copies")
wins = dg.winsorise_standardised(disc, WINS_K)
rows = []
for stat in STATS:
    real_c, real_se = count_and_se(disc, stat)
    wins_c, wins_se = count_and_se(wins, stat)
    null = dg.shuffle_null(disc, statistic=stat, n_shuffles=N_SHUFFLES, n_bootstrap=BOOT, seed=SEED)
    q = null.quantiles((0.0, 0.05, 0.5, 0.95, 1.0))
    le, ge = dg.empirical_position(null.counts, real_c)
    se_ratio = real_se / float(np.median(null.median_se))
    print(
        f"{stat:9s}: tests {null.n_tests}; real {real_c}; null min/5%/median/95%/max "
        f"{q[0.0]:.0f}/{q[0.05]:.0f}/{q[0.5]:.0f}/{q[0.95]:.0f}/{q[1.0]:.0f}; "
        f"P(null<=real) {le:.3f}, P(null>=real) {ge:.3f}"
    )
    print(
        f"           median SE real {real_se:.4f}, shuffled {np.median(null.median_se):.4f} "
        f"(ratio {se_ratio:.2f}); winsorised +-{WINS_K:g}: count {wins_c}, median SE {wins_se:.4f}"
    )
    verdict = dg.classify_conservatism(
        real_c, null.counts, se_ratio=se_ratio, winsorised_count=wins_c
    )
    rows.append((stat, verdict, se_ratio))

print("\n--- D. VERDICT (the pre-fixed rule; no specification is changed by it)")
for stat, verdict, ratio in rows:
    flag = "YES" if ratio > 1.2 else "no"
    print(f"{stat:9s}: {verdict}")
    print(f"           SE inflated against the shuffled copies (ratio > 1.2): {flag} ({ratio:.2f})")
print(f"\nruntime {time.perf_counter() - t0:.0f}s")
