# Follow-ups: what was done and what came out

Written 2026-10-12. Covers every item of `docs/FOLLOW_UPS.md`.

**Read this first.**

- **No verdict changes.** H1-H6 stay as recorded. Everything here is exploratory, declared as E1-E4 in
  `config/experiment.yaml` before it was run.
- **The 2023-2026 window had already been seen.** Every number on it below is in-sample. Where a 2021-2022 or
  2014-2022 number exists it is shown beside it; those are second samples, not hold-outs.
- **Nothing is corrected for the number of comparisons** (dozens of cells in the H5 part alone).
- **Committed after the runs.** The entries were declared in the config before running but committed only on
  2026-10-19, after the runs, which departs from the usual rule in this repo.

## In one paragraph

Three of the four "surprising" results have a plain explanation. H6's large stress-calm gap is the volatility
effect: after adjusting for it the gap is zero. H1's reversed result is what sampling noise produces even when
the structure is identical at every horizon. H5's null is specific to long-only portfolios: once shorting is
allowed, nonlinear shrinkage lowers volatility and the gain grows with N/T, while eigenvalue clipping makes
things worse. For H3, a slower EWMA is not a better baseline, and the Clark-West p of 0.016 is best read as
"EWMA is badly scaled", not "the model found new signal". Two items were not built (below).

## 1. H6: the hedge is not better in stress, the market is just louder (E1)

`scripts/followup_h6.py`, tables `reports/tables/followup_h6_{summary,sectors}.csv`. The script reproduces the
recorded +0.357 exactly before changing anything.

| Stress label | Window | Gap in variance removed (stress - calm) | Gap after volatility adjustment |
|---|---|---|---|
| same-day VIX (H6 design) | 2023-2026 | +0.357 [+0.205, +0.441] | -0.038 [-0.077, +0.042] |
| previous-day VIX | 2023-2026 | +0.324 [+0.155, +0.416] | -0.043 [-0.085, +0.046] |
| previous 21-day SPY volatility | 2023-2026 | +0.287 [-0.001, +0.400] | -0.011 [-0.059, +0.125] |
| same-day VIX | 2016-2022 | +0.355 [+0.241, +0.412] | -0.168 [-0.202, -0.127] |
| previous-day VIX | 2016-2022 | +0.306 [+0.151, +0.374] | -0.186 [-0.219, -0.143] |

- **Lagging the label barely helps.** Volatility is persistent, so yesterday's VIX still picks loud days. The
  follow-up list expected this to be the main fix; it is not.
- **The volatility adjustment removes the gap entirely.** In 2023-2026 it is indistinguishable from zero; in
  2016-2022 it turns negative, meaning the link to SPY was *weaker* in stress once the louder market is
  accounted for.
- **What a hedger is left with is worse in stress.** Hedged positions ran at 27% annualised volatility on
  stress days against 23% on calm days in 2023-2026, and at 38% against 19% in 2016-2022. The average
  worst-5%-day loss of a hedged ETF on stress days was 3.8% (5.9% unhedged) and 5.6% (9.6% unhedged).
- **By sector (2016-2022, same-day label):** hedged energy stays at 62% volatility in stress and miners at
  55%; semiconductors, defence and healthcare come down to about 20-23%. SPY is a poor stress hedge for the
  commodity sectors.

Practical reading: "54% of variance removed in stress" should not be quoted as hedge quality. Quote leftover
volatility instead. Caveat: the 5% tail on about 90 stress days is four or five observations per ETF.

## 2. H1: the reversed result is a noise artefact (E2)

`scripts/followup_h1.py`, tables `reports/tables/followup_h1_{real,simulated}.csv`. Core 25, 2014-2022 only.
The H1 verdict was computed on the 23-ETF panel of Phase 3 (before IHF and REMX were added), where the
daily-weekly drop was 0.998 to 0.944; on the 25-ETF panel used here it is 0.998 to 0.992.

The key check is a simulation: 300 Gaussian panels of the real size with the real daily correlation matrix,
aggregated to weekly and monthly, so the true structure is **identical at every horizon**.

| Daily vs monthly | Real data | Simulated, structure truly invariant (mean, 5%-95%) |
|---|---|---|
| Leading-eigenvector overlap, after minus before | -0.014 | -0.104 (-0.343 to -0.014) |
| Cluster agreement (ARI), before / after | 0.36 / 0.77 | 0.66 / 0.79 |
| Cluster agreement, after minus before | +0.41 [-0.27, +0.54] | +0.13 (-0.25 to +0.46) |
| 5-dimensional subspace overlap, after minus before | +0.108 [-0.068, +0.146] | -0.034 (-0.151 to +0.089) |

- **The statistic H1 was judged on drops after detoning even when nothing differs across horizons.** The real
  drop (-0.014) is smaller than what noise alone gives on average. So "structure becomes less similar after
  removing the market mode" was never shown; the test could not have come out the other way.
- **The order-robust statistics point the way H1 expected** (clusters agree more across horizons after
  removal, the 5-dimensional subspace overlaps more), at the 90th-97th percentile of the noise world, but
  every bootstrap interval includes zero. Suggestive, not established.
- Weekly vs monthly looks the same (ARI 0.38 to 1.00); daily vs weekly shows no change either way.

Practical reading: H1 stays "not supported", but the write-up should say "the test was uninformative", not
"the opposite was found". One oddity to know: for the eigenvector statistic the bootstrap interval does not
contain its own point estimate (resampled matrices are noisier than the original), which is the same problem
seen from another side.

## 3. H5: shrinkage works where the literature says it should (E3)

`scripts/followup_h5.py`, tables `reports/tables/followup_h5_{cells,pooled}.csv`. The long-only core rows
reproduce the recorded +0.14 points [+0.05, +0.23] exactly, and the DCC path matches step 8.5.

Change in annualised volatility against the sample covariance, in volatility points (negative is better):

| Setting | Validation 2021-2022 | Test 2023-2026 (seen) |
|---|---|---|
| Long-only, nonlinear shrinkage | +0.26 [+0.15, +0.36] | -0.05 [-0.20, +0.12] |
| **Long-short, nonlinear shrinkage** | **-0.50 [-0.73, -0.27]** | **-5.1 [-6.7, -3.6]** |
| Long-short, declared three (LW-corr, RMT, DCC) | +1.95 [+1.35, +2.66] | -3.1 [-4.7, -1.7] |
| Long-short, 41 ETFs, nonlinear | -0.09 [-0.27, +0.12] (daily only) | -0.87 [-1.07, -0.65] |
| Full DCC-GARCH, long-only | +0.17 [-0.32, +0.66] | +0.12 [-0.16, +0.40] |

- **Long-only: nothing to find.** Every estimator is within a few hundredths of a point of the sample
  covariance. The constraint does the shrinking, as Jagannathan and Ma say.
- **Long-short: nonlinear shrinkage helps, more as N/T rises.** On the test window: -0.1 points at N/T 0.10,
  -0.7 at 0.24, -2.7 at 0.42, -17 at 0.78 (where the sample-covariance portfolio runs 12x gross leverage and
  31% volatility). The big pooled number is mostly that last cell.
- **RMT clipping hurts in long-short portfolios:** +2.6 to +5.4 volatility points at daily, weekly and
  monthly frequencies in both windows. It only helps where the sample covariance collapses. This was invisible in H5.
- **The larger universe adds little** at the daily frequency; about -1.5 points at weekly (nonlinear, test window).
- **A full DCC covariance is no better** than DCC with EWMA variances, and neither beats the sample.

Practical reading: H5 as written (long-only, 25 funds) was close to untestable. The honest sentence is
"cleaning the covariance does not matter for long-only portfolios of this size and does matter for
unconstrained ones". The long-short portfolios run 2.5x-6x gross leverage with no borrow cost, so this is a
statement about estimators, not about something to trade.

Deviations from the list: the panel has 41 funds, not 47 (the common history of all 47 starts in 2024; the two
spot-crypto series are not ETFs). skfolio has no nonlinear shrinkage, so the Ledoit-Wolf analytical formula
was written here (30 lines, checked against the formula written as plain loops and on known spectra).

## 4. H3: the baseline was not too fast, and Clark-West is not just oversized (E4)

`scripts/followup_h3.py`, tables `reports/tables/followup_h3_{baselines,clark_west}.csv`.

**Slower EWMA.** Loss on H3's target (lower is better):

| Baseline | Validation | Test (seen) |
|---|---|---|
| EWMA 0.94 (the H3 comparator) | 0.0347 | 0.0411 |
| EWMA 0.97 / 0.98 / 0.99 | 0.0316 / 0.0301 / 0.0316 | 0.0414 / 0.0413 / 0.0402 |
| EWMA, decay chosen on training data | 0.0316 | 0.0423 |
| Historical mean | 0.0414 | 0.0346 |
| DCC | 0.0336 | 0.0320 |
| Elastic net | 0.0292 | 0.0334 |

A slower decay helped on validation and did nothing on the test window; the tuned one picked 0.97 every time
and was the worst EWMA out of sample. Every EWMA lost to the plain historical mean in 2023-2026. The decay
was not the problem.

**Clark-West calibration** (500 simulated runs at the real sample sizes; the larger model only adds noise
features, so it has no information the baseline lacks):

| World | DM-HLN rejects at 5% | Clark-West rejects at 5% | Clark-West p below 0.016 |
|---|---|---|---|
| Baseline is already optimal | 0.0% | 7.0% | 2.4% |
| Baseline moves twice as much as it should | 8.0% | 74% | 55% |

- Clark-West is only mildly oversized (7% for a nominal 5%), so 0.016 is not a fluke of the test.
- But it fires three times in four when the baseline is merely badly scaled, with no new information in the
  model. EWMA losing to the historical mean says it is badly scaled.
- DM-HLN almost never rejects in this setting: the verdict test had very little power.

Practical reading: the H3 verdict hinged on a weak test, and the alternative test detects a flaw in the
baseline rather than skill in the model. The fair statement is that elastic net beat a poor baseline and did
not beat DCC, HAR or the historical mean.

**Not done from this item:** intraday realised covariances (no point-in-time minute-bar source; the list calls
it a blocker) and a longer history (needs a new vendor download, which would change the raw panel every
recorded result was built on; that is Filippo's call).

## 5. H4 (sequence models, temporal GNN): not built here (done on 2026-11-01, see `status/phase_7.md`)

`torch` is not installed in the shared environment, and the list itself says this is only worth doing with a
longer history and a fresh hold-out, neither of which exists. H4 stays "deferred, not attempted".

## 6. Housekeeping

- **`cvxpy` installed; skfolio imports.** The HRP cross-check that had always been skipped turned out to be
  broken as written, and once fixed showed a real difference: skfolio reorders the dendrogram leaves
  optimally, this project uses López de Prado's original order. Weights differ by up to 0.08 on a 6-asset
  panel. With the same ordering the two agree to 1e-8, so it is a convention, not a bug; `hrp_weights` now has
  an `optimal_ordering` switch (default unchanged, so recorded results stand).
- **Futures flags of UUP, DBC and BITO confirmed** (all three hold futures) from search listings of the SEC
  filings and the ProShares prospectus; the full prospectuses were not read. Notes updated in
  `config/universe.yaml`.
- **Mutation testing** (`scripts/mutation_followup.py`, 70 mutations over the Phase 7 and 8 modules and the
  new code): 52 caught at first, **70 of 70 after adding tests**. The gaps that mattered were in the original
  Phase 8 code: nothing checked that the DCC forecast and the monthly lasso refit use only data from before
  the rebalance day, that the design's hedge window and stress quantile actually reach the computation, or
  that a tiny p-value cannot mark an unsupported hypothesis as holding. The code was right in each case; the
  tests could not have noticed if it were not. One real defect was found in the new code (a bootstrap draw
  made of one repeated stress day) and fixed; it did not affect any reported number.
- **Two figures built:** `reports/figures/coupling_followup` (rolling coupling with event dates, 2014-2022;
  peak 0.77 on 2020-03-12, low 0.17 in February 2017) and `reports/figures/mst_regimes_followup` (the tree
  by VIX tercile: 18 of 24 edges shared between low and high VIX, mean edge correlation 0.70 against 0.85).
  The event dates are from memory and unverified.

## What changed in the code

- `evaluation/portfolio.py`: `minvar_ls`, `nonlinear`, `nonlinear_corr`, `dcc_garch`, `optimal_ordering`.
  None is part of the declared design; `etfcorr econ` runs exactly what it ran before.
- `correlation/dcc.py`: `rolling_dcc` also returns the GARCH variance forecasts (`var_next`).
- `evaluation/hedging.py`: `regime_risk`, `regime_risk_table`. `correlation/network_stability.py`:
  `partition_similarity`. `models/baselines.py`: `BestColumn`.
- Scripts: `followup_h1.py`, `followup_h3.py`, `followup_h5.py`, `followup_h6.py`, `figures_followup.py`,
  `mutation_followup.py`.
- Full suite: **1804 passed, 0 skipped** (was 1776 and 1 skipped); ruff, format and mypy clean.
- Neither `etfcorr eval --final` nor `etfcorr econ --final` was run; both markers are untouched.

## Open decisions for Filippo

1. Done on 2026-10-19: the H1, H3, H5 and H6 wording of `README.md`, `paper.md`
   and the scoreboard caveats now carry these readings; verdicts and p-values unchanged.
2. Open: whether to download a longer history (item 4) and install `torch` (item 5).
3. Done on 2026-10-19: committed; the config hash is `8c107ace79d6` because of the four exploratory entries.
