# Pre-2014 replication: what was done and what came out

Written 2026-11-01. Covers E5-E8 of `config/experiment.yaml`.

**Read this first.**

- **No H verdict changes.** E5-E7 replicate H2, H5 and H6 on 2007-2013, years no statistic of this
  project had touched. They are a family of their own (Holm over three tests), reported beside H1-H6 and
  never pooled with them. E8 is descriptive: no test, no p-value.
- **Declared before the download.** Design committed and tagged `replication-freeze-v1` (commit `2a26498`)
  before any price before 2014 was read. Three gaps found while writing the script were recorded as an
  amendment (`replication-freeze-v2`, commit `3b62e18`). That amendment was written **after** the download,
  the build and the panel check. What had been seen by then: the download log, the build's validation
  warnings and the first-row dates. No correlation, hedge or portfolio number had been computed. The
  three changes follow from the start dates and the estimation windows, not from the data.
- **Run once.** `scripts/replication_pre2014.py --confirm-run`, behind the same guard as the final runs
  (marker `reports/tables/replication_pre2014_marker.json`, opened at commit `0a3f0d7`, completed).
- **The verdicts were known.** This is a replication of results already seen, not a new hypothesis. The
  2008 crisis is also well studied, so "correlations rise in a crisis" was expected; what was not known is
  whether the volatility share, the long-only H5 result and the hedge gap repeat.

## In one paragraph

All three verdicts repeat. Cross-sector correlation is higher in high-VIX periods (+0.197) and the
volatility adjustment removes all of the rise, so E5 holds after Holm (p = 0.003). Cleaned covariances
again give long-only minimum-variance portfolios slightly *higher* volatility (+0.17 points), as in H5. A
SPY hedge again removes more variance in stress than in calm (+0.21), as in H6, while what it leaves is
much more volatile in stress (32% against 14%). Through the crisis, the sectors decoupled during the
onset (average correlation 0.43 against 0.61 before it, trailing low 0.26 on 2008-08-22) and recoupled
within weeks of Lehman; the highest correlation of the period was in the 2011 euro crisis (0.80), not at
the Lehman peak (0.75).

## Data and panel

- Separate root `replication/pre2014` (universe from 2006-01-01, the 25 core funds and the controls SPY,
  QQQ, IWM, TLT, GLD, HYG and ^VIX); `experiment.yaml` and `hypotheses.lock` are symlinks to the main
  config. Same vendor (yfinance) and same Phase 1 cleaning. The 2014 panel was not touched.
- **Panel rule:** core funds minus SMH and OIH (Merrill Lynch HOLDRS until 2011-12-20, a different
  product), first valid row on or before 2006-07-10. Result, as predicted from memory: **SOXX, IGV, XLE,
  XOP, XLV, IBB, XBI, IHI, IHF, ITA, PPA, GDX, XME** (13 funds, 5 sectors, no clean energy). XOP and XME
  start last (2006-06-22); the complete panel runs 2006-06-23 to 2013-12-31 (1894 days).
- **Stale prices are not a concern.** Share of zero-return days: at most 2.8% (PPA) in the warm-up year
  and 2.4% (PPA) in the window; XLV 2.4% and 2.0%; every other fund below 2%
  (`replication_pre2014_zero_returns.csv`). Several funds traded thinly (median 2013 dollar volume of
  PPA about 0.4M USD, ITA 1.3M), but their closes moved almost every day.

## E5: H2 replicated (supported)

`replication_pre2014_e5_h2.csv`. VIX terciles of 2007-07-02 to 2013-12-31, t-1 labels, 64 cross-sector
pairs, 1637 days, 2000 draws, the step 4.6 rule unchanged.

| Line | Raw high-minus-low | Adjusted | Volatility share |
|---|---|---|---|
| primary (per-asset delta) | +0.197 [+0.120, +0.261] | -0.100 [-0.151, -0.044] | 1.51 [1.18, 2.10] |
| SPY return delta | +0.197 | -0.191 [-0.239, -0.135] | 1.97 |
| VIX log-change delta | +0.197 | +0.084 [+0.013, +0.147] | 0.57 [0.28, 0.91] |
| H2, 2014-2026 (primary) | +0.235 [+0.159, +0.305] | +0.027 [-0.011, +0.064] | 0.88 [0.76, 1.07] |

- **Clause 1 met** (two-sided bootstrap p = 0.0010, the floor at 2000 draws) and **clause 2 met** (share
  above 0.5): supported, and every robustness line (day-t labels, blocks of 60 and 120 days,
  within-sector pairs) gives the same reading.
- **The adjustment overshoots.** A share above 1 means the adjusted difference has the opposite sign: the
  Forbes-Rigobon adjustment over-corrects when the volatility ratio is as large as in 2008 (mean delta
  4.9 in the high tercile, against 0 in the low one). It does not mean "more than 100% explained".
- **The non-volatility part is again not identified:** the adjusted difference is negative on two lines
  and positive on the third.

## E6: H5 replicated (not supported, same opposite sign)

`replication_pre2014_e6_vol.csv`. Long-only minimum-variance portfolios, monthly, the economics block
unchanged but restricted to the daily and weekly cells; all five cells exist from **2008-07-01**
(amendment 1), so E6 covers Lehman onward but not the onset (1386 days).

| | Statistic (cleaned minus sample, annualised volatility) | One-sided p |
|---|---|---|
| E6, 2008-07 to 2013-12 (5 cells) | **+0.0017 [+0.0004, +0.0031]** | 0.994 |
| H5, 2023-2026 (9 cells) | +0.0014 [+0.0005, +0.0023] | 0.997 |

- Per cell: `ledoit_wolf_corr` +0.07 (D) and +0.25 (W) points, `rmt` +0.32 (D) and +0.01 (W), `dcc`
  +0.20 (D). No cell is lower than the sample covariance.
- The reading of H5 holds on a different panel and decade: with a long-only constraint, which already
  shrinks the weights, cleaning the covariance has nothing left to fix. N/T is lower than in the main
  panel (13 funds: 0.05 daily, 0.13 weekly), which makes cleaning matter even less. Follow-up E3 found
  that long-short nonlinear shrinkage does help; it was not repeated here.

## E7: H6 replicated (not supported, same opposite sign)

`replication_pre2014_e7_{hedge,hedge_assets,risk,lasso}.csv`. SPY hedge, 126-day rolling beta, stress =
VIX above its 90th percentile of 2007-07-02 to 2013-12-31 (164 days), calm = at or below the median (821).

| | Removed in stress | Removed in calm | Gap | One-sided p |
|---|---|---|---|---|
| E7 rolling beta (the verdict) | 0.730 | 0.524 | **+0.207 [+0.120, +0.267]** | 1.000 |
| E7 lasso, from 2007-11 (amendment 2) | 0.775 | 0.611 | +0.163 [+0.100, +0.225] | 1.000 |
| H6, 2023-2026 | 0.54 | 0.19 | +0.357 [+0.205, +0.441] | 1.000 |

- Every one of the 13 funds is hedged better in stress (`share_assets_lower_in_stress` = 0).
- **What is left matters more than what is removed:** hedged volatility is **32% in stress against 14% in
  calm** (unhedged 62% against 19%). In 2023-2026 it was 27% against 23%. The hedge removes a larger
  share of a much larger risk and still leaves more than twice the calm risk behind.
- **Read the adjusted gap with care.** After the Forbes-Rigobon adjustment the gap in squared correlation is
  **-0.37 [-0.42, -0.28]** (E1 found about zero on 2023-2026). Taken at face value, that is H6's
  original prediction. But SPY's daily variance in stress is 21 times its calm level here, and E5 shows
  the same adjustment overshooting on the same sample: this is most likely over-correction, not evidence
  that the hedge works worse in stress.

## E8: the crisis phases (descriptive)

`replication_pre2014_e8_phases.csv`, figure `reports/figures/replication_pre2014_coupling.{png,svg}`.
Phase dates fixed with the declaration (event dates set from memory, checked against public records on
2026-11-02: BNP Paribas fund freeze 2007-08-09, Lehman filing 2008-09-15, S&P 500 closing low 2009-03-09,
S&P downgrade of the US 2011-08-05); intervals resample days
within each phase (amendment 3). "Adjusted" is Forbes-Rigobon relative to the pre-crisis phase.

| Phase | Days | Avg correlation | Change vs pre-crisis | Adjusted change | Hedge removes | Hedged vol |
|---|---|---|---|---|---|---|
| pre-crisis (2007-01 to 07) | 145 | 0.614 [0.523, 0.700] | - | - | 0.57 | 11.5% |
| onset (2007-08 to 2008-09-12) | 283 | **0.432** [0.313, 0.550] | -0.18 [-0.33, -0.03] | -0.30 [-0.42, -0.17] | 0.45 | 20.6% |
| peak (Lehman to 2009-03-09) | 121 | 0.750 [0.654, 0.795] | +0.14 [+0.01, +0.24] | -0.30 [-0.37, -0.21] | 0.73 | **34.7%** |
| recovery (to 2011-07) | 604 | 0.609 [0.551, 0.668] | -0.01 [-0.11, +0.11] | -0.14 [-0.24, -0.03] | 0.62 | 15.5% |
| euro (2011-08 to 12) | 107 | **0.803** [0.774, 0.830] | +0.19 [+0.10, +0.28] | -0.06 [-0.14, +0.03] | **0.81** | 16.3% |
| after (2012-13) | 502 | 0.546 [0.501, 0.585] | -0.07 [-0.16, +0.03] | -0.10 [-0.18, -0.01] | 0.57 | 13.5% |

Trailing 63-day coupling: high 0.72 on 2007-05-25, low **0.26 on 2008-08-22**, back to 0.81 by
2008-12-16; 0.80 on 2010-08-18 (after the May 2010 flash crash); highest **0.84 on 2011-12-20**.

Readings (**[Inference]**, not tested):

- **The onset decoupled the sectors.** Coupling fell through the first year of the crisis and was at its
  lowest of the whole period three weeks before Lehman. A likely cause is the 2007-2008 commodity
  boom and its reversal, with energy and mining funds moving against the rest; the sector-level
  correlations were not computed to check this.
- **The recoupling was abrupt.** From 0.26 to above 0.7 in about two months around Lehman.
- **The euro crisis was the most coupled phase,** above the Lehman peak, and the SPY hedge removed the
  most variance then (81%), with hedged volatility of 16% against 35% at the peak: high correlation with
  moderate volatility is where a market hedge works best.
- **Every adjusted change is zero or negative:** the raw rises are volatility, as in H2 and E5, with the
  same caveat that the adjustment over-corrects in the most volatile phases.

## Holm over E5-E7

| | Own rule | p | Holm p (m = 3) | Holds |
|---|---|---|---|---|
| E5 (H2) | supported | 0.0010 | 0.0030 | **yes** |
| E6 (H5) | not supported | 0.9940 | 1.0000 | no |
| E7 (H6) | not supported | 1.0000 | 1.0000 | no |

## What to hold against these results

- **13 funds, 5 sectors, no clean energy;** the main panel has 25 funds and 6 sectors, so E5 is not the
  same average as H2.
- **Survivorship is stronger than in the main study:** the funds were chosen in 2026, so the panel holds
  funds that survived the crisis.
- **E6 covers 2008-07 onward** and only the daily and weekly cells; the onset is not in it.
- **The amendment came after the download** (see the top); the panel check and the validation
  warnings were the only data seen before it.
- **E5's share and every adjusted number** rest on the Forbes-Rigobon adjustment, which over-corrects on
  this sample (shares above 1).
- **The E8 phase dates were set from memory** (checked afterwards, all four correct), and the commodity-boom
  reading is consistent with Büyükşahin and Robe (2011) but not tested here; E8 is a description.

## What changed in the repository

- `config/experiment.yaml`: E5-E8 and the amendment. `replication/pre2014/`: the separate root (config and
  download log committed; vendor data ignored).
- `scripts/replication_pre2014.py`, `tests/unit/test_replication_script.py` (6 tests on synthetic data).
- Outputs: `reports/tables/replication_pre2014_*`, `reports/figures/replication_pre2014_coupling.*`.
