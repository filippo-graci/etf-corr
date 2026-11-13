# Thematic ETF correlation: structure, forecasts, portfolios and hedges

*A pre-registered study with two supported hypotheses out of six, one of them a "no improvement" claim.
Draft, 2026-10-11, revised 2026-10-19 (follow-ups) and 2026-11-01 (H4; pre-2014 replication; E9). Research code, not investment advice.*

## Summary

Twenty-five US-listed thematic ETFs in seven sectors, daily data from 2014. Six hypotheses were locked
before any result existed. One held as an effect: correlation across sectors rises in high-volatility
periods, and most of the rise is a volatility effect. A second held only in the weak sense its wording
allows: a temporal graph neural network did not beat LightGBM with graph features (H4). The other four
failed, three of them with the opposite sign. For forecasting, portfolio construction and hedging, the simplest standard method was
never beaten on a test window that was read once. Follow-ups run after the verdicts show that the three
reversals are not findings in their own right: one is a test that could not have come out the other way
(H1), one is the H2 volatility effect seen through a hedge (H6), and one is specific to the long-only
design (H5). A replication on 2007-2013, declared before those years were downloaded, repeats the H2, H5
and H6 verdicts on a 13-fund panel.

## Data and design

- **Universe.** 25 core ETFs with a common history from 2014 and a liquidity screen; 9 macro controls.
- **Splits.** Train 2016-2020, validation 2021-2022, test 2023-01 to 2026-09. The test window stayed
  sealed for every forecast, portfolio and hedge until the model list, the comparators and the verdict
  rules were committed and tagged. H1 and H2, descriptive statistics with no fitted model, used all the data
  through 2026-09-25 and were computed before that freeze.
- **Inference.** Alpha 0.05, stationary bootstrap with 2000 draws, Holm correction over the six
  hypotheses. Forecasts use expanding walk-forward folds with a purge of the horizon plus five days.
- **Leakage control.** Every function that turns data into a feature, a target, a graph, a weight or a
  hedge ratio is checked by truncation and future-replacement detectors, and each detector is shown to
  catch a leaky twin.

## Results

| | Hypothesis | Verdict | p | Holm p (m = 6) | Holds after Holm | Window | Caveat |
|---|---|---|---|---|---|---|---|
| H1 | After removing the first eigenmode (market mode), the correlation structure is more similar across horizons (D/W/M) than before removal. | **not supported** | 0.9980 | 1.0000 | no | 2014-01 to 2026-09 (all data), in-sample | significant in the opposite direction (two-sided p 0.004, 0.001, 0.001), but the statistic falls the same way on simulated panels with a horizon-invariant truth (follow-up E2): the test could not tell the directions apart; p derived for the hypothesised direction |
| H2 | Average cross-sector correlation is higher in high-VIX regimes, but a large part of the increase is explained by volatility (Forbes-Rigobon bias) rather than by a true change in dependence. | **supported** | 0.0010 | 0.0060 | yes | 2014-01 to 2026-09 (all data), in-sample | descriptive; the p-value is at the floor of 2000 bootstrap draws; the volatility clause is a pre-fixed threshold, not a test |
| H3 | A machine-learning or graph model beats the best simple baseline (persistence, EWMA, HAR-style, constant correlation) for the 1-month-ahead average cross-sector correlation. | **not supported** | 0.3781 | 1.0000 | no | test 2023-2026, read once | DM-HLN over two challengers; Clark-West gives 0.016 for elastic net against EWMA, but rejects 74% of the time when the baseline is merely badly scaled (follow-up E4) |
| H4 | A temporal graph neural network does not significantly beat a LightGBM model that uses graph-derived features. | **supported** | 0.5966 | 1.0000 | yes | test 2023-2026, read once (2026-11-01) | a non-superiority claim: supported because the network did not beat LightGBM-graph (it was worse, +0.0008); a failure to detect a difference, not proof of equality; designed after the H3 test window had been read |
| H5 | Shrinkage, RMT-cleaned and DCC-forecast covariance estimates give lower out-of-sample realized volatility for minimum-variance portfolios than the raw sample covariance, especially at monthly and quarterly horizons. | **not supported** | 0.9970 | 1.0000 | no | test 2023-2026, read once | the difference has the opposite sign; long-only monthly portfolios of 25 ETFs, where the constraint already shrinks; long-short, nonlinear shrinkage lowers volatility (follow-up E3, on seen data) |
| H6 | Hedge effectiveness (variance reduction) is lower in the worst-decile stress periods than in calm periods. | **not supported** | 1.0000 | 1.0000 | no | test 2023-2026, read once | the difference has the opposite sign, but it is a volatility effect: about zero after a Forbes-Rigobon adjustment, and hedged volatility is higher in stress (follow-up E1); stress is 93 days in a few clusters |

**H1 (structure across horizons): not supported, reversed.** After the market mode is removed, the
leading eigenvectors of daily, weekly and monthly correlation matrices overlap less, not more
(for example 0.998 to 0.944 for the daily-weekly pair). *Follow-up (E2, exploratory):* on 300 simulated
panels with the real daily correlation matrix and a structure that is identical at every horizon, the
same statistic falls by 0.10 on average (daily-monthly), more than in the data (0.014; the follow-up
uses the 25-ETF core panel, the verdict used the 23-ETF panel of Phase 3). After the market
mode is removed the remaining eigenvalues are close together, so single eigenvectors are unstable: the
pre-registered test could not separate the two directions. Statistics that do not depend on eigenvalue
order point the way H1 expected (adjusted Rand index of the sector-sized cluster partitions, daily against
monthly, 0.36 before and 0.77 after removal; a five-dimensional subspace overlap +0.11), at the 90th to
97th percentile of the simulated panels, with bootstrap intervals that include zero.

**H2 (regimes): supported.** Average cross-sector correlation is +0.235 [+0.159, +0.305] higher in the
top VIX tercile than in the bottom one; 88% [76%, 107%] of that is accounted for by the volatility
adjustment. In-sample, descriptive. *Declared follow-up (E9):* the adjustment cannot say whether anything
beyond volatility changed, so the question was asked again with nine common factors (Fama-French five,
momentum, oil, gold, long Treasuries). Within each VIX tercile the average correlation splits exactly into
a factor part and a residual part. On 2014-2022 the factors carry 99% [96%, 102%] of the high-VIX rise,
and the residual correlation is about 0.01 in both terciles (+0.009 [-0.009, +0.028] higher in the high
one, one-sided p = 0.16, not supported); 2023-2026 reads the same. Across sectors, the stress rise is
common-factor co-movement, its factor variances and betas, and nothing measurable beyond it.

**H3 (forecasting): not supported.** Target: the average cross-sector correlation over the next 21
trading days. Ladder: naive baselines, HAR and DCC, four penalised linear models, LightGBM, and LightGBM
with features from seven graphs built inside each training window. On the test window (910 origins,
about 43 independent months) elastic net (0.0334) and the kNN-graph LightGBM (0.0343) are below EWMA
(0.0411) but not separably (one-sided DM-HLN p = 0.189 and 0.240; Bonferroni 0.378), and above DCC
(0.0320). The Model Confidence Set keeps all 18 models. Against 200 pipelines fed random persistent
features, DCC, HAR and PCA regression sit near the 4th percentile, elastic net at the 9th, and EWMA at
the 95th: EWMA was worse than noise on this window. The validation ranking did not carry over (elastic
net: first of 18, then sixth), and neither did a small validation edge for graph features.
*Follow-up (E4, exploratory):* EWMA with decays of 0.97-0.995, or with the decay chosen on each training
window, is no better on the test window (0.0395-0.0423). Simulated at these sample sizes with a challenger
that holds no extra information, Clark-West rejects 7% of the time at a nominal 5% when the baseline is
optimal, and 74% when the baseline overreacts by a factor of two; DM-HLN rejects 0% and 8%. The Clark-West
0.016 is best read as a badly scaled baseline, which EWMA losing to the historical mean also shows.

**H4 (graph neural network): supported, as a non-superiority claim.** Declared and frozen on 2026-11-01,
after the H3 test window had been read, and run once on the same window and folds. The network: eight
features per fund at eight dates five days apart; at each step a graph convolution over the fold's graph
(built from training rows only) feeds a GRU cell per fund; the mean fund state and the 17 common features
go through a small MLP. Five seeds, early stopping on the end of each training window. Against
`lightgbm_graph_knn`: 0.0350 against 0.0343, one-sided p = 0.60, so the network did not beat it. Every graph gave the same
loss (kNN 0.0350, sector 0.0350, random 0.0350, no graph 0.0351): message passing added nothing, and the
network was worse than DCC (0.0320, one-sided p = 0.91 for the network being better) and close to the
historical mean (0.0346). The MCS keeps all ten models compared. Rehearsal on validation had given the same
ordering (0.0332 against 0.0317).

**H5 (portfolios): not supported, reversed.** Long-only minimum-variance portfolios, rebalanced
monthly. Averaged over nine cells (three cleaned estimators, four estimation frequencies), cleaned
covariances gave +0.14 points of annualised volatility [+0.05, +0.23] against the sample covariance
(11.4% at the daily frequency). The gap is no larger at monthly and quarterly frequencies (+0.07
[-0.08, +0.22]). One cell goes the hypothesised way: correlation shrinkage with 32 quarters of data,
-0.31 [-0.57, -0.03], the noisiest setting, uncorrected. For HRP portfolios EWMA lowers volatility at every
frequency and RMT at the monthly and quarterly ones (intervals excluding zero); HRP was not the pre-registered portfolio.
*Follow-up (E3, exploratory, partly on the seen window):* the long-only constraint already shrinks
(Jagannathan and Ma 2003), and Ledoit and Wolf (2017, Table 8) found the sample covariance best by small
margins in the same long-only backtest on US stocks. With shorting allowed, the Ledoit-Wolf (2020) nonlinear estimator lowers
volatility against the sample covariance on 2021-2022 (-0.50 points [-0.73, -0.27]) and on 2023-2026
(-5.1 [-6.7, -3.6]), with a gain that grows with N/T (-0.1 points at 0.10, -2.7 at 0.42, -17 at 0.78,
where the sample-covariance portfolio runs 12x gross leverage; the pattern of Ledoit and Wolf's (2017)
unconstrained backtest, with gains 40-70% larger at matched N/T); eigenvalue clipping (RMT) raises it by
2.6-5.4 points at the daily, weekly and monthly frequencies. A 41-fund panel adds little; a full DCC-GARCH
covariance is no better than DCC with EWMA variances.

**H6 (hedging): not supported, reversed.** Each ETF hedged with SPY at a 126-day rolling beta. The hedge
removes 54% of variance on the 93 highest-VIX days and 19% on the 468 calmest, a difference of +0.36
[+0.20, +0.44], with the same sign for all 25 ETFs and for EWMA, static and sparse multi-instrument
hedges. *Follow-up (E1, exploratory):* the difference is the Forbes-Rigobon bias applied to hedging.
Labelling stress by the previous day's VIX or the previous month's SPY volatility leaves it at +0.32 and
+0.29 (volatility is persistent); adjusting the in-regime correlation for the louder market brings it to
-0.04 [-0.08, +0.04] (2016-2022: -0.17 [-0.20, -0.13]). Hedged positions are more volatile in stress, 27%
against 23% annualised (2016-2022: 38% against 19%; energy and mining funds 55-62%).

**Replication on 2007-2013 (E5-E8).** The three economic hypotheses were re-run on years no statistic of
the project had touched. The design was committed and tagged before any price before 2014 was
downloaded; three gaps found while writing the script (below) were recorded after the download and a
check of the funds' first trading dates, before any statistic was computed. The panel is the 13 core
funds that traded by mid-2006 (no clean energy; SMH and OIH left out because they were different
products, HOLDRS baskets, until December 2011). The three tests form a Holm family of their own.

| | Original | 2007-2013 | Holm p (m = 3) |
|---|---|---|---|
| E5 = H2: high-minus-low VIX correlation | +0.235 [+0.159, +0.305]; volatility share 0.88 | +0.197 [+0.120, +0.261]; share 1.51 | 0.003, holds |
| E6 = H5: cleaned minus sample, volatility points | +0.14 [+0.05, +0.23] | +0.17 [+0.04, +0.31] | 1.000 |
| E7 = H6: variance removed, stress minus calm | +0.357 [+0.205, +0.441] | +0.207 [+0.120, +0.267] | 1.000 |

All three verdicts repeat, the two reversals with the same sign and a similar size. In E5 the
volatility adjustment overshoots (a share above 1: the adjusted difference turns negative), which the
4.6 rule reads as supported and flags. E6 starts in July 2008, the first month at which the weekly
estimation window and the DCC have enough history, and uses the daily and weekly cells only. In E7 the
hedged volatility is 32% in stress against 14% in calm. The volatility-adjusted gap in E7 is -0.37
[-0.42, -0.28], the direction H6 predicted, but SPY's variance in stress is 21 times its calm level and
the same adjustment over-corrects in E5, so it is not read as evidence for H6. *Descriptive (E8):* through
the crisis, average cross-sector correlation was 0.61 before it, 0.43 during its first year (a 63-day low
of 0.26 three weeks before Lehman), 0.75 from Lehman to March 2009 and 0.80 in the 2011 euro crisis, the
most coupled phase; every volatility-adjusted change against the pre-crisis level is zero or negative.
Tables and limits: `docs/REPLICATION_PRE2014_RESULTS.md`.

## Reading

The results fit one picture. A single market factor carries most of the common variation, and its share
grows when volatility rises (H2); with nine common factors, nothing measurable is left to rise between
sectors (E9). That one fact explains the hedging result: a market hedge removes a
larger share of a larger variance in stress, while the risk left over is still higher (H6). What is left
after the market factor is small and noisy, which is why its eigenvectors cannot be compared reliably
across horizons (H1), why more flexible forecasters, a graph neural network included, do not beat standard
ones (H3, H4), and why cleaning the covariance does nothing for a long-only portfolio of 25 assets (H5). Cleaning does matter once the
portfolio may go short, where estimation error is no longer hidden by the constraint. The same picture
holds on 2007-2013, a window that includes the 2008 crisis: correlation rises with volatility, a long-only
portfolio gains nothing from cleaning, and a market hedge removes a larger share of a much larger risk.

## Limits

- One test window and one universe for the forecasting hypotheses. The nulls are "not detected here",
  with wide intervals. The economic verdicts (H2, H5, H6) repeat on 2007-2013 with a smaller panel.
- The H3 verdict depends on the test: DM-HLN was fixed before the test window was read but after
  validation p-values were seen; the uncalibrated Clark-West statistic gives 0.016 for elastic net
  against EWMA. The comparison with DCC does not depend on that choice.
- H4 was designed after the test window had been read for H3, with the comparator's test loss known; it
  was frozen on validation and run once. Its support is a non-rejection with about 43 independent months.
- The H5 and H6 designs involve choices not fixed by the pre-registration (long-only, monthly, a single
  hedge instrument, stress as the top VIX decile of the window). One estimator was replaced on
  validation by a rule declared before it was run.
- H2 and H1 are in-sample descriptions of all the data, 2014-01 to 2026-09 (earlier drafts said
  2014-2022, which was wrong; corrected 2026-11-01).
- The follow-ups (E1-E4) were declared before they were run but after every verdict, partly on the
  2023-2026 window that had been read, and over many uncorrected comparisons. They explain results; they
  confirm nothing. Full tables: `docs/FOLLOW_UPS_RESULTS.md`.
- The replication (E5-E7) tests results whose verdicts were known, on 13 funds in 5 sectors chosen in
  2026, so the panel holds only funds that survived the crisis. Its design was amended after the
  download, before any statistic; the crisis-phase dates (E8) were set from memory and checked afterwards against public records, and E8 is
  descriptive.
- Vendor close prices; no costs beyond a flat 0, 5 or 10 basis points; no futures-based funds.

## Reproducibility

`config/experiment.yaml` holds every design choice with its date; tags `prereg-v1`, `design-v1` to
`design-v4b-declared`, `final-freeze-v1`, `econ-freeze-v1`, `h4-freeze-v1` and `replication-freeze-v1`/`v2`
mark what was fixed when (in the private research repository; this public copy is a single-commit
snapshot, see the README). The test-window commands (forecasting,
economics, H4) and the replication run ran once each and are refused afterwards.
